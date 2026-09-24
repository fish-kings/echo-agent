//! demo12_resilience.rs —— 韧性特性开关对比演示

use echo_agent::agent::{Agent, AgentCallback};
use echo_agent::error::ReactError;
use echo_agent::prelude::*;
use echo_agent::testing::MockLlmClient;
use echo_agent::tool;
use echo_agent::tools::{Tool, ToolParameters, ToolResult};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 工具函数：计算两数之和，用于验证工具失败后 Agent 能否切换到备用工具。
#[tool(name = "add", description = "两数相加")]
async fn add(a: f64, b: f64) -> Result<ToolResult> {
    Ok(ToolResult::success(format!("{} + {} = {}", a, b, a + b)))
}

/// demo12 契约测试：对比 `tool_error_feedback` 开关、偶发故障工具自动恢复、
/// 以及 `llm_max_retries` 配置在不同场景下的行为差异。
#[tokio::test]
async fn contract_demo12_resilience() -> Result<()> {
    dotenvy::dotenv().ok();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "echo_agent=warn,demo12_resilience=info".into()),
        )
        .try_init();

    print_banner();

    sep("Part 1: tool_error_feedback = false（旧行为）");
    demo_feedback_off().await?;

    sep("Part 2: tool_error_feedback = true（新行为，默认）");
    demo_feedback_on().await?;

    sep("Part 3: FlakyTool —— 偶发故障后自动恢复");
    demo_flaky_tool().await?;

    sep("Part 4: llm_max_retries 开关对比");
    demo_llm_retry_config()?;

    println!("\n{}", "═".repeat(64));
    println!("  demo12 完成");
    println!("{}", "═".repeat(64));
    Ok(())
}

// ── 测试工具 ──────────────────────────────────────────────────────────────────

/// 模拟总是失败的损坏工具，用于验证 `tool_error_feedback` 关闭时 Agent 会因工具失败而中断。
struct BrokenTool;

impl Tool for BrokenTool {
    /// 返回工具名称 `broken_tool`。
    fn name(&self) -> &str {
        "broken_tool"
    }
    /// 返回工具描述。
    fn description(&self) -> &str {
        "模拟损坏的工具，总是返回失败"
    }
    /// 返回工具入参的 JSON Schema：必填字符串参数 `input`。
    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": { "input": { "type": "string" } }, "required": ["input"] })
    }
    /// 执行时忽略入参，始终返回失败结果，用于模拟服务不可用。
    fn execute(
        &self,
        _params: ToolParameters,
    ) -> futures::future::BoxFuture<'_, echo_agent::error::Result<ToolResult>> {
        Box::pin(async move { Ok(ToolResult::error("BrokenTool: 服务不可用")) })
    }
}

/// 模拟偶发故障、之后自动恢复的天气查询工具：前 `fail_remaining` 次调用返回失败，之后返回成功。
struct FlakyTool {
    /// 剩余需要失败的调用次数，每次调用后递减，归零后开始返回成功结果。
    fail_remaining: Arc<AtomicUsize>,
    /// 累计调用次数，用于日志输出中标注当前是第几次尝试。
    call_count: Arc<AtomicUsize>,
}

impl FlakyTool {
    /// 创建一个新的 `FlakyTool`，前 `fail_times` 次调用会失败，此后调用均成功。
    fn new(fail_times: usize) -> Self {
        Self {
            fail_remaining: Arc::new(AtomicUsize::new(fail_times)),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Tool for FlakyTool {
    /// 返回工具名称 `weather_api`。
    fn name(&self) -> &str {
        "weather_api"
    }
    /// 返回工具描述。
    fn description(&self) -> &str {
        "查询城市天气。服务偶有故障，遇到错误请稍后重试。"
    }
    /// 返回工具入参的 JSON Schema：必填字符串参数 `city`。
    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] })
    }
    /// 执行天气查询：剩余失败次数未归零时返回错误结果，否则返回该城市的模拟天气信息。
    fn execute(
        &self,
        params: ToolParameters,
    ) -> futures::future::BoxFuture<'_, echo_agent::error::Result<ToolResult>> {
        Box::pin(async move {
            let city = params
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("未知");
            let call_idx = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            let remaining = self.fail_remaining.load(Ordering::Relaxed);

            if remaining > 0 {
                self.fail_remaining.fetch_sub(1, Ordering::Relaxed);
                Ok(ToolResult::error(format!(
                    "服务暂时不可用（第 {call_idx} 次尝试）"
                )))
            } else {
                Ok(ToolResult::success(format!("{city}：晴，26°C")))
            }
        })
    }
}

// ── 简易日志回调 ────────────────────────────────────────────────────────────────

/// 简易日志回调：在 Agent 执行各阶段打印带标签的日志，便于对比不同配置下的运行过程。
struct SimpleLog {
    /// 日志前缀标签，用于区分不同场景（如 "NO-FB"、"FB-ON"）打印的日志。
    label: &'static str,
}

impl AgentCallback for SimpleLog {
    /// 每轮迭代开始时打印迭代序号。
    fn on_iteration<'a>(
        &'a self,
        _agent: &'a str,
        iteration: usize,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            println!("  [{}] 🔄 迭代 {}", self.label, iteration + 1);
        })
    }
    /// 工具调用开始时打印工具名与参数。
    fn on_tool_start<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        args: &'a Value,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            println!(
                "  [{}] 🔧 调用: {} args={}",
                self.label,
                tool,
                compact(args)
            );
        })
    }
    /// 工具调用成功结束时打印工具名与截断后的结果。
    fn on_tool_end<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        result: &'a str,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            println!(
                "  [{}] ✅ 成功: {} result=\"{}\"",
                self.label,
                tool,
                trunc(result, 60)
            );
        })
    }
    /// 工具调用失败时打印工具名与错误信息。
    fn on_tool_error<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        err: &'a ReactError,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            println!("  [{}] ❌ 错误: {} err={}", self.label, tool, err);
        })
    }
    /// Agent 得出最终答案时打印截断后的答案文本。
    fn on_final_answer<'a>(
        &'a self,
        _agent: &'a str,
        answer: &'a str,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            println!("  [{}] 🏁 最终答案: \"{}\"", self.label, trunc(answer, 80));
        })
    }
}

// ── Part 1 ──────────────────────────────────────────────────────────────────────

/// 演示 `tool_error_feedback = false`（旧行为）：工具失败会直接中断 Agent 执行。
async fn demo_feedback_off() -> Result<()> {
    println!("  配置：tool_error_feedback = false\n");

    let mock = Arc::new(MockLlmClient::new().then_tool_call(
        "call_broken_once",
        "broken_tool",
        r#"{"input":"trigger failure"}"#,
    ));

    let mut agent = ReactAgentBuilder::new()
        .name("agent_no_feedback")
        .llm_client(mock)
        .system_prompt("你是一个智能助手。请调用 broken_tool 并报告结果。")
        .enable_tools()
        .tool_error_feedback(false)
        .max_iterations(4)
        .callback(Arc::new(SimpleLog { label: "NO-FB" }))
        .build()?;

    agent.add_tool(Box::new(BrokenTool));

    match agent.execute("请调用 broken_tool 并报告结果。").await {
        Ok(answer) => {
            return Err(ReactError::Other(format!(
                "demo12 验收失败：tool_error_feedback=false 时意外成功: {answer}"
            )));
        }
        Err(e) => {
            println!("\n  ✅ 符合预期 —— Agent 因工具失败而中断: {e}");
        }
    }
    Ok(())
}

// ── Part 2 ──────────────────────────────────────────────────────────────────────

/// 演示 `tool_error_feedback = true`（默认新行为）：工具失败后错误会反馈给 LLM，
/// Agent 可据此改用备用工具继续完成任务。
async fn demo_feedback_on() -> echo_agent::error::Result<()> {
    println!("  配置：tool_error_feedback = true（默认）\n");

    let mock = Arc::new(
        MockLlmClient::new()
            .then_tool_call(
                "call_broken_once",
                "broken_tool",
                r#"{"input":"trigger failure"}"#,
            )
            .then_tool_call("call_add_once", "add", r#"{"a":3,"b":4}"#)
            .then_tool_call(
                "call_final_answer",
                "final_answer",
                r#"{"answer":"broken_tool 失败后已自动改用 add，结果是 7。"}"#,
            ),
    );

    let mut agent = ReactAgentBuilder::new()
        .name("agent_with_feedback")
        .llm_client(mock)
        .system_prompt("你是一个智能助手。先尝试 broken_tool，失败后换用 add 计算 3+4。")
        .enable_tools()
        .tool_error_feedback(true)
        .max_iterations(6)
        .callback(Arc::new(SimpleLog { label: "FB-ON" }))
        .build()?;

    agent.add_tool(Box::new(BrokenTool));
    agent.add_tool(Box::new(AddTool));

    let answer = agent
        .execute("先调用 broken_tool，失败后换用 add 计算 3+4。")
        .await?;
    if !answer.contains('7') {
        return Err(ReactError::Other(format!(
            "demo12 验收失败：tool_error_feedback=true 未恢复到正确结果: {answer}"
        )));
    }
    println!("\n  ✅ 任务成功完成: {answer}");
    Ok(())
}

// ── Part 3 ──────────────────────────────────────────────────────────────────────

/// 演示 `FlakyTool` 偶发故障后自动恢复的场景：前两次调用失败，第三次成功返回结果。
async fn demo_flaky_tool() -> echo_agent::error::Result<()> {
    println!("  配置：FlakyTool（前 2 次失败）\n");

    let mock = Arc::new(
        MockLlmClient::new()
            .then_tool_call("call_weather_1", "weather_api", r#"{"city":"北京"}"#)
            .then_tool_call("call_weather_2", "weather_api", r#"{"city":"北京"}"#)
            .then_tool_call("call_weather_3", "weather_api", r#"{"city":"北京"}"#)
            .then_tool_call(
                "call_final_answer",
                "final_answer",
                r#"{"answer":"北京天气查询已在偶发故障后恢复成功：北京：晴，26°C"}"#,
            ),
    );

    let mut agent = ReactAgentBuilder::new()
        .name("agent_flaky")
        .llm_client(mock)
        .system_prompt("你是一个天气查询助手。调用 weather_api 查询北京天气，失败时请重试。")
        .enable_tools()
        .max_iterations(8)
        .callback(Arc::new(SimpleLog { label: "FLAKY" }))
        .build()?;

    agent.add_tool(Box::new(FlakyTool::new(2)));

    let answer = agent.execute("查询北京的实时天气。").await?;
    if !answer.contains("北京") {
        return Err(ReactError::Other(format!(
            "demo12 验收失败：FlakyTool 恢复后的结果不包含目标城市: {answer}"
        )));
    }
    println!("\n  ✅ 任务成功完成: {answer}");
    Ok(())
}

// ── Part 4 ──────────────────────────────────────────────────────────────────────

/// 对比 `llm_max_retries` 开启与关闭时的配置行为，并校验配置值是否正确生效。
fn demo_llm_retry_config() -> Result<()> {
    println!("  LLM 重试配置对比：\n");
    let no_retry = AgentConfig::new("model", "agent", "prompt").llm_max_retries(0);
    let retrying = AgentConfig::new("model", "agent", "prompt")
        .llm_max_retries(3)
        .llm_retry_delay_ms(500);
    if no_retry.get_llm_max_retries() != 0 || retrying.get_llm_max_retries() != 3 {
        return Err(ReactError::Other(
            "demo12 验收失败：LLM 重试配置未正确生效".to_string(),
        ));
    }
    println!("  ── llm_max_retries = 0（关闭重试）──");
    println!("     LLM 调用失败 → 立即返回 Err");
    println!();
    println!("  ── llm_max_retries = 3（开启重试）──");
    println!("     调用失败 → 等 500ms → 重试 1");
    println!("     再失败  → 等 1000ms → 重试 2");
    println!("     再失败  → 等 2000ms → 重试 3");
    Ok(())
}

// ── 辅助 ────────────────────────────────────────────────────────────────────────

/// 将字符串截断到最多 `max` 个字符，超出部分用省略号 `…` 表示。
fn trunc(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let out: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{out}…")
    } else {
        out
    }
}

/// 将 JSON 值压缩为紧凑的单行 `key=value` 形式，便于日志展示。
fn compact(v: &Value) -> String {
    match v {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                format!(
                    "{k}={}",
                    match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    }
                )
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    }
}

/// 打印分隔线和小节标题。
fn sep(title: &str) {
    println!("\n{}", "─".repeat(64));
    println!("{title}\n");
}

/// 打印 demo12 的开场横幅。
fn print_banner() {
    println!("{}", "═".repeat(64));
    println!("      Echo Agent × 韧性特性开关对比 (demo12)");
    println!("{}", "═".repeat(64));
    println!();
}
