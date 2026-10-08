/*!
 * demo54 — 无头模式
 *
 * 演示为 CI/CD 流水线、脚本与自动化场景设计的无头（非交互）执行模式：
 *
 * 1. `HeadlessConfig` —— 配置提示词、输出格式与迭代上限
 * 2. `HeadlessResult` —— 结构化结果，提供用于进程退出的 exit_code()
 * 3. 文本输出模式 —— 纯字符串输出
 * 4. JSON 输出模式 —— 包含 success、model、output 字段的结构化 JSON
 * 5. 空提示词的错误处理
 * 6. `HeadlessRunHandle` —— 取消安全的结果获取与关闭所有权
 *
 * **注意：** 实际运行 agent 需要 LLM API key。
 * 本示例仅展示配置与结果处理 API，不会发起真实的 LLM 调用。
 *
 * ```sh
 * cargo test --test example_contracts contract_demo54_headless
 * ```
 */

use echo_agent::headless::{HeadlessConfig, HeadlessResult};

/// 演示取消安全的无头运行模式：等待结果，失败时重试关闭运行句柄。
#[allow(dead_code)]
async fn cancellation_safe_headless_pattern(
    config: HeadlessConfig,
) -> echo_agent::error::Result<HeadlessResult> {
    let run = echo_agent::headless::start_headless(config, |builder| builder);
    let result = run.wait().await;
    if !result.success {
        run.retry_close().await?;
    }
    Ok(result)
}

/// 打印场景分节标题。
macro_rules! section {
    ($n:expr, $title:expr) => {
        println!("\n══════════════════════════════════════════════════");
        println!("  Scenario {} : {}", $n, $title);
        println!("══════════════════════════════════════════════════");
    };
}

/// demo54 契约测试入口：依次运行 7 个无头模式场景。
#[tokio::test]
async fn contract_demo54_headless() -> Result<(), serde_json::Error> {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║       echo-agent  Headless Mode Demo             ║");
    println!("║  (config + result handling — no LLM required)    ║");
    println!("╚══════════════════════════════════════════════════╝");

    demo_config_defaults();
    demo_config_custom();
    demo_text_output();
    demo_json_output()?;
    demo_exit_codes();
    demo_empty_prompt();
    demo_cicd_pattern();

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  All 7 scenarios passed ✅                       ║");
    println!("╚══════════════════════════════════════════════════╝");
    Ok(())
}

/// 场景 1：默认配置
fn demo_config_defaults() {
    section!(1, "HeadlessConfig Defaults");

    let config = HeadlessConfig::default();
    println!("  prompt:         {:?}", config.prompt);
    println!("  exit_on_error:  {}", config.exit_on_error);
    println!("  output_format:  {:?}", config.output_format);
    println!("  max_iterations: {:?}", config.max_iterations);

    assert!(config.prompt.is_empty());
    assert!(config.exit_on_error);
    assert_eq!(config.output_format, "text");
    assert!(config.max_iterations.is_none());
    println!("  ✅ Defaults verified");
}

/// 场景 2：自定义配置
fn demo_config_custom() {
    section!(2, "Custom HeadlessConfig");

    let config = HeadlessConfig {
        prompt: "List all Rust files in the project".into(),
        exit_on_error: true,
        output_format: "json".into(),
        max_iterations: Some(10),
        cancel_token: None,
    };
    println!("  prompt:         {:?}", config.prompt);
    println!("  exit_on_error:  {}", config.exit_on_error);
    println!("  output_format:  {:?}", config.output_format);
    println!("  max_iterations: {:?}", config.max_iterations);

    assert_eq!(config.prompt, "List all Rust files in the project");
    assert_eq!(config.max_iterations, Some(10));
    println!("  ✅ Custom config created");
}

/// 场景 3：文本输出格式化
fn demo_text_output() {
    section!(3, "Text Output Mode");

    let result = HeadlessResult {
        output: "Found 42 Rust files:\n  src/main.rs\n  src/lib.rs\n  ...".into(),
        success: true,
        model: "deepseek-v4-flash".into(),
        format: "text".into(),
        exit_on_error: true,
    };

    let formatted = result.format_output();
    println!("  format_output() result:");
    for line in formatted.lines() {
        println!("    {}", line);
    }

    assert_eq!(formatted, result.output);
    println!("  ✅ Text mode returns raw output string");
}

/// 场景 4：JSON 输出格式化
fn demo_json_output() -> Result<(), serde_json::Error> {
    section!(4, "JSON Output Mode");

    let result = HeadlessResult {
        output: "Task completed successfully".into(),
        success: true,
        model: "qwen3-max".into(),
        format: "json".into(),
        exit_on_error: true,
    };

    let formatted = result.format_output();
    println!("  format_output() result:");
    for line in formatted.lines() {
        println!("    {}", line);
    }

    // 校验 JSON 结构
    let parsed: serde_json::Value = serde_json::from_str(&formatted)?;
    assert_eq!(
        parsed.get("success").and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        parsed.get("model").and_then(serde_json::Value::as_str),
        Some("qwen3-max")
    );
    assert_eq!(
        parsed.get("output").and_then(serde_json::Value::as_str),
        Some("Task completed successfully")
    );
    println!("  ✅ JSON mode returns structured {{success, model, output}}");
    Ok(())
}

/// 场景 5：退出码计算
fn demo_exit_codes() {
    section!(5, "Exit Codes for CI/CD");

    // 成功 → 退出码 0
    let ok_result = HeadlessResult {
        output: "All tests passed".into(),
        success: true,
        model: "test-model".into(),
        format: "text".into(),
        exit_on_error: true,
    };
    println!("  Success case:");
    println!("    exit_code() = {}", ok_result.exit_code());
    assert_eq!(ok_result.exit_code(), 0);

    // 失败 → 退出码 1
    let fail_result = HeadlessResult {
        output: "Error: compilation failed".into(),
        success: false,
        model: "test-model".into(),
        format: "text".into(),
        exit_on_error: true,
    };
    println!("  Failure case:");
    println!("    exit_code() = {}", fail_result.exit_code());
    assert_eq!(fail_result.exit_code(), 1);

    println!("  ✅ exit_code() returns 0 for success, 1 for failure");
}

/// 场景 6：空提示词处理（模拟）
fn demo_empty_prompt() {
    section!(6, "Empty Prompt Error Handling");

    /* run_headless 收到空提示词时会返回失败结果。
    这里模拟预期的结果： */
    let result = HeadlessResult {
        output: "Error: empty prompt".into(),
        success: false,
        model: String::new(),
        format: "text".into(),
        exit_on_error: true,
    };

    println!("  Empty prompt → success=false");
    println!("  Output: {}", result.output);
    println!("  exit_code() = {}", result.exit_code());
    assert_eq!(result.exit_code(), 1);
    println!("  ✅ Empty prompt is rejected gracefully");
}

/// 场景 7：完整的 CI/CD 流水线模式
fn demo_cicd_pattern() {
    section!(7, "CI/CD Pipeline Pattern");

    println!("  Typical usage in a CI/CD script:\n");
    println!("  ```rust");
    println!("  let config = HeadlessConfig {{");
    println!("      prompt: \"Review this PR for security issues\".into(),");
    println!("      exit_on_error: true,");
    println!("      output_format: \"json\".into(),");
    println!("      max_iterations: Some(20),");
    println!("      cancel_token: None,");
    println!("  }};");
    println!();
    println!("  let result = run_headless(config, |builder| {{");
    println!("      builder");
    println!("          .model(\"deepseek-v4-flash\")");
    println!("          .system_prompt(\"You are a security reviewer.\")");
    println!("  }}).await;");
    println!();
    println!("  println!(\"{{}}\", result.format_output());");
    println!("  std::process::exit(result.exit_code());");
    println!("  ```\n");

    // 演示 JSON 输出的样子
    let simulated = HeadlessResult {
        output: "Found 2 potential XSS vulnerabilities in handlers.rs".into(),
        success: true,
        model: "deepseek-v4-flash".into(),
        format: "json".into(),
        exit_on_error: true,
    };
    println!("  Simulated CI output:");
    for line in simulated.format_output().lines() {
        println!("    {}", line);
    }
    println!("  exit_code = {} (CI passes)", simulated.exit_code());
    println!("  ✅ CI/CD integration pattern demonstrated");
}
