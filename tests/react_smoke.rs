/*!
框架级集成测试。

这些测试位于 `tests/` crate 中（不使用 `#[cfg(test)]`），覆盖真实的生产代码路径。
它们聚焦于纯逻辑（不依赖真实 LLM）且价值较高的领域：
token 校准收敛（P5），以及 spawn_task / prompt_exec 共享的 shell 安全分类器（S1/S6）。

需要用 mock LLM 驱动完整 ReAct 循环的测试，需要 `testing` feature
以及快照级别的 `LlmClient` 字段（作为一项更大的重构单独跟踪）。
*/

/**
校准分词器在接收真实用量后，其校准因子必须向真实比例收敛。
用于守护 P5 的接线（因子必须偏离 1.0）。
*/
#[tokio::test]
async fn calibrated_tokenizer_converges_on_feedback() {
    use echo_agent::tokenizer::{CalibratedTokenizer, HeuristicTokenizer, Tokenizer};
    use std::sync::Arc;

    let base = Arc::new(HeuristicTokenizer);
    let calibrated = CalibratedTokenizer::with_alpha(base.clone(), 1.0); // 立即更新

    let text = "hello world this is a calibration test";
    let estimated = base.count_tokens(text);

    // 模拟真实模型报告的用量为启发式估算值的 2 倍。
    calibrated.calibrate(estimated, (estimated * 2) as u32);

    assert!(
        (calibrated.calibration_factor() - 2.0).abs() < 0.01,
        "factor should converge to 2.0, got {}",
        calibrated.calibration_factor()
    );
    assert_eq!(calibrated.count_tokens(text), estimated * 2);
}

/**
shell 安全分类器（由 spawn_task 和 prompt_exec 共享）必须拒绝
shell 元字符和危险命令。用于守护 S1/S6。
*/
#[test]
fn shell_safety_rejects_metacharacters() {
    #[cfg(feature = "shell")]
    {
        use echo_tools::shell::{CommandSafety, validate_command_safety};

        // 管道 / 分号是注入途径 → 拒绝。
        let dangerous = validate_command_safety("ls; rm -rf /");
        assert!(
            matches!(dangerous, CommandSafety::Dangerous(_)),
            "metacharacter command must be rejected"
        );

        // 白名单中的只读命令是安全的。
        let safe = validate_command_safety("ls -la");
        assert!(
            matches!(safe, CommandSafety::Safe),
            "plain ls should be safe, got {:?}",
            safe
        );
    }
}
