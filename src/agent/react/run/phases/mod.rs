/*!
`AgentRunSnapshot::run_core_loop` 所使用的各阶段（phase）函数。

`run_core_loop` 是**单一、统一的核心循环** —— 流式与非流式两个
入口最终都会启动同一个循环体。为了让这个循环体保持可读、可审查，
每次迭代的工作被拆分为若干职责单一的阶段函数：

```text
prepare_turn  →  ┌─ run_compact ─→ run_think ─→ ┬─ run_tools（通过校验 → finalize_completed_run）
                 │                              ├─ verify_final_text（通过校验 → emit_final_text）
                 │                              └─ NoResponse → finalize_no_response
                 └─（循环继续……）              max_iterations 耗尽 → finalize_max_iterations
```

各阶段函数均为 `pub(crate) async fn`，且都返回 `Result<…>`。
来自 `super::super::stream_macros` 的 `yield_event!` /
`yield_final_event!` / `try_send!` 宏，在接收端已关闭时会让
**外层函数**通过 `return Ok(())` 直接短路返回，因此各阶段函数
必须返回 `Result<…>`，以便这种“放弃流”的行为能够一路冒泡到
循环驱动器（loop driver）—— 驱动器随后以 `Ok(())` 退出。
*/

use crate::llm::types::Message;
use std::collections::HashMap;

/// 上下文压缩阶段。
pub(crate) mod compact;
/// 最终答案收尾阶段。
pub(crate) mod finalize;
/// 运行前准备阶段。
pub(crate) mod prepare;
/// LLM 思考（推理）阶段。
pub(crate) mod think;
/// 工具调用执行阶段。
pub(crate) mod tools;
/// 最终答案校验阶段。
pub(crate) mod verify;

// ── 循环级可变状态 ─────────────────────────────────────────

/**
在各阶段函数调用之间共享的、按 turn 划分的可变状态。

`agent` 与 `callbacks` 直接从 `snap.config` 读取，而不在此处
再镜像一份 —— 它们克隆代价很低，且从不会被修改。
*/
pub(crate) struct LoopState {
    /**
    本轮是否已经允许过 `Stop` 生命周期钩子注入 continue_reason。
    该钩子只允许触发一次，以避免陷入循环。
    */
    pub stop_hook_continued: bool,
    /// 本轮中校验器（verifier）拒绝答案的次数。
    pub verifier_retry_count: usize,
    /**
    本次调用范围内的预算计数器。在等待 HITL（人工介入）批准期间，
    它们会保留在循环状态中，因而能在暂停/恢复之间保持不变。
    */
    pub budget: RunBudgetState,
}

/// 单次运行（invocation）范围内的预算相关计数状态。
#[derive(Default)]
pub(crate) struct RunBudgetState {
    /// 已上报的模型 token 数量。
    pub reported_model_tokens: usize,
    /// 用量信息是否已完整上报。
    pub usage_complete: bool,
    /// 是否已发出“收尾”（wind-down）事件。
    pub wind_down_emitted: bool,
    /// 是否已发出“仅最终答案”事件。
    pub final_only_emitted: bool,
    /// 是否处于“仅最终答案”模式。
    pub final_only: bool,
}

impl RunBudgetState {
    /// 记录一次 token 用量，并更新用量是否完整上报的标记。
    pub(crate) fn record_usage(&mut self, tokens: usize, reported: bool) {
        self.reported_model_tokens = self.reported_model_tokens.saturating_add(tokens);
        self.usage_complete = self.usage_complete && reported;
    }
}

impl LoopState {
    /// 创建一个新的、处于初始状态的 [`LoopState`]。
    pub(crate) fn new() -> Self {
        Self {
            stop_hook_continued: false,
            verifier_retry_count: 0,
            budget: RunBudgetState {
                usage_complete: true,
                ..RunBudgetState::default()
            },
        }
    }
}

// ── 结果枚举 ────────────────────────────────────────────────────

/// `prepare_turn` 阶段做出的决策。
pub(crate) enum PrepareOutcome {
    /// 继续进入迭代循环。
    Continue,
    /**
    `UserPromptSubmit` 钩子返回了 `block: true`。此时已经发出过
    一次 `FinalAnswer` 事件，并且已经触发了 `SessionEnd("blocked")`。
    */
    BlockedAndDone,
    /// 在准备阶段过程中通道被关闭；循环驱动器返回 `Ok(())`。
    Abandoned,
}

/// `run_compact` 在本次迭代中产生的结果。
#[derive(Debug)]
pub(crate) enum CompactOutcome {
    /// 已准备好、可供 think 阶段使用的 LLM 消息列表。
    Continue(Vec<Message>),
    /// 通道已关闭；循环驱动器返回 `Ok(())`。
    Abandoned,
    /// 上下文准备失败，且已发出一个错误终态事件。
    Failed,
}

/// `run_think` 在本次迭代中产生的结果。
pub(crate) enum ThinkOutcome {
    /// 流已被消费完毕；已提取出工具调用 / 正文内容 / token 用量。
    Continue(ThinkOutput),
    /// think 阶段过程中通道被关闭。
    Abandoned,
    /// 本阶段已经完成了持久化、trace 记录，以及一个终态事件的落地。
    TerminalSettled {
        outcome: crate::agent::AgentSteerTurnOutcome,
    },
}

/// think 阶段的输出，将被送入 tools 分支或 verify 分支继续处理。
pub(crate) struct ThinkOutput {
    /// 从流式分片中累积得到的助手纯文本内容。
    pub content_buffer: String,
    /// 用于助手消息回放（replay）而累积的 DeepSeek/Qwen 推理内容。
    pub reasoning_buffer: String,
    /// 供提供商有效回放所需的、经过签名/脱敏处理的推理块。
    pub reasoning_blocks: Vec<crate::llm::types::ReasoningBlock>,
    /// 按索引累积的工具调用：index → (tool_call_id, function_name, args)。
    pub tool_call_map: HashMap<u32, (String, String, String)>,
    /// LLM 上报的 prompt token 数。
    pub pt: usize,
    /// LLM 上报的 completion token 数。
    pub ct: usize,
    /// 该响应中提供商是否附带了用量（usage）元数据。
    pub usage_reported: bool,
}

/// 单次迭代主体所做出的决策。
pub(crate) enum IterOutcome {
    /// 校验失败或钩子要求继续：进入下一次迭代。
    Continue,
    /**
    工具分支：一次 `final_answer` 工具调用通过了校验并被接受。
    驱动器将调用 `phases::finalize::finalize_completed_run`。
    */
    Finish { output: String },
    /**
    纯文本分支：LLM 产出的正文内容通过了校验。
    驱动器将调用 `phases::finalize::emit_final_text`。
    */
    FinalText {
        answer: String,
        reasoning_content: String,
        reasoning_blocks: Vec<crate::llm::types::ReasoningBlock>,
    },
    /// LLM 既没有产出工具调用，也没有产出正文内容。属于终态失败。
    NoResponse,
    /**
    迭代过程中通道被关闭（某个 yield/try_send 宏触发了
    `return Ok(())`）。驱动器立即返回 `Ok(())`。
    */
    Abandoned,
    /// 某个阶段已经完成了持久化、trace 记录，以及唯一一个终态的落地。
    TerminalSettled {
        outcome: crate::agent::AgentSteerTurnOutcome,
    },
}

/// 在消息上附加推理内容（reasoning content）与推理块（如果非空）。
pub(crate) fn with_reasoning_content(
    mut message: Message,
    reasoning_content: String,
    reasoning_blocks: Vec<crate::llm::types::ReasoningBlock>,
) -> Message {
    if !reasoning_content.is_empty() {
        message.reasoning_content = Some(reasoning_content);
    }
    if !reasoning_blocks.is_empty() {
        message.reasoning_blocks = Some(reasoning_blocks);
    }
    message
}

/// 预算计数器在暂停/恢复边界上的行为测试。
#[cfg(test)]
mod budget_tests {
    use super::LoopState;

    /// 验证预算计数器跨越异步暂停/恢复边界后保持不变。
    #[tokio::test]
    async fn budget_counters_survive_async_pause_resume_boundary() {
        let mut state = LoopState::new();
        state.budget.record_usage(12, true);

        /*
        HITL 批准会在同一个 run future 内部暂停。跨越一次 await
        边界不应重置调用范围内的计数器。
        */
        tokio::task::yield_now().await;
        state.budget.record_usage(8, false);

        assert_eq!(state.budget.reported_model_tokens, 20);
        assert!(!state.budget.usage_complete);
    }
}
