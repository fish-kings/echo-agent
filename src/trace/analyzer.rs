/*!
 * TraceAnalyzer —— 面向已存储运行记录（run）的可观测性分析器。
 *
 * 提供会话级汇总、工具使用统计、token 用量拆分、慢工具检测以及错误
 * 模式分析。所有方法都基于共享的 [`RunStore`] 后端运行，因此与具体的
 * 存储实现（内存、JSONL 等）无关。
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::trace::{RunEvent, RunStatus, RunStore, TokenUsage};

// ── 分析结果结构体 ────────────────────────────────────────────

/// 单个会话的汇总信息（对该会话下所有运行记录的聚合）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    /// 会话标识。
    pub session_id: String,
    /// 该会话中的运行次数。
    pub run_count: usize,
    /// 已完成的运行次数。
    pub completed_count: usize,
    /// 失败的运行次数。
    pub failed_count: usize,
    /// 被取消的运行次数。
    pub cancelled_count: usize,
    /// 所有运行的 prompt token 总数。
    pub total_prompt_tokens: u32,
    /// 所有运行的 completion token 总数。
    pub total_completion_tokens: u32,
    /// 所有运行的 token 总数。
    pub total_tokens: u32,
    /// 所有运行累计的墙钟（wall-clock）耗时（毫秒）。
    pub total_duration_ms: u64,
    /// 累计的 LLM 调用耗时（毫秒）。
    pub total_llm_duration_ms: u64,
    /// 累计的工具执行耗时（毫秒）。
    pub total_tool_duration_ms: u64,
    /// 该会话中最早的 `started_at`。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub first_started_at: Option<DateTime<Utc>>,
    /// 该会话中最晚的 `finished_at`。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub last_finished_at: Option<DateTime<Utc>>,
    /// 该会话所有运行中使用过的工具名称。
    pub tools_used: Vec<String>,
    /// 所有运行的 LLM 调用次数。
    pub llm_call_count: usize,
}

/// 单个工具的使用统计信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageStats {
    /// 工具名称。
    pub name: String,
    /// 该工具被调用的次数。
    pub call_count: usize,
    /// 调用成功的次数。
    pub success_count: usize,
    /// 调用失败的次数。
    pub failure_count: usize,
    /// 单次调用的平均耗时（毫秒）。
    pub avg_duration_ms: u64,
    /// 最短耗时（毫秒）。
    pub min_duration_ms: u64,
    /// 最长耗时（毫秒）。
    pub max_duration_ms: u64,
    /// 总耗时（毫秒）。
    pub total_duration_ms: u64,
    /// 占全部工具总耗时的百分比。
    pub pct_of_total_time: f64,
}

/// 跨运行记录的 token 用量拆分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBreakdown {
    /// prompt token 总数。
    pub prompt_tokens: u32,
    /// completion token 总数。
    pub completion_tokens: u32,
    /// token 总数（prompt + completion）。
    pub total_tokens: u32,
    /// 按运行拆分的明细（run_id -> TokenUsage）。
    pub per_run: HashMap<String, TokenUsage>,
    /// 按每次 LLM 调用拆分的明细（run_id -> (prompt, completion) 列表）。
    pub per_llm_call: HashMap<String, Vec<(u32, u32)>>,
}

/// 在多次运行中识别出的一个重复出现的错误模式。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    /// 归一化后的错误信息（按相似度分组）。
    pub pattern: String,
    /// 出现次数。
    pub occurrence_count: usize,
    /// 出现该模式的运行 ID 列表。
    pub run_ids: Vec<String>,
    /// 与该错误关联最频繁的工具名称。
    pub associated_tools: Vec<String>,
    /// 首次出现的时间戳。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub first_seen: Option<DateTime<Utc>>,
    /// 最近一次出现的时间戳。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub last_seen: Option<DateTime<Utc>>,
}

/// 运行时工具失败分类的向后兼容别名。
pub type ToolFailureClass = crate::tools::ToolFailureCategory;

/// 在多次运行中重复出现的工具失败分组，不暴露原始工具参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFailurePattern {
    /// 发生失败的工具。
    pub tool_name: String,
    /// 确定性的失败分类。
    pub error_class: ToolFailureClass,
    /// 用于分组的归一化错误预览。
    pub pattern: String,
    /// 输入的结构形状（不含具体参数取值）。
    pub input_shape: String,
    /// 失败发生的次数。
    pub occurrence_count: usize,
    /// 出现该失败的不同运行数量。
    pub distinct_run_count: usize,
    /// 出现该失败的运行 ID 列表。
    pub run_ids: Vec<String>,
    /// 同一运行中，首次失败之后重复的相同尝试次数。
    pub ineffective_retry_count: usize,
    /// 出现该模式的最早运行起始时间。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub first_seen: Option<DateTime<Utc>>,
    /// 出现该模式的最近运行结束时间。
    #[serde(with = "crate::utils::time::option_local_rfc3339")]
    pub last_seen: Option<DateTime<Utc>>,
}

/// 在一批有限数量的已存储运行记录上得到的工具可靠性汇总。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolReliabilityReport {
    /// 参与统计的运行记录数量。
    pub run_count: usize,
    /// 工具调用总次数。
    pub total_calls: usize,
    /// 调用成功的次数。
    pub success_count: usize,
    /// 调用失败的次数。
    pub failure_count: usize,
    /// 无效重试（同参数重复失败）的次数。
    pub ineffective_retry_count: usize,
    /// 归纳出的失败模式列表。
    pub failure_patterns: Vec<ToolFailurePattern>,
}

// ── TraceAnalyzer ──────────────────────────────────────────────────────

/**
 * 面向执行轨迹（trace）的可观测性分析器。
 *
 * 包装一个 [`RunStore`]，提供跨运行、跨会话、跨事件聚合数据的分析方法。
 * 所有方法都是异步的，以适应可能较慢的存储后端（例如 JSONL 存储的磁盘
 * I/O）。
 */
pub struct TraceAnalyzer {
    /// 底层的运行记录存储后端。
    run_store: Arc<dyn RunStore>,
}

impl TraceAnalyzer {
    /// 创建一个基于给定 [`RunStore`] 的新分析器。
    pub fn new(run_store: Arc<dyn RunStore>) -> Self {
        Self { run_store }
    }

    /// 通过聚合某个会话下的所有运行记录，生成该会话的汇总信息。
    pub async fn summarize_session(
        &self,
        session_id: &str,
    ) -> crate::error::Result<SessionSummary> {
        let summaries = self.run_store.list_by_session(session_id).await?;

        let mut completed_count = 0;
        let mut failed_count = 0;
        let mut cancelled_count = 0;
        let mut total_prompt_tokens = 0u32;
        let mut total_completion_tokens = 0u32;
        let mut total_tokens = 0u32;
        let mut total_duration_ms = 0u64;
        let mut total_llm_duration_ms = 0u64;
        let mut total_tool_duration_ms = 0u64;
        let mut first_started_at: Option<DateTime<Utc>> = None;
        let mut last_finished_at: Option<DateTime<Utc>> = None;
        let mut tools_used_set: HashMap<String, ()> = HashMap::new();
        let mut llm_call_count = 0usize;

        for summary in &summaries {
            match summary.status {
                RunStatus::Completed => completed_count += 1,
                RunStatus::Failed => failed_count += 1,
                RunStatus::Cancelled => cancelled_count += 1,
                _ => {}
            }
            total_prompt_tokens =
                total_prompt_tokens.saturating_add(summary.token_usage.prompt_tokens);
            total_completion_tokens =
                total_completion_tokens.saturating_add(summary.token_usage.completion_tokens);
            total_tokens = total_tokens.saturating_add(summary.token_usage.total_tokens);
            total_duration_ms += summary.total_duration_ms;

            if first_started_at
                .as_ref()
                .is_none_or(|first| summary.started_at < *first)
            {
                first_started_at = Some(summary.started_at);
            }
            if let Some(fa) = summary.finished_at
                && last_finished_at.as_ref().is_none_or(|last| fa > *last)
            {
                last_finished_at = Some(fa);
            }

            // 加载完整的运行记录，以提取工具名称和 LLM 调用次数
            if let Some(run) = self.run_store.load(&summary.run_id).await? {
                total_llm_duration_ms += run.timings.llm_duration_ms;
                total_tool_duration_ms += run.timings.tool_duration_ms;
                let skipped = super::skipped_tool_call_ids(&run.events);
                for event in &run.events {
                    match event {
                        RunEvent::ToolCall { call_id, name, .. }
                            if !skipped.contains(call_id.as_str()) =>
                        {
                            tools_used_set.insert(name.clone(), ());
                        }
                        RunEvent::LlmCall { .. } => {
                            llm_call_count += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(SessionSummary {
            session_id: session_id.to_string(),
            run_count: summaries.len(),
            completed_count,
            failed_count,
            cancelled_count,
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            total_duration_ms,
            total_llm_duration_ms,
            total_tool_duration_ms,
            first_started_at,
            last_finished_at,
            tools_used: tools_used_set.keys().cloned().collect(),
            llm_call_count,
        })
    }

    /// 计算所有运行记录（最多 `limit` 条）中各工具的使用统计信息。
    pub async fn tool_usage_stats(
        &self,
        limit: usize,
    ) -> crate::error::Result<Vec<ToolUsageStats>> {
        let summaries = self.run_store.list_all(limit).await?;

        // 按工具累加数据
        let mut tool_data: HashMap<String, ToolAccumulator> = HashMap::new();

        for summary in &summaries {
            if let Some(run) = self.run_store.load(&summary.run_id).await? {
                let mut failed_call_ids = HashSet::new();
                let skipped = super::skipped_tool_call_ids(&run.events);
                for event in &run.events {
                    match event {
                        RunEvent::ToolCall {
                            call_id,
                            name,
                            duration_ms,
                            ..
                        } if !skipped.contains(call_id.as_str()) => {
                            let acc = tool_data.entry(name.clone()).or_default();
                            acc.call_count += 1;
                            acc.total_duration_ms += *duration_ms;
                            if *duration_ms < acc.min_duration_ms {
                                acc.min_duration_ms = *duration_ms;
                            }
                            if *duration_ms > acc.max_duration_ms {
                                acc.max_duration_ms = *duration_ms;
                            }
                        }
                        RunEvent::ToolResult {
                            call_id,
                            name,
                            success,
                            ..
                        } if !skipped.contains(call_id.as_str()) => {
                            if let Some(acc) = tool_data.get_mut(name) {
                                if *success {
                                    acc.success_count += 1;
                                } else if failed_call_ids.insert(call_id.clone()) {
                                    acc.failure_count += 1;
                                }
                            }
                        }
                        RunEvent::ToolError { call_id, name, .. }
                            if !skipped.contains(call_id.as_str()) =>
                        {
                            if failed_call_ids.insert(call_id.clone())
                                && let Some(acc) = tool_data.get_mut(name)
                            {
                                acc.failure_count += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // 计算工具总耗时，用于后续百分比计算
        let grand_total_ms: u64 = tool_data.values().map(|a| a.total_duration_ms).sum();

        let mut stats: Vec<ToolUsageStats> = tool_data
            .into_iter()
            .map(|(name, acc)| {
                let avg = if acc.call_count > 0 {
                    acc.total_duration_ms / acc.call_count as u64
                } else {
                    0
                };
                let pct = if grand_total_ms > 0 {
                    (acc.total_duration_ms as f64 / grand_total_ms as f64) * 100.0
                } else {
                    0.0
                };
                ToolUsageStats {
                    name,
                    call_count: acc.call_count,
                    success_count: acc.success_count,
                    failure_count: acc.failure_count,
                    avg_duration_ms: avg,
                    min_duration_ms: acc.min_duration_ms,
                    max_duration_ms: acc.max_duration_ms,
                    total_duration_ms: acc.total_duration_ms,
                    pct_of_total_time: pct,
                }
            })
            .collect();

        // 按 total_duration 降序排序
        stats.sort_by_key(|b| std::cmp::Reverse(b.total_duration_ms));
        Ok(stats)
    }

    /// 计算所有运行记录（最多 `limit` 条）的 token 用量拆分。
    pub async fn token_usage_breakdown(
        &self,
        limit: usize,
    ) -> crate::error::Result<TokenBreakdown> {
        let summaries = self.run_store.list_all(limit).await?;

        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut total_tokens = 0u32;
        let mut per_run: HashMap<String, TokenUsage> = HashMap::new();
        let mut per_llm_call: HashMap<String, Vec<(u32, u32)>> = HashMap::new();

        for summary in &summaries {
            prompt_tokens = prompt_tokens.saturating_add(summary.token_usage.prompt_tokens);
            completion_tokens =
                completion_tokens.saturating_add(summary.token_usage.completion_tokens);
            total_tokens = total_tokens.saturating_add(summary.token_usage.total_tokens);
            per_run.insert(summary.run_id.clone(), summary.token_usage);

            if let Some(run) = self.run_store.load(&summary.run_id).await? {
                let mut llm_calls: Vec<(u32, u32)> = Vec::new();
                for event in &run.events {
                    if let RunEvent::LlmCall {
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        ..
                    } = event
                    {
                        llm_calls.push((*pt, *ct));
                    }
                }
                if !llm_calls.is_empty() {
                    per_llm_call.insert(summary.run_id.clone(), llm_calls);
                }
            }
        }

        Ok(TokenBreakdown {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            per_run,
            per_llm_call,
        })
    }

    /// 查找平均耗时超过 `threshold_ms` 的工具。
    pub async fn find_slow_tools(
        &self,
        threshold_ms: u64,
        limit: usize,
    ) -> crate::error::Result<Vec<ToolUsageStats>> {
        let all_stats = self.tool_usage_stats(limit).await?;
        Ok(all_stats
            .into_iter()
            .filter(|s| s.avg_duration_ms >= threshold_ms)
            .collect())
    }

    /**
     * 分析各次运行中的错误模式。按归一化后的消息（转小写、去空白）对
     * 错误进行分组，并报告出现次数、关联工具及时间信息。
     */
    pub async fn error_pattern_analysis(
        &self,
        limit: usize,
    ) -> crate::error::Result<Vec<ErrorPattern>> {
        let summaries = self.run_store.list_all(limit).await?;

        // pattern_key -> 累加器
        let mut patterns: HashMap<String, ErrorAccumulator> = HashMap::new();

        for summary in &summaries {
            if let Some(run) = self.run_store.load(&summary.run_id).await? {
                let skipped = super::skipped_tool_call_ids(&run.events);
                // 运行级别的错误
                if run.status == RunStatus::Failed
                    && let Some(ref error_msg) = run.error
                {
                    let key = normalize_error(error_msg);
                    let acc = patterns.entry(key.clone()).or_default();
                    acc.occurrence_count += 1;
                    acc.run_ids.push(run.run_id.clone());
                    acc.update_time(run.started_at, run.finished_at);
                }

                // 事件级别的错误
                for event in &run.events {
                    match event {
                        RunEvent::ToolError {
                            call_id,
                            name,
                            message,
                            ..
                        } if !skipped.contains(call_id.as_str()) => {
                            let key = normalize_error(message);
                            let acc = patterns.entry(key.clone()).or_default();
                            acc.occurrence_count += 1;
                            acc.run_ids.push(run.run_id.clone());
                            acc.associated_tools.insert(name.clone(), ());
                            acc.update_time(run.started_at, run.finished_at);
                        }
                        RunEvent::Error { message } => {
                            let key = normalize_error(message);
                            let acc = patterns.entry(key.clone()).or_default();
                            acc.occurrence_count += 1;
                            acc.run_ids.push(run.run_id.clone());
                            acc.update_time(run.started_at, run.finished_at);
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut result: Vec<ErrorPattern> = patterns
            .into_iter()
            .map(|(pattern, acc)| ErrorPattern {
                pattern,
                occurrence_count: acc.occurrence_count,
                run_ids: acc.run_ids,
                associated_tools: acc.associated_tools.keys().cloned().collect(),
                first_seen: acc.first_seen,
                last_seen: acc.last_seen,
            })
            .collect();

        // 按出现次数降序排序
        result.sort_by_key(|b| std::cmp::Reverse(b.occurrence_count));
        Ok(result)
    }

    /**
     * 分析发生在 `after`（含）之后的运行记录的工具可靠性。
     *
     * 工具参数仅用于检测同一运行内重复出现的相同尝试；返回的模式只
     * 暴露结构化的输入形状，绝不暴露原始参数值。
     */
    pub async fn tool_reliability_report(
        &self,
        limit: usize,
        after: Option<DateTime<Utc>>,
    ) -> crate::error::Result<ToolReliabilityReport> {
        let summaries = self.run_store.list_all(limit).await?;
        let mut report = ToolReliabilityReport::default();
        let mut patterns: HashMap<String, ToolFailureAccumulator> = HashMap::new();

        for summary in summaries
            .iter()
            .filter(|summary| after.is_none_or(|after| summary.started_at >= after))
        {
            let Some(run) = self.run_store.load(&summary.run_id).await? else {
                continue;
            };
            report.run_count = report.run_count.saturating_add(1);
            let mut calls: HashMap<String, ToolCallContext> = HashMap::new();
            let mut attempts: HashMap<String, usize> = HashMap::new();
            let mut failed_call_ids = HashSet::new();
            let skipped = super::skipped_tool_call_ids(&run.events);

            for event in &run.events {
                match event {
                    RunEvent::ToolCall {
                        call_id,
                        name,
                        args,
                        ..
                    } if !skipped.contains(call_id.as_str()) => {
                        report.total_calls = report.total_calls.saturating_add(1);
                        calls.insert(
                            call_id.clone(),
                            ToolCallContext {
                                tool_name: name.clone(),
                                input_shape: json_shape(args.as_ref()),
                                args_key: json_identity(args.as_ref()),
                            },
                        );
                    }
                    RunEvent::ToolResult {
                        call_id,
                        success: true,
                        ..
                    } if !skipped.contains(call_id.as_str()) => {
                        report.success_count = report.success_count.saturating_add(1);
                    }
                    RunEvent::ToolError {
                        call_id,
                        name,
                        message,
                        failure,
                    } if !skipped.contains(call_id.as_str())
                        && failed_call_ids.insert(call_id.clone()) =>
                    {
                        report.failure_count = report.failure_count.saturating_add(1);
                        record_tool_failure(
                            &mut patterns,
                            &mut attempts,
                            &run,
                            calls.get(call_id),
                            name,
                            message,
                            failure.as_ref(),
                        );
                    }
                    _ => {}
                }
            }
        }

        let mut failure_patterns = patterns
            .into_values()
            .map(|accumulator| accumulator.into_pattern())
            .collect::<Vec<_>>();
        failure_patterns.sort_by_key(|pattern| std::cmp::Reverse(pattern.occurrence_count));
        report.ineffective_retry_count = failure_patterns
            .iter()
            .map(|pattern| pattern.ineffective_retry_count)
            .sum();
        report.failure_patterns = failure_patterns;
        Ok(report)
    }

    /// 列出所有会话（从已存储的运行记录中提取去重后的会话 ID）。
    pub async fn list_sessions(&self, limit: usize) -> crate::error::Result<Vec<String>> {
        let summaries = self.run_store.list_all(limit).await?;
        let mut session_ids: Vec<String> = summaries.iter().map(|s| s.session_id.clone()).collect();
        session_ids.sort();
        session_ids.dedup();
        Ok(session_ids)
    }
}

// ── 内部累加器 ──────────────────────────────────────────────

/// 用于统计单个工具耗时数据的累加器。
struct ToolAccumulator {
    call_count: usize,
    success_count: usize,
    failure_count: usize,
    total_duration_ms: u64,
    min_duration_ms: u64,
    max_duration_ms: u64,
}

impl Default for ToolAccumulator {
    fn default() -> Self {
        Self {
            call_count: 0,
            success_count: 0,
            failure_count: 0,
            total_duration_ms: 0,
            min_duration_ms: u64::MAX,
            max_duration_ms: 0,
        }
    }
}

/// 用于错误模式分组的累加器。
#[derive(Default)]
struct ErrorAccumulator {
    occurrence_count: usize,
    run_ids: Vec<String>,
    associated_tools: HashMap<String, ()>,
    first_seen: Option<DateTime<Utc>>,
    last_seen: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct ToolCallContext {
    tool_name: String,
    input_shape: String,
    args_key: String,
}

#[derive(Default)]
struct ToolFailureAccumulator {
    tool_name: String,
    error_class: Option<ToolFailureClass>,
    pattern: String,
    input_shape: String,
    occurrence_count: usize,
    run_ids: HashMap<String, ()>,
    ineffective_retry_count: usize,
    first_seen: Option<DateTime<Utc>>,
    last_seen: Option<DateTime<Utc>>,
}

impl ToolFailureAccumulator {
    fn into_pattern(self) -> ToolFailurePattern {
        let mut run_ids = self.run_ids.into_keys().collect::<Vec<_>>();
        run_ids.sort();
        ToolFailurePattern {
            tool_name: self.tool_name,
            error_class: self.error_class.unwrap_or(ToolFailureClass::Permanent),
            pattern: self.pattern,
            input_shape: self.input_shape,
            occurrence_count: self.occurrence_count,
            distinct_run_count: run_ids.len(),
            run_ids,
            ineffective_retry_count: self.ineffective_retry_count,
            first_seen: self.first_seen,
            last_seen: self.last_seen,
        }
    }
}

impl ErrorAccumulator {
    fn update_time(&mut self, started_at: DateTime<Utc>, finished_at: Option<DateTime<Utc>>) {
        if self
            .first_seen
            .as_ref()
            .is_none_or(|first| started_at < *first)
        {
            self.first_seen = Some(started_at);
        }
        if let Some(fa) = finished_at
            && self.last_seen.as_ref().is_none_or(|last| fa > *last)
        {
            self.last_seen = Some(fa);
        }
    }
}

/// 为模式分组归一化错误信息：转小写、去首尾空白、合并连续空白。
fn normalize_error(msg: &str) -> String {
    let lower = msg.to_lowercase();
    // 将连续的空白字符合并为一个空格
    let mut result = String::with_capacity(lower.len());
    let mut prev_was_space = false;
    for ch in lower.trim().chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(ch);
            prev_was_space = false;
        }
    }
    result
}

/// 在没有结构化失败分类时，根据错误信息的关键字启发式地分类失败原因。
fn classify_tool_failure(message: &str) -> ToolFailureClass {
    let normalized = message.to_lowercase();
    if normalized.contains("timeout")
        || normalized.contains("timed out")
        || normalized.contains("超时")
    {
        ToolFailureClass::Timeout
    } else if normalized.contains("permission denied")
        || normalized.contains("access denied")
        || normalized.contains("not permitted")
        || normalized.contains("权限不足")
        || normalized.contains("拒绝访问")
    {
        ToolFailureClass::Permanent
    } else if normalized.contains("not found")
        || normalized.contains("no such file")
        || normalized.contains("未找到")
        || normalized.contains("不存在")
        || normalized.contains("invalid parameter")
        || normalized.contains("missing parameter")
        || normalized.contains("invalid argument")
        || normalized.contains("无效参数")
        || normalized.contains("缺少参数")
    {
        ToolFailureClass::InvalidArguments
    } else if normalized.contains("network")
        || normalized.contains("connection")
        || normalized.contains("dns")
        || normalized.contains("网络")
        || normalized.contains("连接")
    {
        ToolFailureClass::Transient
    } else if normalized.contains("dependency")
        || normalized.contains("module")
        || normalized.contains("package")
        || normalized.contains("依赖")
        || normalized.contains("模块")
        || normalized.contains("软件包")
    {
        ToolFailureClass::Unavailable
    } else if normalized.contains("cancelled")
        || normalized.contains("canceled")
        || normalized.contains("取消")
    {
        ToolFailureClass::Cancelled
    } else {
        ToolFailureClass::Permanent
    }
}

/// 计算 JSON 值的结构形状（不含具体取值），用于分组时避免暴露原始参数。
fn json_shape(value: Option<&serde_json::Value>) -> String {
    json_shape_at(value, 0)
}

/// `json_shape` 的递归实现，`depth` 用于限制递归深度，避免无界嵌套。
fn json_shape_at(value: Option<&serde_json::Value>, depth: usize) -> String {
    if depth >= 4 {
        return "nested".to_string();
    }
    match value {
        None | Some(serde_json::Value::Null) => "none".to_string(),
        Some(serde_json::Value::Bool(_)) => "bool".to_string(),
        Some(serde_json::Value::Number(_)) => "number".to_string(),
        Some(serde_json::Value::String(_)) => "string".to_string(),
        Some(serde_json::Value::Array(values)) => {
            let item_shape = values
                .first()
                .map(|value| json_shape_at(Some(value), depth.saturating_add(1)))
                .unwrap_or_else(|| "empty".to_string());
            format!("array<{item_shape}>")
        }
        Some(serde_json::Value::Object(values)) => {
            let mut fields = values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{key}:{}",
                        json_shape_at(Some(value), depth.saturating_add(1))
                    )
                })
                .collect::<Vec<_>>();
            fields.sort();
            fields.truncate(24);
            format!("object{{{}}}", fields.join(","))
        }
    }
}

/// 计算 JSON 值的稳定哈希标识，用于识别“同一次工具调用参数”的重复出现。
fn json_identity(value: Option<&serde_json::Value>) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_json_value(value, 0, &mut hasher);
    format!("{:016x}", hasher.finish())
}

/// 递归地将 JSON 值写入哈希器，`depth` 用于限制递归深度。
fn hash_json_value<H: Hasher>(value: Option<&serde_json::Value>, depth: usize, hasher: &mut H) {
    if depth >= 8 {
        "nested".hash(hasher);
        return;
    }
    match value {
        None | Some(serde_json::Value::Null) => "null".hash(hasher),
        Some(serde_json::Value::Bool(value)) => value.hash(hasher),
        Some(serde_json::Value::Number(value)) => value.to_string().hash(hasher),
        Some(serde_json::Value::String(value)) => value.hash(hasher),
        Some(serde_json::Value::Array(values)) => {
            "array".hash(hasher);
            values.len().hash(hasher);
            for value in values.iter().take(32) {
                hash_json_value(Some(value), depth.saturating_add(1), hasher);
            }
        }
        Some(serde_json::Value::Object(values)) => {
            "object".hash(hasher);
            values.len().hash(hasher);
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_by_key(|(left, _)| *left);
            for (key, value) in fields.into_iter().take(32) {
                key.hash(hasher);
                hash_json_value(Some(value), depth.saturating_add(1), hasher);
            }
        }
    }
}

/// 将一次工具失败记录归并进对应的失败模式累加器，并统计是否属于无效重试。
fn record_tool_failure(
    patterns: &mut HashMap<String, ToolFailureAccumulator>,
    attempts: &mut HashMap<String, usize>,
    run: &crate::trace::Run,
    call: Option<&ToolCallContext>,
    event_tool_name: &str,
    message: &str,
    failure: Option<&crate::tools::ToolFailure>,
) {
    let tool_name = call
        .map(|context| context.tool_name.as_str())
        .unwrap_or(event_tool_name);
    let input_shape = call
        .map(|context| context.input_shape.as_str())
        .unwrap_or("unknown");
    let args_key = call.map(|context| context.args_key.as_str()).unwrap_or("");
    let error_class = failure
        .map(|failure| failure.category)
        .unwrap_or_else(|| classify_tool_failure(message));
    let pattern = normalize_error(message)
        .chars()
        .take(200)
        .collect::<String>();
    let grouping_pattern = "classified";
    let key = format!("{tool_name}|{error_class:?}|{input_shape}|{grouping_pattern}");
    let attempt_key = format!("{key}|{args_key}");
    let prior_attempts = attempts.entry(attempt_key).or_insert(0);
    let is_retry = *prior_attempts > 0;
    *prior_attempts = prior_attempts.saturating_add(1);

    let accumulator = patterns.entry(key).or_default();
    accumulator.tool_name = tool_name.to_string();
    accumulator.error_class = Some(error_class);
    accumulator.pattern = pattern;
    accumulator.input_shape = input_shape.to_string();
    accumulator.occurrence_count = accumulator.occurrence_count.saturating_add(1);
    accumulator.run_ids.insert(run.run_id.clone(), ());
    if is_retry {
        accumulator.ineffective_retry_count = accumulator.ineffective_retry_count.saturating_add(1);
    }
    accumulator.first_seen = match accumulator.first_seen {
        Some(first_seen) => Some(first_seen.min(run.started_at)),
        None => Some(run.started_at),
    };
    let last_seen = run.finished_at.unwrap_or(run.started_at);
    accumulator.last_seen = match accumulator.last_seen {
        Some(current) => Some(current.max(last_seen)),
        None => Some(last_seen),
    };
}

// ── 单元测试 ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{InMemoryRunStore, Run};

    fn make_run(id: &str, session: &str, status: RunStatus) -> Run {
        Run {
            run_id: id.to_string(),
            parent_run_id: None,
            agent_name: String::new(),
            model: String::new(),
            provider: None,
            turn_id: None,
            execution_id: None,
            session_id: session.to_string(),
            status,
            input: "test input".to_string(),
            events: vec![
                RunEvent::LlmCall {
                    messages: 1,
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    cached_prompt_tokens: 80,
                    cache_creation_prompt_tokens: 0,
                    usage_reported: true,
                    estimated_context_tokens: 95,
                    protected_context_tokens: 20,
                    protected_message_count: 1,
                    context_limit_tokens: 0,
                    context_breakdown: Default::default(),
                    cache_fingerprint: Default::default(),
                    duration_ms: 200,
                },
                RunEvent::ToolCall {
                    call_id: "c1".into(),
                    name: "read_file".into(),
                    args: None,
                    risk: None,
                    duration_ms: 50,
                },
                RunEvent::ToolResult {
                    call_id: "c1".into(),
                    name: "read_file".into(),
                    success: true,
                    output_preview: Some("ok".into()),
                    output_truncated: false,
                    duration_ms: 50,
                    original_bytes: 0,
                    returned_bytes: 0,
                    estimated_tokens: 0,
                    output_handling: None,
                    artifact: None,
                },
            ],
            final_output: if status == RunStatus::Completed {
                Some("ok".to_string())
            } else {
                None
            },
            error: if status == RunStatus::Failed {
                Some("something went wrong".to_string())
            } else {
                None
            },
            token_usage: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                ..Default::default()
            },
            timings: crate::trace::RunTimings {
                total_duration_ms: 300,
                llm_duration_ms: 200,
                tool_duration_ms: 100,
            },
            started_at: Utc::now(),
            finished_at: Some(Utc::now()),
        }
    }

    fn make_failed_run_with_tool_error(id: &str, session: &str) -> Run {
        Run {
            run_id: id.to_string(),
            parent_run_id: None,
            agent_name: String::new(),
            model: String::new(),
            provider: None,
            turn_id: None,
            execution_id: None,
            session_id: session.to_string(),
            status: RunStatus::Failed,
            input: "test".to_string(),
            events: vec![
                RunEvent::ToolCall {
                    call_id: "c2".into(),
                    name: "write_file".into(),
                    args: None,
                    risk: None,
                    duration_ms: 150,
                },
                RunEvent::ToolError {
                    call_id: "c2".into(),
                    name: "write_file".into(),
                    message: "Permission denied: cannot write to /etc/config".to_string(),
                    failure: None,
                },
            ],
            final_output: None,
            error: Some("something went wrong".to_string()),
            token_usage: TokenUsage {
                prompt_tokens: 80,
                completion_tokens: 40,
                total_tokens: 120,
                ..Default::default()
            },
            timings: crate::trace::RunTimings {
                total_duration_ms: 200,
                llm_duration_ms: 0,
                tool_duration_ms: 150,
            },
            started_at: Utc::now(),
            finished_at: Some(Utc::now()),
        }
    }

    #[tokio::test]
    async fn test_summarize_session() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_run("r1", "s1", RunStatus::Completed))
            .await
            .unwrap();
        store
            .save(make_run("r2", "s1", RunStatus::Failed))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        let summary = analyzer.summarize_session("s1").await.unwrap();
        assert_eq!(summary.session_id, "s1");
        assert_eq!(summary.run_count, 2);
        assert_eq!(summary.completed_count, 1);
        assert_eq!(summary.failed_count, 1);
        assert_eq!(summary.total_prompt_tokens, 200);
        assert_eq!(summary.total_tokens, 300);
        assert!(summary.tools_used.contains(&"read_file".to_string()));
        assert_eq!(summary.llm_call_count, 2);
    }

    #[tokio::test]
    async fn test_tool_usage_stats() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_run("r1", "s1", RunStatus::Completed))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        let stats = analyzer.tool_usage_stats(100).await.unwrap();
        assert!(!stats.is_empty());

        let read_file_stat = stats.iter().find(|s| s.name == "read_file").unwrap();
        assert_eq!(read_file_stat.call_count, 1);
        assert_eq!(read_file_stat.success_count, 1);
        assert_eq!(read_file_stat.total_duration_ms, 50);
    }

    #[tokio::test]
    async fn skipped_invocations_are_not_reported_as_tool_usage_or_failures()
    -> crate::error::Result<()> {
        let store = Arc::new(InMemoryRunStore::new());
        let mut run = make_run("skipped-run", "skipped-session", RunStatus::Completed);
        run.events = vec![
            RunEvent::ToolCall {
                call_id: "future-wave".to_string(),
                name: "write_file".to_string(),
                args: None,
                risk: None,
                duration_ms: 0,
            },
            RunEvent::ToolExecutionSkipped {
                call_id: "future-wave".to_string(),
                name: "write_file".to_string(),
                reason: "cancelled before execution".to_string(),
            },
            RunEvent::ToolResult {
                call_id: "future-wave".to_string(),
                name: "write_file".to_string(),
                success: false,
                output_preview: Some(String::new()),
                output_truncated: false,
                duration_ms: 0,
                original_bytes: 0,
                returned_bytes: 0,
                estimated_tokens: 0,
                output_handling: None,
                artifact: None,
            },
            RunEvent::ToolError {
                call_id: "future-wave".to_string(),
                name: "write_file".to_string(),
                message: "not executed".to_string(),
                failure: None,
            },
        ];
        store.save(run).await?;
        let analyzer = TraceAnalyzer::new(store);
        let session = analyzer.summarize_session("skipped-session").await?;
        assert!(session.tools_used.is_empty());
        assert!(analyzer.tool_usage_stats(10).await?.is_empty());
        let reliability = analyzer.tool_reliability_report(10, None).await?;
        assert_eq!(reliability.total_calls, 0);
        assert_eq!(reliability.failure_count, 0);
        assert!(analyzer.error_pattern_analysis(10).await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_token_usage_breakdown() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_run("r1", "s1", RunStatus::Completed))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        let breakdown = analyzer.token_usage_breakdown(100).await.unwrap();
        assert_eq!(breakdown.prompt_tokens, 100);
        assert_eq!(breakdown.completion_tokens, 50);
        assert_eq!(breakdown.total_tokens, 150);
        assert!(breakdown.per_run.contains_key("r1"));
        assert!(breakdown.per_llm_call.contains_key("r1"));
    }

    #[tokio::test]
    async fn test_find_slow_tools() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_run("r1", "s1", RunStatus::Completed))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        // read_file 的平均耗时为 50ms，所以 threshold=40 时应该包含它
        let slow = analyzer.find_slow_tools(40, 100).await.unwrap();
        assert!(slow.iter().any(|s| s.name == "read_file"));

        // threshold=100 时应该排除 read_file（平均耗时=50）
        let not_slow = analyzer.find_slow_tools(100, 100).await.unwrap();
        assert!(not_slow.is_empty());
    }

    #[tokio::test]
    async fn test_error_pattern_analysis() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_failed_run_with_tool_error("r1", "s1"))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        let patterns = analyzer.error_pattern_analysis(100).await.unwrap();
        assert!(!patterns.is_empty());

        /* "permission denied" 只出现在 ToolError 事件中，不在运行级别的
         * 错误信息（"something went wrong"）里，所以次数为 1。 */
        let perm_pattern = patterns
            .iter()
            .find(|p| p.pattern.contains("permission denied"))
            .unwrap();
        assert_eq!(perm_pattern.occurrence_count, 1);
        assert!(
            perm_pattern
                .associated_tools
                .contains(&"write_file".to_string())
        );
    }

    #[tokio::test]
    async fn tool_reliability_groups_cross_run_failures_and_retries() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryRunStore::new());
        let args = serde_json::json!({"path": "/protected/config"});
        let mut first = make_run("r1", "s1", RunStatus::Failed);
        first.events = vec![
            RunEvent::ToolCall {
                call_id: "c1".into(),
                name: "write_file".into(),
                args: Some(args.clone()),
                risk: None,
                duration_ms: 10,
            },
            RunEvent::ToolResult {
                call_id: "c1".into(),
                name: "write_file".into(),
                success: false,
                output_preview: Some("Permission denied: /protected/config".into()),
                output_truncated: false,
                duration_ms: 10,
                original_bytes: 0,
                returned_bytes: 0,
                estimated_tokens: 0,
                output_handling: None,
                artifact: None,
            },
            RunEvent::ToolError {
                call_id: "c1".into(),
                name: "write_file".into(),
                message: "Permission denied: /protected/config".into(),
                failure: None,
            },
            RunEvent::ToolCall {
                call_id: "c2".into(),
                name: "write_file".into(),
                args: Some(args),
                risk: None,
                duration_ms: 10,
            },
            RunEvent::ToolResult {
                call_id: "c2".into(),
                name: "write_file".into(),
                success: false,
                output_preview: Some("Permission denied: /protected/config".into()),
                output_truncated: false,
                duration_ms: 10,
                original_bytes: 0,
                returned_bytes: 0,
                estimated_tokens: 0,
                output_handling: None,
                artifact: None,
            },
            RunEvent::ToolError {
                call_id: "c2".into(),
                name: "write_file".into(),
                message: "Permission denied: /protected/config".into(),
                failure: None,
            },
        ];
        let mut second = make_run("r2", "s1", RunStatus::Failed);
        second.events = vec![
            RunEvent::ToolCall {
                call_id: "c3".into(),
                name: "write_file".into(),
                args: Some(serde_json::json!({"path": "/other/config"})),
                risk: None,
                duration_ms: 10,
            },
            RunEvent::ToolResult {
                call_id: "c3".into(),
                name: "write_file".into(),
                success: false,
                output_preview: Some("Access denied: /other/config".into()),
                output_truncated: false,
                duration_ms: 10,
                original_bytes: 0,
                returned_bytes: 0,
                estimated_tokens: 0,
                output_handling: None,
                artifact: None,
            },
            RunEvent::ToolError {
                call_id: "c3".into(),
                name: "write_file".into(),
                message: "Access denied: /other/config".into(),
                failure: None,
            },
        ];
        store.save(first).await?;
        store.save(second).await?;

        let analyzer = TraceAnalyzer::new(store);
        let usage = analyzer.tool_usage_stats(100).await?;
        let write_stats = usage
            .iter()
            .find(|stats| stats.name == "write_file")
            .ok_or_else(|| crate::error::ReactError::Other("missing write_file stats".into()))?;
        assert_eq!(write_stats.failure_count, 3);

        let report = analyzer.tool_reliability_report(100, None).await?;
        assert_eq!(report.total_calls, 3);
        assert_eq!(report.failure_count, 3);
        assert_eq!(report.ineffective_retry_count, 1);
        let pattern = report
            .failure_patterns
            .first()
            .ok_or_else(|| crate::error::ReactError::Other("missing failure pattern".into()))?;
        assert_eq!(pattern.error_class, ToolFailureClass::Permanent);
        assert_eq!(pattern.occurrence_count, 3);
        assert_eq!(pattern.distinct_run_count, 2);
        assert_eq!(pattern.ineffective_retry_count, 1);
        assert!(!pattern.input_shape.contains("protected"));
        Ok(())
    }

    #[tokio::test]
    async fn tool_reliability_prefers_structured_failure_category() -> crate::error::Result<()> {
        let store = Arc::new(InMemoryRunStore::new());
        let mut run = make_run("structured", "s1", RunStatus::Failed);
        run.events = vec![
            RunEvent::ToolCall {
                call_id: "c1".into(),
                name: "web_search".into(),
                args: Some(serde_json::json!({"query": "rust"})),
                risk: None,
                duration_ms: 5,
            },
            RunEvent::ToolError {
                call_id: "c1".into(),
                name: "web_search".into(),
                message: "permission denied wording from an upstream proxy".into(),
                failure: Some(crate::tools::ToolFailure::new(
                    crate::tools::ToolFailureCategory::Transient,
                )),
            },
        ];
        store.save(run).await?;

        let report = TraceAnalyzer::new(store)
            .tool_reliability_report(10, None)
            .await?;
        let pattern = report
            .failure_patterns
            .first()
            .ok_or_else(|| crate::error::ReactError::Other("missing failure pattern".into()))?;

        assert_eq!(pattern.error_class, ToolFailureClass::Transient);
        Ok(())
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let store = Arc::new(InMemoryRunStore::new());
        store
            .save(make_run("r1", "s1", RunStatus::Completed))
            .await
            .unwrap();
        store
            .save(make_run("r2", "s2", RunStatus::Completed))
            .await
            .unwrap();

        let analyzer = TraceAnalyzer::new(store);
        let sessions = analyzer.list_sessions(100).await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&"s1".to_string()));
        assert!(sessions.contains(&"s2".to_string()));
    }

    #[test]
    fn test_normalize_error() {
        assert_eq!(
            normalize_error("  Permission   DENIED:  /etc  "),
            "permission denied: /etc"
        );
        assert_eq!(normalize_error("timeout"), "timeout");
        assert_eq!(normalize_error(""), "");
    }
}
