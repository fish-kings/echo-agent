/*!
上下文压缩 —— trait 及输入/输出类型

具体实现位于 `echo_state::compression::compressor`。
*/

use crate::error::Result;
use crate::llm::types::Message;
use crate::tokenizer::{HeuristicTokenizer, Tokenizer};
use chrono::{DateTime, Utc};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
pub use tokio_util::sync::CancellationToken;

/// 一个可替换的、带标记标签的、投影进模型上下文中的消息。
#[derive(Debug, Clone)]
pub struct ContextProjection {
    /// 稳定的内容标记，用于在多次刷新之间识别同一个投影。
    pub marker: String,
    /// 当前投影出的消息；为 `None` 表示移除一个过期的投影。
    pub message: Option<Message>,
}

/// 提供给“模型调用前上下文投影器”使用的通用运行元数据。
#[derive(Debug, Clone)]
pub struct ProjectionContext {
    /// 当前所处的迭代轮次。
    pub iteration: usize,
    /// 当前 agent 的名称。
    pub agent_name: String,
    /// 当前会话（session）ID，若有。
    pub session_id: Option<String>,
    /// 当前对话（conversation）ID，若有。
    pub conversation_id: Option<String>,
    /// 当前运行（run）ID，若有。
    pub run_id: Option<String>,
    /// 当前轮次（turn）ID，若有。
    pub turn_id: Option<String>,
}

/// 在准备模型输入之前，即时产出可替换上下文的投影器。
pub trait PreModelContextProjector: Send + Sync {
    /// 根据给定的运行上下文，异步生成一组待投影的上下文消息。
    fn project<'a>(
        &'a self,
        context: &'a ProjectionContext,
    ) -> BoxFuture<'a, Result<Vec<ContextProjection>>>;
}

/// 压缩流水线的输入。
pub struct CompressionInput {
    /// 待压缩的消息列表。
    pub messages: Vec<Message>,
    /// token 上限，超出该值时触发压缩。
    pub token_limit: usize,
    /// 当前用户查询 —— 用于保护活跃任务上下文不被淘汰。
    pub current_query: Option<String>,
    /// 聚焦指令 —— 用户提供的、关于摘要应优先保留哪些内容的指导。
    pub focus_instructions: Option<String>,
    /// 本次压缩过程中每次 provider 调用所使用的取消令牌。
    pub cancel_token: Option<CancellationToken>,
    /// 与调用方预算计算共享的分词器。
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl CompressionInput {
    /// 返回配置的分词器；若未配置，回退到框架默认分词器。
    pub fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tokenizer
            .clone()
            .unwrap_or_else(|| Arc::new(HeuristicTokenizer))
    }
}

/// 压缩流水线的输出。
pub struct CompressionOutput {
    /// 保留下来、将发送给 LLM 的最终消息列表。
    pub messages: Vec<Message>,
    /// 本次压缩过程中被淘汰的消息。
    pub evicted: Vec<Message>,
    /// 用于审计、回放和恢复的压缩检查点。
    pub checkpoint: Option<CompressionCheckpoint>,
}

/// 记录为在压缩后维持合法的 tool_call ↔ tool_result 配对关系而做出的修复。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPairFix {
    /// 受影响的工具调用 ID。
    pub tool_call_id: String,
    /// 应用的修复类型。
    pub fix_type: ToolPairFixType,
}

/// 工具调用配对修复的具体类型。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolPairFixType {
    /// 移除了一个孤立的 tool_result（没有对应的 assistant tool_calls）。
    OrphanedResultRemoved,
    /// 清空了某条 assistant 消息中的所有 tool_calls（没有对应的结果）。
    DanglingCallCleared,
    /// 为缺失的 tool_result 插入了一个占位结果。
    PlaceholderResultInserted,
}

/**
捕获一次压缩过程完整结果的一等状态对象。

这是上下文压缩的审计轨迹 —— 记录了做了什么、移除了什么、修复了什么，
以及提升到了长期记忆中的内容。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionCheckpoint {
    /// 该检查点的唯一标识符（UUID v4）。
    pub checkpoint_id: String,
    /// 使用的压缩策略名称（例如 "SlidingWindow"、"Summary"）。
    pub strategy: String,
    /// 本次压缩所覆盖的消息索引范围 [first, last]。
    pub covered_range: Option<(usize, usize)>,
    /// LLM 生成的摘要（如果该策略会产出摘要）。
    pub summary: Option<String>,
    /// 压缩后保留的消息数量。
    pub retained_count: usize,
    /// 压缩过程中被淘汰的消息数量。
    pub evicted_count: usize,
    /// 被排除在压缩之外的受保护消息数量。
    pub protected_count: usize,
    /// 压缩后应用的工具调用配对修复列表。
    pub tool_pair_fixes: Vec<ToolPairFix>,
    /**
    发送给 MemoryPromoter 用于事实提取的被淘汰消息数量。
    注意：这里统计的是被淘汰的消息数，而非实际提取出的事实数（promoter 是异步运行的）。
    */
    pub memory_promotion_count: usize,
    /// 压缩前的估计 token 数。
    pub token_before: usize,
    /// 压缩后的估计 token 数。
    pub token_after: usize,
    /// 本次压缩过程的实际耗时（毫秒）。
    pub compression_duration_ms: u64,
    /// 该检查点的创建时间戳。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub created_at: DateTime<Utc>,
    /// 本次压缩使用的用户提供的聚焦指令（如果有）。
    pub focus_instructions: Option<String>,
    /// 已应用的压缩级别（与 AdaptiveCompressor 相关）。
    pub levels_applied: Vec<String>,
}

impl CompressionCheckpoint {
    /// 创建一个新的检查点，使用随机 UUID 与当前时间戳。
    pub fn new(strategy: impl Into<String>) -> Self {
        Self {
            checkpoint_id: uuid::Uuid::new_v4().to_string(),
            strategy: strategy.into(),
            covered_range: None,
            summary: None,
            retained_count: 0,
            evicted_count: 0,
            protected_count: 0,
            tool_pair_fixes: Vec::new(),
            memory_promotion_count: 0,
            token_before: 0,
            token_after: 0,
            compression_duration_ms: 0,
            created_at: Utc::now(),
            focus_instructions: None,
            levels_applied: Vec::new(),
        }
    }

    /// 构建器风格：设置覆盖的消息范围。
    pub fn with_covered_range(mut self, first: usize, last: usize) -> Self {
        self.covered_range = Some((first, last));
        self
    }

    /// 构建器风格：设置摘要文本。
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// 构建器风格：设置压缩前后的 token 数。
    pub fn with_tokens(mut self, before: usize, after: usize) -> Self {
        self.token_before = before;
        self.token_after = after;
        self
    }

    /// 构建器风格：设置保留与淘汰的消息数量。
    pub fn with_counts(mut self, retained: usize, evicted: usize) -> Self {
        self.retained_count = retained;
        self.evicted_count = evicted;
        self
    }

    /// 构建器风格：设置受保护的消息数量。
    pub fn with_protected_count(mut self, count: usize) -> Self {
        self.protected_count = count;
        self
    }

    /// 构建器风格：设置记忆提升数量。
    pub fn with_memory_promotion_count(mut self, count: usize) -> Self {
        self.memory_promotion_count = count;
        self
    }

    /// 构建器风格：设置工具调用配对修复列表。
    pub fn with_tool_fixes(mut self, fixes: Vec<ToolPairFix>) -> Self {
        self.tool_pair_fixes = fixes;
        self
    }

    /// 构建器风格：设置压缩耗时。
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.compression_duration_ms = ms;
        self
    }

    /// 构建器风格：设置聚焦指令。
    pub fn with_focus(mut self, focus: Option<String>) -> Self {
        self.focus_instructions = focus;
        self
    }

    /// 构建器风格：设置已应用的压缩级别（用于 AdaptiveCompressor）。
    pub fn with_levels(mut self, levels: Vec<String>) -> Self {
        self.levels_applied = levels;
        self
    }
}

/**
由基于 LLM 的压缩器产出的结构化摘要。

与自由文本摘要不同，这里每个字段都跟踪对话的一个特定方面。这样可以实现：
- **字段级合并**，用于增量压缩（不会出现摘要漂移）
- **可验证性** —— 可独立检查每个字段
- **可编程访问** —— 提取文件路径、待办任务等信息
*/
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct StructuredSummary {
    /// 用户的主要目标与意图。
    pub goal: String,
    /// 当前正在处理的具体任务。
    pub current_task: String,
    /// 已完成的操作。
    pub completed_actions: Vec<String>,
    /// 仍需完成的任务。
    pub pending_tasks: Vec<String>,
    /// 对话过程中做出的决策。
    pub decisions: Vec<String>,
    /// 被引用或修改过的文件路径。
    pub files_touched: Vec<String>,
    /// 遇到的错误及其解决方式。
    pub errors: Vec<String>,
    /// 工具输出中关键发现的摘要。
    pub tool_outputs_summary: String,
    /// 发现的用户偏好或约束。
    pub user_preferences: Vec<String>,
    /// 不可协商的需求、安全边界与验收标准。
    pub constraints: Vec<String>,
    /// 对后续任务延续至关重要的精确标识符、数值、版本号、命令等事实。
    pub key_facts: Vec<String>,
    /// 建议的下一步操作。
    pub next_step: String,
}

impl StructuredSummary {
    /// 从 JSON 字符串（LLM 响应）解析。
    pub fn from_json(json: &str) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// 尝试从可能包含 markdown 代码围栏或多余文本的 LLM 响应中提取并解析 JSON。
    pub fn from_llm_response(response: &str) -> Option<Self> {
        // 先尝试直接解析
        if let Ok(s) = Self::from_json(response) {
            return Some(s);
        }
        // 尝试从 markdown 代码围栏中提取
        if let Some(json) = extract_json_from_text(response)
            && let Ok(s) = Self::from_json(&json)
        {
            return Some(s);
        }
        None
    }

    /// 序列化为紧凑的 JSON 字符串以便存储。
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// 格式化为系统消息内容：`[对话历史摘要]\n{json}`
    pub fn to_system_message(&self) -> String {
        format!("[对话历史摘要]\n{}", self.to_json())
    }

    /**
    将较新的摘要信息合并进当前摘要（按字段级别）。

    由 `IncrementalSummaryCompressor` 用于更新缓存的摘要，同时不丢失
    早期压缩过程中的信息。
    */
    pub fn merge_with(&mut self, newer: &StructuredSummary) {
        // goal：跟踪最新目标（以最新为准，避免噪声累积）
        if !newer.goal.is_empty() && newer.goal != self.goal {
            self.goal = newer.goal.clone();
        }

        // current_task：以最新值为准
        if !newer.current_task.is_empty() {
            self.current_task = newer.current_task.clone();
        }

        // completed_actions：追加新增项并去重
        for action in &newer.completed_actions {
            if !self.completed_actions.contains(action) {
                self.completed_actions.push(action.clone());
            }
        }

        // pending_tasks：移除已完成的项，添加新任务（但排除已完成的任务）
        let completed_set: std::collections::HashSet<&String> =
            newer.completed_actions.iter().collect();
        // 判断保留哪些任务时，也要考虑我们自己已有的 completed_actions
        let all_completed: std::collections::HashSet<&String> =
            self.completed_actions.iter().collect();
        self.pending_tasks.retain(|t| !completed_set.contains(t));
        for task in &newer.pending_tasks {
            if !self.pending_tasks.contains(task)
                && !completed_set.contains(task)
                && !all_completed.contains(task)
            {
                self.pending_tasks.push(task.clone());
            }
        }

        // decisions：追加新增项并去重
        for decision in &newer.decisions {
            if !self.decisions.contains(decision) {
                self.decisions.push(decision.clone());
            }
        }

        // files_touched：规范化路径，合并并去重
        for file in &newer.files_touched {
            let normalized = normalize_path(file);
            if !self.files_touched.contains(&normalized) {
                self.files_touched.push(normalized);
            }
        }

        // errors：追加新增项并去重
        for error in &newer.errors {
            if !self.errors.contains(error) {
                self.errors.push(error.clone());
            }
        }

        // tool_outputs_summary：以最新值为准
        if !newer.tool_outputs_summary.is_empty() {
            self.tool_outputs_summary = newer.tool_outputs_summary.clone();
        }

        // user_preferences：追加新增项并去重
        for pref in &newer.user_preferences {
            if !self.user_preferences.contains(pref) {
                self.user_preferences.push(pref.clone());
            }
        }

        // constraints 和 key_facts 是持久锚点：只追加并去重。
        for constraint in &newer.constraints {
            if !self.constraints.contains(constraint) {
                self.constraints.push(constraint.clone());
            }
        }
        for fact in &newer.key_facts {
            if !self.key_facts.contains(fact) {
                self.key_facts.push(fact.clone());
            }
        }

        // next_step：以最新值为准
        if !newer.next_step.is_empty() {
            self.next_step = newer.next_step.clone();
        }
    }
}

/**
跟踪应在压缩过程中幸存下来的权威上下文来源。

压缩淘汰消息时，关键上下文（系统提示词、项目规则、技能注入内容）可能
会丢失。`CanonicalContext` 保存这些权威来源，以便在压缩后重新注入。
*/
#[derive(Debug, Clone, Default)]
pub struct CanonicalContext {
    /// 基础系统提示词（不含技能/规则扩展部分）。
    pub system_prompt: Option<String>,
    /// 项目级规则（AGENT.md / RULES.md 内容）。
    pub project_rules: Option<String>,
    /// 来自已激活技能的系统提示词注入内容。
    pub skill_injections: Vec<String>,
    /// 当前已激活技能的名称。
    pub active_skill_names: Vec<String>,
}

impl CanonicalContext {
    /// 若配置了任意一种权威上下文来源，返回 true。
    pub fn has_any(&self) -> bool {
        self.system_prompt.is_some()
            || self.project_rules.is_some()
            || !self.skill_injections.is_empty()
    }

    /**
    构建压缩后插入的补充性重新注入消息。

    基础系统提示词由 `ContextManager` 直接负责恢复；本方法只返回
    补充性的权威上下文，以避免提示词被重复表示。
    若没有需要重新注入的内容，返回 `None`。
    */
    pub fn to_reinjection_messages(&self) -> Option<Vec<String>> {
        if !self.has_any() {
            return None;
        }

        let mut msgs: Vec<String> = Vec::new();

        // 将项目规则作为独立的系统消息重新注入
        if let Some(ref rules) = self.project_rules {
            let truncated: String = rules.chars().take(2000).collect();
            msgs.push(format!(
                "[Canonical context — project rules restored]:\n{}",
                truncated
            ));
        }

        // 列出已激活技能，供 LLM 感知
        if !self.active_skill_names.is_empty() {
            msgs.push(format!(
                "[Canonical context — active skills]: {}",
                self.active_skill_names.join(", ")
            ));
        }

        Some(msgs)
    }
}

/// 规范化文件路径以便去重：去除开头的 `./`，并去除首尾空白。
fn normalize_path(path: &str) -> String {
    let trimmed = path.trim();
    if let Some(stripped) = trimmed.strip_prefix("./") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

/// 从可能包含 markdown 代码围栏或多余文本的文本中提取 JSON 内容。
fn extract_json_from_text(text: &str) -> Option<String> {
    // 尝试 ```json ... ``` 代码围栏
    if let Some(start) = text.find("```json") {
        let after_fence = text.get(start.saturating_add(7)..)?;
        if let Some(end) = after_fence.find("```") {
            return after_fence.get(..end).map(|value| value.trim().to_string());
        }
    }
    // 尝试 ``` ... ``` 代码围栏
    if let Some(start) = text.find("```") {
        let after_fence = text.get(start.saturating_add(3)..)?;
        if let Some(end) = after_fence.find("```") {
            let candidate = after_fence.get(..end)?.trim().to_string();
            if candidate.starts_with('{') {
                return Some(candidate);
            }
        }
    }
    // 尝试查找 { ... } 代码块
    if let Some(start) = text.find('{')
        && let Some(end) = text.rfind('}')
    {
        let candidate = text.get(start..=end)?.to_string();
        if candidate.chars().count() >= 10 {
            return Some(candidate);
        }
    }
    None
}

/// 所有压缩策略的统一接口（异步，支持 `dyn` trait object）。
pub trait ContextCompressor: Send + Sync {
    /// 对给定输入执行一次压缩，返回保留/淘汰的消息与压缩检查点。
    fn compress(&self, input: CompressionInput) -> BoxFuture<'_, Result<CompressionOutput>>;

    /**
    该压缩器的人类可读名称，用于指标跟踪。
    实现方可覆盖此方法以提供更具描述性的名称。
    */
    fn name(&self) -> &str {
        "custom"
    }
}

/**
允许将 `Box<dyn ContextCompressor>` 直接传给任何接受
`impl ContextCompressor` 的函数，而无需引入额外的包装枚举。
*/
impl ContextCompressor for Box<dyn ContextCompressor> {
    /// 转发到被装箱的具体压缩器实现。
    fn compress(&self, input: CompressionInput) -> BoxFuture<'_, Result<CompressionOutput>> {
        (**self).compress(input)
    }

    /// 转发到被装箱的具体压缩器实现的名称。
    fn name(&self) -> &str {
        (**self).name()
    }
}

#[cfg(test)]
mod tests {
    use super::StructuredSummary;

    #[test]
    fn structured_summary_accepts_partial_provider_json() -> Result<(), serde_json::Error> {
        let summary = StructuredSummary::from_json(
            r#"{"goal":"ship compaction","constraints":["preserve transcript"]}"#,
        )?;

        assert_eq!(summary.goal, "ship compaction");
        assert_eq!(summary.constraints, vec!["preserve transcript"]);
        assert!(summary.key_facts.is_empty());
        assert!(summary.pending_tasks.is_empty());
        Ok(())
    }

    #[test]
    fn structured_summary_merge_preserves_constraints_and_key_facts() {
        let mut previous = StructuredSummary {
            constraints: vec!["preserve transcript".to_string()],
            key_facts: vec!["window=128000".to_string()],
            ..StructuredSummary::default()
        };
        let newer = StructuredSummary {
            constraints: vec![
                "preserve transcript".to_string(),
                "keep recent raw messages".to_string(),
            ],
            key_facts: vec!["window=128000".to_string(), "keep_recent=20".to_string()],
            ..StructuredSummary::default()
        };

        previous.merge_with(&newer);

        assert_eq!(
            previous.constraints,
            vec!["preserve transcript", "keep recent raw messages"]
        );
        assert_eq!(previous.key_facts, vec!["window=128000", "keep_recent=20"]);
    }
}
