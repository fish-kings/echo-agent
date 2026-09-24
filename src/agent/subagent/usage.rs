//! 面向子 Agent 派发（dispatch）的 LLM 用量统计累加器。
//!
//! 跟踪单次子 Agent 派发内所有 LLM 调用的累计 token 用量，
//! 字段与 `AgentEvent::LlmUsage` 保持一致。

use serde::{Deserialize, Serialize};

/**
单次子 Agent 派发内，一次或多次 LLM 调用的累计用量快照。

字段与 `AgentEvent::LlmUsage` 对齐。token 计数使用 `u64`，
以便在多次调用累加时安全地避免溢出。
*/
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmUsageStats {
    /// 模型名称（取最近一次看到的值 —— 单次派发内通常都是同一个模型）。
    pub model: String,
    /// 累计的 prompt / 输入 token 数。
    pub prompt_tokens: u64,
    /// 累计的 completion / 输出 token 数。
    pub completion_tokens: u64,
    /// 累计的总 token 数。
    pub total_tokens: u64,
    /// 累计的已缓存 prompt token 数。
    pub cached_prompt_tokens: u64,
    /// 累计的缓存创建 prompt token 数。
    pub cache_creation_prompt_tokens: u64,
    /// 是否有任意一次调用上报了真实的用量元数据。
    /// 一旦被置为 `true`，就会一直保持 `true`（粘滞状态）。
    pub usage_reported: bool,
    /// 已记录的 LLM 调用次数。
    pub call_count: u64,
}

impl LlmUsageStats {
    /**
    记录一次 LLM 用量事件，将其累加到运行总量中。

    `model` 会在每次调用时被覆盖（以最近一次为准），这没有问题，
    因为单次派发通常只会使用一个模型。
    */
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        model: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
        cached_prompt_tokens: usize,
        cache_creation_prompt_tokens: usize,
        usage_reported: bool,
    ) {
        self.model = model.to_string();
        self.prompt_tokens += prompt_tokens as u64;
        self.completion_tokens += completion_tokens as u64;
        self.total_tokens += total_tokens as u64;
        self.cached_prompt_tokens += cached_prompt_tokens as u64;
        self.cache_creation_prompt_tokens += cache_creation_prompt_tokens as u64;
        if usage_reported {
            self.usage_reported = true;
        }
        self.call_count += 1;
    }

    /// 转换为与前端 `cacheUsageFromEvents` 兼容的 JSON payload。
    pub fn to_payload(&self, session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "session_id": session_id,
            "model": if self.model.is_empty() { "unknown" } else { &self.model },
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cache_creation_prompt_tokens": self.cache_creation_prompt_tokens,
            "usage_reported": self.usage_reported,
            "call_count": self.call_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_multiple_calls() {
        let mut stats = LlmUsageStats::default();
        stats.record("claude-x", 100, 50, 150, 80, 10, true);
        stats.record("claude-x", 200, 60, 260, 150, 20, true);
        assert_eq!(stats.prompt_tokens, 300);
        assert_eq!(stats.completion_tokens, 110);
        assert_eq!(stats.total_tokens, 410);
        assert_eq!(stats.cached_prompt_tokens, 230);
        assert_eq!(stats.cache_creation_prompt_tokens, 30);
        assert_eq!(stats.call_count, 2);
        assert!(stats.usage_reported);
    }

    #[test]
    fn usage_reported_stays_false_until_any_true() {
        let mut stats = LlmUsageStats::default();
        stats.record("m", 10, 5, 15, 0, 0, false);
        assert!(!stats.usage_reported);
        stats.record("m", 10, 5, 15, 0, 0, true);
        assert!(stats.usage_reported);
    }

    #[test]
    fn payload_uses_unknown_when_model_empty() {
        let stats = LlmUsageStats::default();
        let p = stats.to_payload("sess-1");
        assert_eq!(p["model"], serde_json::json!("unknown"));
        assert_eq!(p["usage_reported"], serde_json::json!(false));
    }

    #[test]
    fn payload_uses_real_model_when_set() {
        let mut stats = LlmUsageStats::default();
        stats.record("claude-opus-4", 500, 200, 700, 300, 50, true);
        let p = stats.to_payload("sess-2");
        assert_eq!(p["model"], serde_json::json!("claude-opus-4"));
        assert_eq!(p["prompt_tokens"], serde_json::json!(500));
        assert_eq!(p["call_count"], serde_json::json!(1));
    }
}
