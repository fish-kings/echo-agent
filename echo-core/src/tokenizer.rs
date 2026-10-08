/*!
Token 估算 trait 与用量统计。

为上下文管理器提供可插拔的 token 计数能力，替代固定的 `chars / 4` 启发式估算。

# 内置实现

| 类型 | 算法 | 精度 |
|------|----------|----------|
| [`HeuristicTokenizer`] | ASCII 权重 1，CJK 权重 2，总权重 / 4 | 中（推荐用于中英文混合场景） |
| [`SimpleTokenizer`] | `byte_count / 4 + 1` | 低（向后兼容） |

# 用量统计

[`TokenUsageTracker`] 提供跨请求的 token 累计统计，
类似 Claude Code / ChatGPT 的 token 用量展示能力。

# 扩展

实现 [`Tokenizer`] trait 即可接入精确的分词器（例如 tiktoken-rs）。
*/

/// Token 计数器抽象
pub trait Tokenizer: Send + Sync {
    /// 估算输入文本将消耗多少模型 token。
    fn count_tokens(&self, text: &str) -> usize;
}

impl Tokenizer for Box<dyn Tokenizer> {
    /// 委托给被装箱的内部分词器进行计数。
    fn count_tokens(&self, text: &str) -> usize {
        (**self).count_tokens(text)
    }
}

/**
基于字符权重估算 token 数的启发式分词器。

**注意：这只是一个粗略的估算器，而不是精确的 token 计数器。**

估算规则：
- ASCII 字符权重为 1（约 4 个字符 = 1 个 token）
- CJK 及其他非 ASCII 字符权重为 2（约 1-2 个字符 = 1 个 token）
- 总权重 / 4 即为估算的 token 数
- 空字符串返回 0

与 `byte_count / 4` 相比，对 CJK 内容的估算精度提升约 40-60%，
但仍不应用于需要精确 token 计数的场景（例如配额管理、计费等）。
如需精确计数，请使用 tiktoken 或模型原生的分词器。
*/
pub struct HeuristicTokenizer;

impl Tokenizer for HeuristicTokenizer {
    /// 按字符权重估算 token 数，非空文本至少返回 1。
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let weight: usize = text.chars().map(|c| if c.is_ascii() { 1 } else { 2 }).sum();
        (weight / 4).max(1)
    }
}

/// 简单分词器：`byte_count / 4 + 1`（与旧行为向后兼容）
pub struct SimpleTokenizer;

impl Tokenizer for SimpleTokenizer {
    /// 按字节数估算 token 数，空文本返回 0。
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            text.len() / 4 + 1
        }
    }
}

/**
自校准分词器包装器，利用实际 API 响应数据提升估算精度。

包装任意 [`Tokenizer`] 实现，并用一个校准系数调整其估算值；该系数由
（LLM API 报告的）实际 token 数与估算 token 数之比推导而来。

# 工作原理

1. `count_tokens()` 返回 `inner.estimate × calibration_factor`
2. 每次 LLM 调用后，通过 `calibrate()` 回馈实际 token 数
3. 该系数是一个指数移动平均（EMA），会随时间逐渐收敛

# 示例

```rust
use echo_core::tokenizer::{CalibratedTokenizer, HeuristicTokenizer, Tokenizer};
use std::sync::Arc;

let base = Arc::new(HeuristicTokenizer);
let calibrated = CalibratedTokenizer::new(base);

// 初始估算（系数 = 1.0）
let est = calibrated.count_tokens("hello world");

// API 返回实际数量后：
calibrated.calibrate(est, 15); // 实际为 15 个 token

// 后续的估算值已被调整
let adjusted = calibrated.count_tokens("another text");
```
*/
pub struct CalibratedTokenizer {
    /// 被包装的基础分词器。
    inner: std::sync::Arc<dyn Tokenizer>,
    /**
    校准系数，以 f64 位模式存储在 AtomicU64 中，以实现无锁访问。
    系数 = actual_tokens / estimated_tokens（经 EMA 平滑）。
    */
    factor_bits: std::sync::atomic::AtomicU64,
    /// 已接收的校准样本数量。
    sample_count: std::sync::atomic::AtomicU64,
    /**
    EMA 平滑系数（0.0 = 不更新，1.0 = 完全替换）。
    默认值 0.3 能提供良好的收敛性。
    */
    ema_alpha: f64,
}

impl CalibratedTokenizer {
    /**
    创建一个包装给定基础分词器的校准分词器。

    初始校准系数为 1.0（不做调整）。
    */
    pub fn new(inner: std::sync::Arc<dyn Tokenizer>) -> Self {
        Self {
            inner,
            factor_bits: std::sync::atomic::AtomicU64::new(1.0_f64.to_bits()),
            sample_count: std::sync::atomic::AtomicU64::new(0),
            ema_alpha: 0.3,
        }
    }

    /**
    使用自定义的 EMA alpha（平滑系数）创建。

    - `alpha = 0.1`：收敛慢，更稳定
    - `alpha = 0.3`：均衡（默认）
    - `alpha = 0.5`：收敛快，响应更灵敏
    */
    pub fn with_alpha(inner: std::sync::Arc<dyn Tokenizer>, alpha: f64) -> Self {
        Self {
            inner,
            factor_bits: std::sync::atomic::AtomicU64::new(1.0_f64.to_bits()),
            sample_count: std::sync::atomic::AtomicU64::new(0),
            ema_alpha: alpha.clamp(0.01, 1.0),
        }
    }

    /**
    回馈 API 响应中的实际 token 数，以改进后续估算。

    - `estimated`：API 调用前 `count_tokens()` 返回的值
    - `actual`：API 响应中的实际 token 数（`usage.prompt_tokens`）
    */
    pub fn calibrate(&self, estimated: usize, actual: u32) {
        if estimated == 0 || actual == 0 {
            return;
        }

        let observed_ratio = actual as f64 / estimated as f64;
        // 将比率限制在合理范围内，避免剧烈波动
        let observed_ratio = observed_ratio.clamp(0.2, 5.0);

        let current_bits = self.factor_bits.load(std::sync::atomic::Ordering::Relaxed);
        let current_factor = f64::from_bits(current_bits);

        // EMA：new_factor = alpha * observed + (1 - alpha) * current
        let new_factor = self.ema_alpha * observed_ratio + (1.0 - self.ema_alpha) * current_factor;

        self.factor_bits
            .store(new_factor.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.sample_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// 获取当前的校准系数。
    pub fn calibration_factor(&self) -> f64 {
        let bits = self.factor_bits.load(std::sync::atomic::Ordering::Relaxed);
        f64::from_bits(bits)
    }

    /// 获取已接收的校准样本数量。
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 将校准系数重置为 1.0。
    pub fn reset_calibration(&self) {
        self.factor_bits
            .store(1.0_f64.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.sample_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Tokenizer for CalibratedTokenizer {
    /// 返回基础估算值乘以校准系数后四舍五入的结果。
    fn count_tokens(&self, text: &str) -> usize {
        let base = self.inner.count_tokens(text);
        let factor = self.calibration_factor();
        (base as f64 * factor).round() as usize
    }
}

impl std::fmt::Debug for CalibratedTokenizer {
    /// 输出当前校准系数与样本数。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalibratedTokenizer")
            .field("factor", &self.calibration_factor())
            .field("samples", &self.sample_count())
            .finish()
    }
}

// ── Token 用量统计 ─────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

/**
线程安全的 token 用量统计器。

类似 Claude Code / ChatGPT 的 token 用量展示。

```rust
use echo_core::tokenizer::TokenUsageTracker;

let tracker = TokenUsageTracker::new("gpt-5.5");
tracker.record(1500, 800, Some(2300));

let stats = tracker.summary();
assert_eq!(stats.total_prompt_tokens, 1500);
```
*/
pub struct TokenUsageTracker {
    /// 统计所针对的模型名称。
    model_name: String,
    /// 累计输入（prompt）token 数。
    total_prompt_tokens: AtomicU64,
    /// 累计输出（completion）token 数。
    total_completion_tokens: AtomicU64,
    /// 累计总 token 数。
    total_tokens: AtomicU64,
    /// 累计命中 prompt 缓存的输入 token 数。
    total_cached_prompt_tokens: AtomicU64,
    /// 累计写入 prompt 缓存的输入 token 数。
    total_cache_creation_prompt_tokens: AtomicU64,
    /// 已记录的请求数。
    request_count: AtomicU64,
}

impl TokenUsageTracker {
    /// 为指定模型创建一个计数全部为 0 的统计器。
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            total_cached_prompt_tokens: AtomicU64::new(0),
            total_cache_creation_prompt_tokens: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
        }
    }

    /// 目前已记录的请求数。
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// 记录单次请求的 token 用量。
    pub fn record(&self, prompt: u32, completion: u32, total: Option<u32>) {
        self.total_prompt_tokens
            .fetch_add(prompt as u64, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion as u64, Ordering::Relaxed);
        let t = total.unwrap_or(prompt + completion);
        self.total_tokens.fetch_add(t as u64, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 从 API 响应中记录用量。
    pub fn record_usage(&self, usage: &crate::llm::types::Usage) {
        let prompt = usage.effective_prompt_tokens();
        let completion = usage.completion_tokens.unwrap_or(0);
        self.record(prompt, completion, Some(usage.effective_total_tokens()));
        self.total_cached_prompt_tokens
            .fetch_add(usage.cached_prompt_tokens() as u64, Ordering::Relaxed);
        self.total_cache_creation_prompt_tokens.fetch_add(
            usage.cache_creation_prompt_tokens() as u64,
            Ordering::Relaxed,
        );
    }

    /// 获取用量摘要。
    pub fn summary(&self) -> UsageSummary {
        let total_prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let total_completion = self.total_completion_tokens.load(Ordering::Relaxed);
        let total = self.total_tokens.load(Ordering::Relaxed);
        let total_cached_prompt = self.total_cached_prompt_tokens.load(Ordering::Relaxed);
        let total_cache_creation_prompt = self
            .total_cache_creation_prompt_tokens
            .load(Ordering::Relaxed);
        let requests = self.request_count.load(Ordering::Relaxed);

        UsageSummary {
            model_name: self.model_name.clone(),
            total_prompt_tokens: total_prompt,
            total_completion_tokens: total_completion,
            total_tokens: total,
            total_cached_prompt_tokens: total_cached_prompt,
            total_cache_creation_prompt_tokens: total_cache_creation_prompt,
            request_count: requests,
        }
    }

    /**
    累计 prompt 缓存命中率（0.0–1.0）。

    按所有请求的 `cached_prompt / prompt` 计算。记录的 prompt token 已按
    provider 归一化，包含缓存读取部分。
    若尚未记录任何带 prompt token 的请求，返回 `None`。
    */
    pub fn cumulative_cache_hit_rate(&self) -> Option<f64> {
        let prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let cached = self.total_cached_prompt_tokens.load(Ordering::Relaxed);
        if prompt == 0 {
            None
        } else {
            Some(cached as f64 / prompt as f64)
        }
    }

    /// 重置所有计数器。
    pub fn reset(&self) {
        self.total_prompt_tokens.store(0, Ordering::Relaxed);
        self.total_completion_tokens.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_cached_prompt_tokens.store(0, Ordering::Relaxed);
        self.total_cache_creation_prompt_tokens
            .store(0, Ordering::Relaxed);
        self.request_count.store(0, Ordering::Relaxed);
    }
}

/// Token 用量摘要快照
#[derive(Debug, Clone, Default)]
pub struct UsageSummary {
    /// 模型名称。
    pub model_name: String,
    /// 累计输入（prompt）token 数。
    pub total_prompt_tokens: u64,
    /// 累计输出（completion）token 数。
    pub total_completion_tokens: u64,
    /// 累计总 token 数。
    pub total_tokens: u64,
    /// 累计命中 prompt 缓存的输入 token 数。
    pub total_cached_prompt_tokens: u64,
    /// 累计写入 prompt 缓存的输入 token 数。
    pub total_cache_creation_prompt_tokens: u64,
    /// 请求总数。
    pub request_count: u64,
}

impl std::fmt::Display for UsageSummary {
    /// 以多行可读文本形式输出用量摘要。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Token Usage [{model}]:
  Requests:   {requests}
  Input tokens:  {prompt}
  Cached input:  {cached_prompt}
  Cache writes:  {cache_creation_prompt}
  Output tokens: {completion}
  Total tokens:  {total}",
            model = self.model_name,
            requests = self.request_count,
            prompt = self.total_prompt_tokens,
            cached_prompt = self.total_cached_prompt_tokens,
            cache_creation_prompt = self.total_cache_creation_prompt_tokens,
            completion = self.total_completion_tokens,
            total = self.total_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_heuristic_ascii() {
        let t = HeuristicTokenizer;
        // 16 个 ASCII 字符 → 权重 16 → 16/4 = 4 个 token
        assert_eq!(t.count_tokens("hello world 1234"), 4);
    }

    #[test]
    fn test_heuristic_cjk() {
        let t = HeuristicTokenizer;
        // 4 个非 ASCII 字符 → 权重 8 → 8/4 = 2 个 token
        assert_eq!(t.count_tokens("éñôà"), 2);
    }

    #[test]
    fn test_heuristic_mixed() {
        let t = HeuristicTokenizer;
        // "hello éñ" → 6 个 ASCII(6) + 2 个非 ASCII(4) = 权重 10 → 10/4 = 2
        assert_eq!(t.count_tokens("hello éñ"), 2);
    }

    #[test]
    fn test_heuristic_empty() {
        let t = HeuristicTokenizer;
        assert_eq!(t.count_tokens(""), 0); // 空字符串返回 0
    }

    #[test]
    fn test_simple_tokenizer() {
        let t = SimpleTokenizer;
        assert_eq!(t.count_tokens("hello"), 2); // 5/4+1 = 2
    }

    #[test]
    fn test_calibrated_tokenizer_initial_factor() {
        let base = Arc::new(HeuristicTokenizer);
        let calibrated = CalibratedTokenizer::new(base);

        // 初始系数应为 1.0
        assert!((calibrated.calibration_factor() - 1.0).abs() < f64::EPSILON);
        assert_eq!(calibrated.sample_count(), 0);
    }

    #[test]
    fn test_calibrated_tokenizer_calibration() {
        let base = Arc::new(HeuristicTokenizer);
        let calibrated = CalibratedTokenizer::with_alpha(base.clone(), 1.0); // alpha=1.0 表示立即更新

        let text = "hello world test";
        let estimated = base.count_tokens(text); // HeuristicTokenizer 的估算值
        let actual = estimated * 2; // 模拟实际值为估算值的 2 倍

        calibrated.calibrate(estimated, actual as u32);

        // alpha=1.0 时，系数应直接跳到 2.0
        assert!((calibrated.calibration_factor() - 2.0).abs() < f64::EPSILON);
        assert_eq!(calibrated.sample_count(), 1);

        // 此时 count_tokens 应返回约 2 倍的基础估算值
        let adjusted = calibrated.count_tokens(text);
        assert_eq!(adjusted, estimated * 2);
    }

    #[test]
    fn test_calibrated_tokenizer_ema_smoothing() {
        let base = Arc::new(HeuristicTokenizer);
        let calibrated = CalibratedTokenizer::with_alpha(base.clone(), 0.5);

        let estimated = 100;

        // 第一次校准：actual = 200（比率 = 2.0）
        calibrated.calibrate(estimated, 200);
        let f1 = calibrated.calibration_factor();
        // alpha=0.5 时：0.5 * 2.0 + 0.5 * 1.0 = 1.5
        assert!((f1 - 1.5).abs() < 0.01);

        // 第二次校准：actual 仍为 200（比率 = 2.0）
        calibrated.calibrate(estimated, 200);
        let f2 = calibrated.calibration_factor();
        // alpha=0.5 时：0.5 * 2.0 + 0.5 * 1.5 = 1.75
        assert!((f2 - 1.75).abs() < 0.01);
    }

    #[test]
    fn test_calibrated_tokenizer_reset() {
        let base = Arc::new(HeuristicTokenizer);
        let calibrated = CalibratedTokenizer::new(base);

        calibrated.calibrate(100, 200);
        assert!(calibrated.calibration_factor() > 1.0);

        calibrated.reset_calibration();
        assert!((calibrated.calibration_factor() - 1.0).abs() < f64::EPSILON);
        assert_eq!(calibrated.sample_count(), 0);
    }

    #[test]
    fn test_calibrated_tokenizer_skips_zero() {
        let base = Arc::new(HeuristicTokenizer);
        let calibrated = CalibratedTokenizer::new(base);

        // 遇到 0 值时不应更新
        calibrated.calibrate(0, 100);
        calibrated.calibrate(100, 0);
        assert!((calibrated.calibration_factor() - 1.0).abs() < f64::EPSILON);
        assert_eq!(calibrated.sample_count(), 0);
    }

    #[test]
    fn test_token_usage_tracker_records_prompt_cache_tokens() {
        let tracker = TokenUsageTracker::new("test-model");
        tracker.record_usage(&crate::llm::types::Usage {
            prompt_tokens: Some(1000),
            completion_tokens: Some(20),
            total_tokens: Some(1020),
            prompt_tokens_details: Some(crate::llm::types::TokenUsageDetails {
                cached_tokens: Some(980),
                ..Default::default()
            }),
            ..Default::default()
        });

        let summary = tracker.summary();
        assert_eq!(summary.total_prompt_tokens, 1000);
        assert_eq!(summary.total_cached_prompt_tokens, 980);
        assert_eq!(summary.total_cache_creation_prompt_tokens, 0);
        assert_eq!(tracker.cumulative_cache_hit_rate(), Some(0.98));
    }

    #[test]
    fn test_token_usage_tracker_normalizes_anthropic_cache_reads() {
        let tracker = TokenUsageTracker::new("test-model");
        tracker.record_usage(&crate::llm::types::Usage {
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
            total_tokens: Some(120),
            cache_read_input_tokens: Some(900),
            cache_creation_input_tokens: Some(50),
            ..Default::default()
        });

        let summary = tracker.summary();
        assert_eq!(summary.total_prompt_tokens, 1050);
        assert_eq!(summary.total_tokens, 1070);
        assert_eq!(summary.total_cached_prompt_tokens, 900);
        assert_eq!(summary.total_cache_creation_prompt_tokens, 50);
        assert_eq!(tracker.cumulative_cache_hit_rate(), Some(900.0 / 1050.0));
    }
}
