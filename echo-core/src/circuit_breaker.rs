/*!
LLM 熔断器

当 LLM 服务持续不可用时，防止 Agent 陷入徒劳的重试循环。

## 状态机

```text
Closed ──(连续失败次数 ≥ failure_threshold)──→ Open
                                                       │
                                                  (超时之后)
                                                       ↓
Closed ←──(成功次数 ≥ success_threshold)── HalfOpen
Open   ←──(失败)──────────────────────────── HalfOpen
```

- **Closed**：正常运行，记录失败次数
- **Open**：拒绝所有请求，超时后转为 HalfOpen
- **HalfOpen**：允许有限数量的探测请求，据此决定恢复还是重新熔断
*/

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// 熔断器配置
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// 进入 Open 状态所需的连续失败次数阈值
    pub failure_threshold: u32,
    /// HalfOpen 状态下恢复为 Closed 所需的连续成功次数阈值
    pub success_threshold: u32,
    /// Open 状态的持续时间；超时后自动转为 HalfOpen
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    /// 默认：连续失败 5 次熔断，连续成功 2 次恢复，熔断持续 60 秒。
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        }
    }
}

/// 熔断器内部状态
#[derive(Debug)]
enum State {
    /// 正常状态，记录连续失败次数
    Closed {
        /// 当前的连续失败次数。
        consecutive_failures: u32,
    },
    /// 熔断（跳闸）状态，记录熔断开始的时间
    Open {
        /// 进入 Open 状态的时刻。
        opened_at: Instant,
    },
    /// 半开状态，记录连续成功次数
    HalfOpen {
        /// 当前的连续成功次数。
        consecutive_successes: u32,
    },
}

/**
熔断器

线程安全，可在多个异步任务间共享（`Arc<CircuitBreaker>`）。
*/
pub struct CircuitBreaker {
    /// 当前的熔断状态。
    state: Mutex<State>,
    /// 熔断器配置。
    config: CircuitBreakerConfig,
    /// Open 状态下被拒绝的请求数（用于监控）。
    rejected_count: std::sync::atomic::AtomicU32,
    /**
    HalfOpen 期间当前在途的探测请求数。
    将并发探测限制为 1，以避免恢复时出现惊群效应。
    */
    probes_in_flight: std::sync::atomic::AtomicU32,
}

/**
一次受熔断保护的操作所持有的准入许可。

drop 一个尚未结算的半开许可会释放其探测名额。这样取消和提前返回都是安全的，
不需要每个调用方都记得调用对应的记账方法。
*/
pub struct CircuitPermit {
    /// 签发该许可的熔断器。
    breaker: Arc<CircuitBreaker>,
    /// 该许可是否占用了 HalfOpen 状态下的探测名额。
    half_open_probe: bool,
    /// 是否已通过 `success`/`failure` 结算。
    settled: bool,
}

impl CircuitPermit {
    /// 将已准入的操作结算为成功。
    pub fn success(mut self) {
        self.breaker.record_success();
        self.settled = true;
    }

    /// 将已准入的操作结算为失败。
    pub fn failure(mut self) {
        self.breaker.record_failure();
        self.settled = true;
    }
}

impl Drop for CircuitPermit {
    /// 未结算的半开探测许可在 drop 时释放探测名额。
    fn drop(&mut self) {
        if !self.settled && self.half_open_probe {
            self.breaker.release_probe();
        }
    }
}

impl CircuitBreaker {
    /// 使用显式配置创建熔断器。
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Mutex::new(State::Closed {
                consecutive_failures: 0,
            }),
            config,
            rejected_count: std::sync::atomic::AtomicU32::new(0),
            probes_in_flight: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// 使用默认配置创建熔断器。
    pub fn default_config() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /**
    纯查询：检查当前是否处于 Open 状态（不改变任何状态）。

    仅用于监控和日志；不会触发状态转换。
    */
    pub fn is_open(&self) -> bool {
        let state = self.state.lock();
        matches!(&*state, State::Open { .. })
    }

    /**
    尝试推进状态：若处于 Open 且已超时，则自动转为 HalfOpen。

    若请求应被拒绝（仍处于 Open 状态）返回 `true`，
    若可以继续处理（Closed 或 HalfOpen）返回 `false`。

    HalfOpen 状态下最多允许 1 个并发探测请求，以避免恢复期间出现惊群效应。

    调用方应在请求完成后调用 `record_success()` 或 `record_failure()`，
    它们会自动释放探测名额。
    若请求被拒绝，调用方应调用 `record_rejected()` 更新统计。
    */
    pub fn try_advance(&self) -> bool {
        self.try_admit().is_err()
    }

    /// 尝试准入一次请求：`Ok(true)` 表示占用了半开探测名额，`Ok(false)` 表示正常放行，`Err(())` 表示拒绝。
    fn try_admit(&self) -> std::result::Result<bool, ()> {
        let mut state = self.state.lock();
        match &*state {
            State::Closed { .. } => Ok(false),
            State::HalfOpen { .. } => self
                .probes_in_flight
                .compare_exchange(
                    0,
                    1,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .map(|_| true)
                .map_err(|_| ()),
            State::Open { opened_at } => {
                if opened_at.elapsed() >= self.config.timeout {
                    info!(
                        timeout_secs = self.config.timeout.as_secs(),
                        "Circuit breaker timeout elapsed, transitioning to HalfOpen"
                    );
                    *state = State::HalfOpen {
                        consecutive_successes: 0,
                    };
                    // 进入 HalfOpen 后的第一个探测请求
                    self.probes_in_flight
                        .store(1, std::sync::atomic::Ordering::Release);
                    Ok(true)
                } else {
                    Err(())
                }
            }
        }
    }

    /**
    为一次操作获取自有的准入许可；当熔断器处于 Open 或已有另一个半开
    探测在进行时则拒绝（返回 `None`）。
    */
    pub fn acquire(self: &Arc<Self>) -> Option<CircuitPermit> {
        match self.try_admit() {
            Ok(half_open_probe) => Some(CircuitPermit {
                breaker: Arc::clone(self),
                half_open_probe,
                settled: false,
            }),
            Err(()) => {
                self.record_rejected();
                None
            }
        }
    }

    /// 记录一次被拒绝的请求（处于 Open 状态时）。
    pub fn record_rejected(&self) {
        self.rejected_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// 获取被拒绝请求的总数（用于监控）。
    pub fn rejected_count(&self) -> u32 {
        self.rejected_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 记录一次成功调用。
    pub fn record_success(&self) {
        self.release_probe();

        let mut state = self.state.lock();
        match &*state {
            State::HalfOpen {
                consecutive_successes,
            } => {
                let new_count = consecutive_successes + 1;
                if new_count >= self.config.success_threshold {
                    info!(
                        successes = new_count,
                        threshold = self.config.success_threshold,
                        "✅ Circuit breaker recovered, transitioning to Closed"
                    );
                    *state = State::Closed {
                        consecutive_failures: 0,
                    };
                } else {
                    *state = State::HalfOpen {
                        consecutive_successes: new_count,
                    };
                }
            }
            State::Closed { .. } => {
                *state = State::Closed {
                    consecutive_failures: 0,
                };
            }
            State::Open { .. } => {}
        }
    }

    /// 记录一次失败调用。
    pub fn record_failure(&self) {
        self.release_probe();

        let mut state = self.state.lock();
        match &*state {
            State::Closed {
                consecutive_failures,
            } => {
                let new_count = consecutive_failures.saturating_add(1);
                if new_count >= self.config.failure_threshold {
                    warn!(
                        failures = new_count,
                        threshold = self.config.failure_threshold,
                        "Circuit breaker opened due to consecutive failures"
                    );
                    *state = State::Open {
                        opened_at: Instant::now(),
                    };
                } else {
                    *state = State::Closed {
                        consecutive_failures: new_count,
                    };
                }
            }
            State::HalfOpen { .. } => {
                warn!("Circuit breaker re-opened after HalfOpen probe failed");
                *state = State::Open {
                    opened_at: Instant::now(),
                };
            }
            State::Open { .. } => {
                *state = State::Open {
                    opened_at: Instant::now(),
                };
            }
        }
    }

    /// 释放一个半开探测名额（计数不会低于 0）。
    fn release_probe(&self) {
        self.probes_in_flight
            .fetch_update(
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Acquire,
                |v| if v > 0 { Some(v - 1) } else { Some(0) },
            )
            .ok();
    }

    /// 获取当前状态的可读名称（用于日志/监控）。
    pub fn state_name(&self) -> &'static str {
        let state = self.state.lock();
        match &*state {
            State::Closed { .. } => "closed",
            State::Open { .. } => "open",
            State::HalfOpen { .. } => "half_open",
        }
    }

    /// 获取当前的连续失败次数（仅在 Closed 状态下有意义）。
    pub fn consecutive_failures(&self) -> u32 {
        let state = self.state.lock();
        match &*state {
            State::Closed {
                consecutive_failures,
            } => *consecutive_failures,
            _ => 0,
        }
    }
}

impl std::fmt::Debug for CircuitBreaker {
    /// 输出状态名称与配置。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.state_name())
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// 阈值小、超时短的测试配置。
    fn fast_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_millis(10),
        }
    }

    #[test]
    fn test_initial_state_closed() {
        let cb = CircuitBreaker::new(fast_config());
        assert!(!cb.is_open());
        assert_eq!(cb.state_name(), "closed");
    }

    #[test]
    fn test_opens_after_threshold_failures() {
        let cb = CircuitBreaker::new(fast_config());
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        cb.record_failure();
        assert!(cb.is_open());
        assert_eq!(cb.state_name(), "open");
    }

    #[test]
    fn test_success_resets_failure_count() {
        let cb = CircuitBreaker::new(fast_config());
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
    }

    #[test]
    fn test_transitions_to_half_open_after_timeout() {
        let cb = CircuitBreaker::new(fast_config());
        for _ in 0..3 {
            cb.record_failure();
        }
        assert!(cb.is_open());
        std::thread::sleep(Duration::from_millis(20));
        // try_advance 会从 Open 转为 HalfOpen
        assert!(!cb.try_advance());
        assert_eq!(cb.state_name(), "half_open");
    }

    #[test]
    fn test_try_advance_rejects_while_open() {
        let cb = CircuitBreaker::new(fast_config());
        for _ in 0..3 {
            cb.record_failure();
        }
        // 仍在超时时间内，应当拒绝
        assert!(cb.try_advance());
        assert!(cb.is_open());
    }

    #[test]
    fn test_rejected_count_tracking() {
        let cb = CircuitBreaker::new(fast_config());
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.rejected_count(), 0);
        cb.record_rejected();
        cb.record_rejected();
        cb.record_rejected();
        assert_eq!(cb.rejected_count(), 3);
    }

    #[test]
    fn test_recovers_after_half_open_successes() {
        let cb = CircuitBreaker::new(fast_config());
        for _ in 0..3 {
            cb.record_failure();
        }
        std::thread::sleep(Duration::from_millis(20));
        cb.try_advance(); // 转为 HalfOpen
        cb.record_success();
        assert_eq!(cb.state_name(), "half_open");
        cb.record_success();
        assert_eq!(cb.state_name(), "closed");
    }

    #[test]
    fn test_reopens_on_half_open_failure() {
        let cb = CircuitBreaker::new(fast_config());
        for _ in 0..3 {
            cb.record_failure();
        }
        std::thread::sleep(Duration::from_millis(20));
        cb.try_advance(); // 转为 HalfOpen
        cb.record_failure();
        assert_eq!(cb.state_name(), "open");
    }

    #[test]
    fn dropped_half_open_permit_releases_probe_slot() {
        let cb = Arc::new(CircuitBreaker::new(fast_config()));
        for _ in 0..3 {
            cb.record_failure();
        }
        std::thread::sleep(Duration::from_millis(20));
        let permit = cb.acquire();
        assert!(permit.is_some());
        assert!(cb.acquire().is_none());
        drop(permit);
        assert!(cb.acquire().is_some());
    }
}
