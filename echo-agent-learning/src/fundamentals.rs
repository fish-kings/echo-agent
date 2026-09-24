//! 值、标量与复合类型、表达式、控制流以及切片。

/// 一个带类型的汇总结果，演示整数累加与浮点数输出。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreSummary {
    /// 分数中的最小值。
    pub minimum: u32,
    /// 分数中的最大值。
    pub maximum: u32,
    /// 所有分数的总和。
    pub total: u64,
    /// 所有分数的平均值。
    pub average: f64,
}

/// 汇总一个借用的分数切片；空切片没有意义明确的最小值，因此返回 `None`。
pub fn summarize_scores(scores: &[u32]) -> Option<ScoreSummary> {
    let (&first, rest) = scores.split_first()?;
    let mut minimum = first;
    let mut maximum = first;
    let mut total = u64::from(first);

    for &score in rest {
        minimum = minimum.min(score);
        maximum = maximum.max(score);
        total = total.checked_add(u64::from(score))?;
    }

    Some(ScoreSummary {
        minimum,
        maximum,
        total,
        average: total as f64 / scores.len() as f64,
    })
}

/// 描述下一步应如何处理某次尝试的决策结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptDecision {
    /// 应当开始第一次尝试。
    Start,
    /// 应当重试，并携带剩余可尝试的次数。
    Retry { remaining: u32 },
    /// 已达到最大尝试次数，应当停止。
    Stop,
}

/// `if` 是一个表达式：每个分支都会产出被返回的枚举值。
pub fn decide_attempt(attempt: u32, max_attempts: u32) -> AttemptDecision {
    if max_attempts == 0 || attempt >= max_attempts {
        AttemptDecision::Stop
    } else if attempt == 0 {
        AttemptDecision::Start
    } else {
        AttemptDecision::Retry {
            remaining: max_attempts.saturating_sub(attempt),
        }
    }
}

/// 数组的长度是固定的；切片可以借用任意一段连续序列，按索引安全访问。
pub fn safe_item<T>(items: &[T], index: usize) -> Option<&T> {
    items.get(index)
}

/// 元组可以将一组含义可能不同的固定数量的值组合在一起；本函数交换二元组的两个分量。
pub fn swap_coordinates((x, y): (i32, i32)) -> (i32, i32) {
    (y, x)
}

/// 使用区间配合迭代器适配器，往往比使用可变的 `while` 循环更清晰；本函数生成倒计时序列。
pub fn countdown(start: u32) -> Vec<u32> {
    (1..=start).rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证非空切片能被正确汇总，空切片返回 `None`。
    #[test]
    fn summarizes_non_empty_slice() {
        let summary = summarize_scores(&[10, 20, 30]);
        assert_eq!(summary.map(|value| value.total), Some(60));
        assert_eq!(summary.map(|value| value.average), Some(20.0));
        assert_eq!(summarize_scores(&[]), None);
    }

    /// 验证控制流表达式能正确返回对应的领域枚举值。
    #[test]
    fn control_flow_returns_domain_values() {
        assert_eq!(decide_attempt(0, 3), AttemptDecision::Start);
        assert_eq!(
            decide_attempt(1, 3),
            AttemptDecision::Retry { remaining: 2 }
        );
        assert_eq!(decide_attempt(3, 3), AttemptDecision::Stop);
    }

    /// 验证越界的切片访问会安全地返回 `None`，而不会 panic。
    #[test]
    fn slice_access_never_indexes_out_of_bounds() {
        let values = ["one", "two"];
        assert_eq!(safe_item(&values, 1), Some(&"two"));
        assert_eq!(safe_item(&values, 2), None);
    }
}
