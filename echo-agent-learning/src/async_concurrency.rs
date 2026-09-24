//! Tokio 任务、有界 channel 以及结构化的结果收集。

use crate::errors::LearningError;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::time::{Duration, timeout};

/// 将 `Duration` 转换为毫秒数；超出 `u64` 表示范围时饱和为 `u64::MAX`，不会 panic。
fn duration_millis_saturated(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// 并发运行多个子任务（subagent），通过有界 channel 收集各自的完成结果并按字典序排序返回。
pub async fn run_subagents(names: Vec<String>) -> Result<Vec<String>, LearningError> {
    let capacity = names.len().max(1);
    let (sender, mut receiver) = mpsc::channel(capacity);
    let mut handles = Vec::with_capacity(names.len());

    for name in names {
        let sender = sender.clone();
        handles.push(tokio::spawn(async move {
            sender
                .send(format!("{name}: completed"))
                .await
                .map_err(|_| LearningError::ChannelClosed)
        }));
    }
    drop(sender);

    for handle in handles {
        handle
            .await
            .map_err(|error| LearningError::SubagentJoin(error.to_string()))??;
    }

    let mut results = Vec::new();
    while let Some(result) = receiver.recv().await {
        results.push(result);
    }
    results.sort();
    Ok(results)
}

/// 在给定的 `deadline` 之内等待一次休眠（`delay`）完成；超时则返回 `LearningError::TimedOut`。
pub async fn complete_within(
    delay: Duration,
    deadline: Duration,
) -> Result<&'static str, LearningError> {
    let deadline_ms = duration_millis_saturated(deadline);
    timeout(deadline, tokio::time::sleep(delay))
        .await
        .map_err(|_| LearningError::TimedOut(deadline_ms))?;
    Ok("completed")
}

/// 等待给定时长或直到收到取消信号，两者以先到者为准；若已被取消或期间收到取消信号则返回错误。
pub async fn wait_or_cancel(
    delay: Duration,
    mut cancelled: watch::Receiver<bool>,
) -> Result<&'static str, LearningError> {
    if *cancelled.borrow() {
        return Err(LearningError::Cancelled);
    }
    tokio::select! {
        _ = tokio::time::sleep(delay) => Ok("completed"),
        changed = cancelled.changed() => {
            changed.map_err(|_| LearningError::ChannelClosed)?;
            if *cancelled.borrow() {
                Err(LearningError::Cancelled)
            } else {
                Ok("cancel signal cleared")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证每个子任务的结果都会被收集到，且结果按字典序排序。
    #[tokio::test]
    async fn collects_each_subagent_result() -> Result<(), LearningError> {
        let results = run_subagents(vec!["writer".to_string(), "reviewer".to_string()]).await?;
        assert_eq!(results.len(), 2);
        assert_eq!(
            results.first().map(String::as_str),
            Some("reviewer: completed")
        );
        Ok(())
    }

    /// 验证超时错误与取消错误是两种不同的、可区分的错误类型。
    #[tokio::test]
    async fn timeout_and_cancellation_have_distinct_errors() {
        assert!(matches!(
            complete_within(Duration::from_millis(20), Duration::from_millis(1)).await,
            Err(LearningError::TimedOut(_))
        ));

        let (sender, receiver) = watch::channel(false);
        let task = tokio::spawn(wait_or_cancel(Duration::from_secs(1), receiver));
        assert!(sender.send(true).is_ok());
        let result = task
            .await
            .map_err(|error| LearningError::SubagentJoin(error.to_string()));
        assert!(matches!(result, Ok(Err(LearningError::Cancelled))));
    }

    /// 验证超出 `u64` 范围的时长转换会饱和为最大值，而不是发生截断。
    #[test]
    fn duration_conversion_saturates_instead_of_truncating() {
        assert_eq!(duration_millis_saturated(Duration::MAX), u64::MAX);
    }
}
