//! `Pin<Box<Future>>` 为动态分发的 future 提供了稳定的内存位置。

use futures::future::BoxFuture;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// 返回一个装箱的 future：短暂休眠后产出传入的消息，演示 `BoxFuture` 的用法。
pub fn boxed_message(message: String) -> BoxFuture<'static, String> {
    Box::pin(async move {
        tokio::time::sleep(Duration::from_millis(1)).await;
        message
    })
}

/// 返回一个固定（pinned）且装箱的 future，直接产出传入的消息。
pub fn pinned_message(message: String) -> Pin<Box<dyn Future<Output = String> + Send + 'static>> {
    Box::pin(async move { message })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证两种 future 都能被正常 await 并得到预期结果。
    #[tokio::test]
    async fn boxed_future_can_be_awaited() {
        assert_eq!(boxed_message("done".to_string()).await, "done");
        assert_eq!(pinned_message("pinned".to_string()).await, "pinned");
    }
}
