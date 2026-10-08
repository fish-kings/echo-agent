/*!
`run_core_loop` 及其各阶段函数共用的 channel 发送宏。

这些宏封装了 `mpsc::Sender` 的使用方式，使得各阶段函数无需在
每个调用点都重复编写“channel 关闭时提前返回”的样板代码。

这三个宏都会在接收端已消失时，用 `return Ok(())` 使**外层函数**
短路返回。因此每个阶段函数都返回 `Result<…>`，这个短路意味着
“优雅地放弃这个流”。
*/

/*
── 面向阶段函数（Phase-fn）的变体 ────────────────────────────────────

阶段函数返回的是 `Result<SomeOutcome>` 而非 `Result<()>`，因此上面
那些直接用 `return Ok(())` 短路的宏无法使用：类型检查通不过。
`_or!` 系列宏额外接受一个 `$bail` 表达式，改用 `return Ok($bail)`
短路返回 —— 通常是该阶段结果枚举中的 `…::Abandoned` 变体。
*/

/**
带背压（backpressure）地发送一个有序事件。若接收端已关闭，
则通过该阶段特定的结果值放弃本次调用。
*/
macro_rules! yield_event_or {
    ($tx:expr, $event:expr, $bail:expr) => {
        if $tx.send(Ok($event)).await.is_err() {
            return Ok($bail);
        }
    };
}

/// 类似 `yield_final_event!`，但在接收端被丢弃时返回 `Ok($bail)`。
macro_rules! yield_final_event_or {
    ($tx:expr, $event:expr, $bail:expr) => {
        if $tx.send(Ok($event)).await.is_err() {
            return Ok($bail);
        }
    };
}

/// 可靠地转发一个错误，并使该阶段函数返回 `Ok($bail)`。
macro_rules! try_send_or {
    ($tx:expr, $fallible:expr, $bail:expr) => {
        match $fallible {
            Ok(v) => v,
            Err(e) => {
                let error: crate::error::ReactError = e.into();
                let _ = $tx
                    .send(Ok(crate::agent::AgentEvent::from_error(
                        "react_loop",
                        &error,
                    )))
                    .await;
                return Ok($bail);
            }
        }
    };
}

pub(crate) use try_send_or;
pub(crate) use yield_event_or;
pub(crate) use yield_final_event_or;
