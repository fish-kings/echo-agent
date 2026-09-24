use echo_agent_learning::smart_pointers::pinning::{boxed_message, pinned_message};

/// 演示 `BoxFuture` 与 `Pin<Box<dyn Future>>` 两种异步返回方式的用法。
#[tokio::main]
async fn main() {
    let first = boxed_message("BoxFuture 完成".to_string()).await;
    let second = pinned_message("Pin<Box<Future>> 完成".to_string()).await;
    println!("{first}");
    println!("{second}");
}
