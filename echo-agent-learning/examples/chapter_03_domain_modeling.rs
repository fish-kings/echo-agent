use echo_agent_learning::basics::{LearningTask, completed_titles, unicode_preview};
use echo_agent_learning::errors::LearningError;

/// 演示使用 struct/enum 建模学习任务，并安全处理中文与 Emoji 文本。
fn main() -> Result<(), LearningError> {
    let pending = LearningTask::new("理解 struct 和 enum")?;
    let mut completed = LearningTask::new("安全处理中文与🦀")?;
    completed.complete("使用 chars() 而不是字节索引");

    let tasks = vec![pending, completed];
    println!("预览: {}", unicode_preview("你好，Rust🦀", 6));
    println!("已完成: {:?}", completed_titles(&tasks));
    Ok(())
}
