use echo_agent_learning::errors::{LearningError, parse_positive_limit};
use echo_agent_learning::traits::{
    LearningTaskBuilder, PlainFormatter, StatusFormatter, TaskFormatter, format_dynamic,
};

/// 演示 trait、泛型约束以及通过 trait object 实现动态分发。
fn main() -> Result<(), LearningError> {
    let limit = parse_positive_limit("8")?;
    let task = LearningTaskBuilder::new()
        .title("理解 trait object")
        .running(true)
        .build()?;
    let formatters: Vec<Box<dyn TaskFormatter>> =
        vec![Box::new(PlainFormatter), Box::new(StatusFormatter)];

    println!("限制: {limit}");
    for line in format_dynamic(&formatters, &task) {
        println!("{line}");
    }
    Ok(())
}
