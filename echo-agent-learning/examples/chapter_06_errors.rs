use echo_agent_learning::errors::{LearningError, parse_positive_limit, required_text};

/// 校验并解析设置项，返回名称与最大轮数，任一校验失败都会返回 `LearningError`。
fn load_settings(name: Option<&str>, limit: &str) -> Result<(String, usize), LearningError> {
    let name = required_text("name", name)?;
    let limit = parse_positive_limit(limit)?;
    Ok((name, limit))
}

/// 演示错误处理：分别展示校验成功与校验失败两种场景。
fn main() -> Result<(), LearningError> {
    let (name, limit) = load_settings(Some(" assistant "), "8")?;
    println!("名称: {name}, 最大轮数: {limit}");

    match load_settings(None, "not-a-number") {
        Ok(settings) => println!("意外成功: {settings:?}"),
        Err(error) => println!("预期错误: {error}"),
    }
    Ok(())
}
