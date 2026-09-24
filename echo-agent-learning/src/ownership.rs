//! 移动、借用、克隆以及显式生命周期标注。

/// 取得 vector 的所有权，追加一个标签后再将其返回。
pub fn append_label(mut labels: Vec<String>, label: impl Into<String>) -> Vec<String> {
    labels.push(label.into());
    labels
}

/// 借用字符串而非取得其所有权。
pub fn character_count(value: &str) -> usize {
    value.chars().count()
}

/// 返回的引用可能来自任意一个入参，因此需要显式标注生命周期。
pub fn first_non_empty<'a>(primary: &'a str, fallback: &'a str) -> &'a str {
    if primary.trim().is_empty() {
        fallback
    } else {
        primary
    }
}

/// 当调用方需要一份独立所有权的值时，显式地进行克隆。
pub fn owned_title(value: &str) -> String {
    value.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证借用不会转移所有权，原变量在借用之后依然可用。
    #[test]
    fn borrowing_keeps_the_original_available() {
        let title = String::from("学习所有权");
        assert_eq!(character_count(&title), 5);
        assert_eq!(title, "学习所有权");
    }
}
