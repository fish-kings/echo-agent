//! 当借入的文本本身已经是规范形式时，`Cow<'a, str>` 可以避免不必要的分配。

use std::borrow::Cow;

/// 去除输入两端空白；若原文本本就没有多余空白，则直接借用而不分配新内存。
pub fn normalize_prompt(input: &str) -> Cow<'_, str> {
    let trimmed = input.trim();
    if trimmed == input {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(trimmed.to_string())
    }
}

/// 只有确实需要修改时，`to_mut` 才会克隆借用的数据；此函数确保字符串以句号结尾。
pub fn ensure_period(mut input: Cow<'_, str>) -> Cow<'_, str> {
    if !input.ends_with('.') {
        input.to_mut().push('.');
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证已规范化的输入保持借用状态，而需要修改的输入会转为拥有所有权。
    #[test]
    fn borrows_clean_input_and_owns_modified_input() {
        assert!(matches!(normalize_prompt("hello"), Cow::Borrowed(_)));
        assert!(matches!(normalize_prompt(" hello "), Cow::Owned(_)));
    }

    /// 验证只有在真正发生修改时，`to_mut` 才会触发一次分配。
    #[test]
    fn to_mut_allocates_only_for_a_change() {
        assert!(matches!(
            ensure_period(Cow::Borrowed("done.")),
            Cow::Borrowed(_)
        ));
        assert_eq!(ensure_period(Cow::Borrowed("done")), "done.");
    }
}
