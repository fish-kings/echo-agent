/*!
结构化输出（Structured Output）支持

提供一个 `StructuredAgent<T>` 包装器，在 Agent 执行完成后，
自动将文本输出解析为类型 `T`。

# 用法

```rust,no_run
use echo_agent::prelude::*;
use echo_agent::agent::react::structured::StructuredAgent;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Sentiment { label: String, score: f64 }

# async fn run() -> echo_agent::error::Result<()> {
let agent = ReactAgentBuilder::simple("qwen3-max", "You are a sentiment analysis assistant, always output in JSON format")?;
let mut sa = StructuredAgent::<Sentiment>::new(agent);
let result = sa.execute("I love Rust!").await?;
println!("{}: {}", result.label, result.score);
# Ok(())
# }
```
*/

use crate::agent::Agent;
use crate::error::{ReactError, Result};

use super::ReactAgent;

/**
自动将 Agent 文本输出解析为类型 `T` 的包装器。

要求 Agent 输出合法的 JSON 文本（可以通过系统提示词约束，
也可以配合 `ResponseFormat::JsonSchema` 使用）。
*/
pub struct StructuredAgent<T> {
    /// 被包装的底层 Agent。
    inner: ReactAgent,
    /// 用于携带目标类型 `T` 的占位标记。
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StructuredAgent<T>
where
    T: serde::de::DeserializeOwned + Send + 'static,
{
    /// 包装一个已有的 ReactAgent。
    pub fn new(agent: ReactAgent) -> Self {
        Self {
            inner: agent,
            _phantom: std::marker::PhantomData,
        }
    }

    /// 获取内部 Agent 的不可变引用。
    pub fn inner(&self) -> &ReactAgent {
        &self.inner
    }

    /// 获取内部 Agent 的可变引用。
    pub fn inner_mut(&mut self) -> &mut ReactAgent {
        &mut self.inner
    }

    /// 执行一个任务，并将结果解析为类型 `T`。
    pub async fn execute(&mut self, task: &str) -> Result<T> {
        let text = self.inner.execute(task).await?;
        parse_json_output(&text)
    }

    /// 进行一次对话，并将结果解析为类型 `T`。
    pub async fn chat(&mut self, message: &str) -> Result<T> {
        let text = self.inner.chat(message).await?;
        parse_json_output(&text)
    }
}

/**
从 LLM 输出文本中提取并解析 JSON。

支持两种格式：
1. 纯 JSON 字符串
2. 包裹在 Markdown 代码块中的 JSON（```json ... ```）
*/
fn parse_json_output<T: serde::de::DeserializeOwned>(text: &str) -> Result<T> {
    let trimmed = text.trim();

    // 尝试直接解析为 JSON
    if let Ok(v) = serde_json::from_str::<T>(trimmed) {
        return Ok(v);
    }

    // 尝试从 markdown 代码块中提取 JSON
    if let Some(json_str) = extract_json_from_markdown(trimmed)
        && let Ok(v) = serde_json::from_str::<T>(json_str)
    {
        return Ok(v);
    }

    Err(ReactError::Other(format!(
        "Failed to parse LLM output as target type. Raw output:\n{text}"
    )))
}

/// 从 markdown 代码块中提取 JSON 内容。
fn extract_json_from_markdown(text: &str) -> Option<&str> {
    // 匹配 ```json\n...\n``` 或 ```\n...\n```
    let start = if let Some(pos) = text.find("```json") {
        pos + 7
    } else {
        text.find("```")?.checked_add(3)?
    };

    let remaining = &text[start..];
    // 跳过换行符
    let content_start = remaining.find('\n').map(|p| p + 1).unwrap_or(0);
    let content = &remaining[content_start..];

    // 查找结束的 ```
    content.find("```").map(|end| content[..end].trim())
}

/// 结构化输出解析的单元测试。
#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    /// 测试用的结构化输出目标类型。
    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        /// 姓名。
        name: String,
        /// 年龄。
        age: u32,
    }

    /// 验证可直接解析纯 JSON 文本。
    #[test]
    fn test_parse_json_direct() {
        let result: Person = parse_json_output(r#"{"name": "Alice", "age": 30}"#).unwrap();
        assert_eq!(
            result,
            Person {
                name: "Alice".to_string(),
                age: 30
            }
        );
    }

    /// 验证可解析首尾带空白的 JSON 文本。
    #[test]
    fn test_parse_json_with_whitespace() {
        let result: Person = parse_json_output("  \n{\"name\": \"Bob\", \"age\": 25}\n  ").unwrap();
        assert_eq!(result.name, "Bob");
    }

    /// 验证可从 Markdown 代码块中解析 JSON。
    #[test]
    fn test_parse_json_from_markdown() {
        let text = r#"Here is the result:
```json
{"name": "Charlie", "age": 35}
```
"#;
        let result: Person = parse_json_output(text).unwrap();
        assert_eq!(result.name, "Charlie");
        assert_eq!(result.age, 35);
    }

    /// 验证非 JSON 文本解析失败。
    #[test]
    fn test_parse_json_failure() {
        let result = parse_json_output::<Person>("not json at all");
        assert!(result.is_err());
    }

    /// 验证可从 Markdown 代码块中提取 JSON 字符串。
    #[test]
    fn test_extract_json_from_markdown() {
        let text = "```json\n{\"a\": 1}\n```";
        assert_eq!(extract_json_from_markdown(text), Some("{\"a\": 1}"));
    }

    /// 验证没有 Markdown 代码块时返回 None。
    #[test]
    fn test_extract_json_no_markdown() {
        assert_eq!(extract_json_from_markdown("{\"a\": 1}"), None);
    }
}
