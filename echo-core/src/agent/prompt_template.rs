/*!
提示词模板管理系统

提供一个集中式的 `PromptTemplateManager`，用于注册、存储以及渲染带变量替换功能的
提示词模板。模板使用 `{{variable_name}}` 语法，支持嵌套变量解析、默认值以及
条件区块。

# 快速开始

```rust
use echo_core::agent::prompt_template::PromptTemplateManager;

let manager = PromptTemplateManager::new();
manager.register("greeting", "Hello, {{name}}! Welcome to {{project}}.");

let result = manager.render("greeting", &[("name", "Alice"), ("project", "EchoAgent")]);
assert_eq!(result.unwrap(), "Hello, Alice! Welcome to EchoAgent.");
```

# 模板语法

- **变量**：`{{variable_name}}` —— 替换为提供的值。
- **默认值**：`{{variable_name:default_value}}` —— 未提供该变量时使用
  `default_value`。
- **条件块**：`{{#if variable_name}}...{{#endif}}` —— 仅当 `variable_name`
  存在且非空时才包含该块。
- **带备选分支的条件块**：`{{#if variable_name}}...{{#else}}...{{#endif}}`
  —— 变量存在时包含第一个块，否则包含备选块。
*/

use crate::error::{ReactError, Result};
use std::collections::HashMap;
use std::sync::RwLock;

// ── 模板语法解析 ───────────────────────────────────────────────────

/// 提示词模板字符串中解析出的一个模板元素。
#[derive(Debug, Clone, PartialEq)]
enum TemplateElement {
    /// 字面文本片段（无需替换）。
    Literal(String),
    /// 变量占位符：`{{name}}` 或 `{{name:default}}`。
    Variable {
        /// 变量名。
        name: String,
        /// 变量未提供时使用的默认值。
        default: Option<String>,
    },
    /// 条件块起始标记：`{{#if name}}`。
    ConditionalStart {
        /// 用于判断条件是否成立的变量名。
        name: String,
    },
    /// 条件块的 else 标记：`{{#else}}`。
    ConditionalElse,
    /// 条件块结束标记：`{{#endif}}`。
    ConditionalEnd,
}

/// 将模板字符串解析为一串 `TemplateElement`。
fn parse_template(template: &str) -> Vec<TemplateElement> {
    let mut elements = Vec::new();
    let mut remaining = template;

    while !remaining.is_empty() {
        // 查找下一个 `{{` 标记
        let open_pos = remaining.find("{{");
        match open_pos {
            None => {
                // 没有更多标记；剩余部分全部作为字面文本
                if !remaining.is_empty() {
                    elements.push(TemplateElement::Literal(remaining.to_string()));
                }
                break;
            }
            Some(pos) => {
                // 输出标记之前的字面文本
                if pos > 0 {
                    elements.push(TemplateElement::Literal(remaining[..pos].to_string()));
                }
                // 查找闭合的 `}}`
                let after_open = &remaining[pos + 2..];
                let close_pos = after_open.find("}}");
                match close_pos {
                    None => {
                        // 标记未闭合 —— 将剩余部分整体作为字面文本处理
                        elements.push(TemplateElement::Literal(remaining.to_string()));
                        break;
                    }
                    Some(cpos) => {
                        let tag_content = &after_open[..cpos];
                        let tag_content_trimmed = tag_content.trim();

                        if let Some(if_variable) = tag_content_trimmed.strip_prefix("#if ") {
                            // 条件块起始：{{#if variable_name}}
                            let var_name = if_variable.trim();
                            elements.push(TemplateElement::ConditionalStart {
                                name: var_name.to_string(),
                            });
                        } else if tag_content_trimmed == "#else" {
                            elements.push(TemplateElement::ConditionalElse);
                        } else if tag_content_trimmed == "#endif" {
                            elements.push(TemplateElement::ConditionalEnd);
                        } else if !tag_content_trimmed.is_empty() {
                            // 变量：{{name}} 或 {{name:default}}
                            let colon_pos = tag_content_trimmed.find(':');
                            match colon_pos {
                                None => {
                                    elements.push(TemplateElement::Variable {
                                        name: tag_content_trimmed.to_string(),
                                        default: None,
                                    });
                                }
                                Some(cp) => {
                                    let name = tag_content_trimmed[..cp].trim();
                                    let default = tag_content_trimmed[cp + 1..].trim();
                                    elements.push(TemplateElement::Variable {
                                        name: name.to_string(),
                                        default: Some(default.to_string()),
                                    });
                                }
                            }
                        }
                        // 否则：空的 {{}} 标记，跳过

                        remaining = &after_open[cpos + 2..];
                    }
                }
            }
        }
    }

    elements
}

/// 使用提供的变量渲染一串已解析的模板元素。
fn render_elements(elements: &[TemplateElement], variables: &HashMap<String, String>) -> String {
    let mut result = String::new();
    let mut i = 0;

    while i < elements.len() {
        match &elements[i] {
            TemplateElement::Literal(text) => {
                result.push_str(text);
                i += 1;
            }
            TemplateElement::Variable { name, default } => {
                let value = variables
                    .get(name)
                    .map(|s| s.as_str())
                    .or(default.as_deref());
                result.push_str(value.unwrap_or(""));
                i += 1;
            }
            TemplateElement::ConditionalStart { name } => {
                // 查找匹配的 #else 和 #endif
                let mut else_index = None;
                let mut end_index = None;
                let mut depth = 0;
                for (j, element) in elements.iter().enumerate().skip(i.saturating_add(1)) {
                    match element {
                        TemplateElement::ConditionalStart { .. } => depth += 1,
                        TemplateElement::ConditionalEnd => {
                            if depth == 0 {
                                end_index = Some(j);
                                break;
                            }
                            depth -= 1;
                        }
                        TemplateElement::ConditionalElse if depth == 0 => {
                            else_index = Some(j);
                        }
                        _ => {}
                    }
                }

                let has_value = variables.get(name).is_some_and(|value| !value.is_empty());
                let end = end_index.unwrap_or(elements.len());

                if has_value {
                    // 渲染 #if 与 #else/#endif 之间的块
                    let block_end = else_index.unwrap_or(end);
                    let block = &elements[i + 1..block_end];
                    result.push_str(&render_elements(block, variables));
                } else if let Some(else_idx) = else_index {
                    // 渲染 #else 与 #endif 之间的块
                    let block = &elements[else_idx + 1..end];
                    result.push_str(&render_elements(block, variables));
                }

                i = end.saturating_add(1);
            }
            TemplateElement::ConditionalElse | TemplateElement::ConditionalEnd => {
                /* 这些标记本应在上面的 ConditionalStart 渲染逻辑中被消耗掉。
                如果在顶层遇到它们，直接跳过即可。 */
                i += 1;
            }
        }
    }

    result
}

// ── PromptTemplateManager ──────────────────────────────────────────────────────

/**
集中式提示词模板注册表与渲染引擎。

`PromptTemplateManager` 存储具名的提示词模板，并对其进行变量替换渲染。支持：

- 简单变量替换：`{{name}}`
- 默认值：`{{name:default}}`
- 条件块：`{{#if var}}...{{#else}}...{{#endif}}`

该管理器是线程安全的（内部使用 `RwLock`），通常通过 `Arc<PromptTemplateManager>`
在多个 agent 之间共享。

# 示例

```rust
use echo_core::agent::prompt_template::PromptTemplateManager;
use std::sync::Arc;

let manager = Arc::new(PromptTemplateManager::new());
manager.register("system_prompt", "You are a {{role}} assistant for {{domain}}.");

let rendered = manager.render("system_prompt", &[
    ("role", "coding"),
    ("domain", "Rust development"),
]);
assert_eq!(rendered.unwrap(), "You are a coding assistant for Rust development.");
```
*/
pub struct PromptTemplateManager {
    /// 按名称索引的已注册模板集合。
    templates: RwLock<HashMap<String, String>>,
}

impl PromptTemplateManager {
    /// 创建一个空的模板管理器。
    pub fn new() -> Self {
        Self {
            templates: RwLock::new(HashMap::new()),
        }
    }

    /**
    注册一个具名的提示词模板。

    如果同名模板已存在，则会被覆盖。
    */
    pub fn register(&self, name: &str, template: &str) {
        // 从锁中毒中恢复 —— 数据依然有效
        let mut guard = self.templates.write().unwrap_or_else(|e| e.into_inner());
        guard.insert(name.to_string(), template.to_string());
    }

    /**
    移除一个具名模板。

    若该模板存在并被移除，返回 `true`。
    */
    pub fn remove(&self, name: &str) -> bool {
        // 从锁中毒中恢复 —— 数据依然有效
        let mut guard = self.templates.write().unwrap_or_else(|e| e.into_inner());
        guard.remove(name).is_some()
    }

    /// 检查给定名称的模板是否已注册。
    pub fn contains(&self, name: &str) -> bool {
        // 从锁中毒中恢复 —— 数据依然有效
        let guard = self.templates.read().unwrap_or_else(|e| e.into_inner());
        guard.contains_key(name)
    }

    /// 列出所有已注册模板的名称。
    pub fn template_names(&self) -> Vec<String> {
        // 从锁中毒中恢复 —— 数据依然有效
        let guard = self.templates.read().unwrap_or_else(|e| e.into_inner());
        guard.keys().cloned().collect()
    }

    /**
    使用提供的变量替换渲染一个具名模板。

    变量以 `(name, value)` 对组成的切片形式提供。缺失的变量要么使用模板中
    通过 `{{name:default}}` 指定的默认值替换，要么保留为空。

    # 错误

    若模板名未注册，返回 `ReactError::Other`。
    */
    pub fn render(&self, name: &str, variables: &[(&str, &str)]) -> Result<String> {
        // 从锁中毒中恢复 —— 数据依然有效
        let guard = self.templates.read().unwrap_or_else(|e| e.into_inner());
        let template = guard
            .get(name)
            .ok_or_else(|| ReactError::Other(format!("Prompt template '{}' not found", name)))?;

        let vars: HashMap<String, String> = variables
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let elements = parse_template(template);
        Ok(render_elements(&elements, &vars))
    }

    /**
    直接渲染一个模板字符串（不通过名称查找）。

    适用于模板未在管理器中注册的一次性渲染场景。
    */
    pub fn render_template(&self, template: &str, variables: &[(&str, &str)]) -> String {
        let vars: HashMap<String, String> = variables
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let elements = parse_template(template);
        render_elements(&elements, &vars)
    }

    /**
    获取一个已注册模板的原始模板字符串。

    若模板名未注册，返回 `None`。
    */
    pub fn get_template(&self, name: &str) -> Option<String> {
        // 从锁中毒中恢复 —— 数据依然有效
        let guard = self.templates.read().unwrap_or_else(|e| e.into_inner());
        guard.get(name).cloned()
    }

    /**
    渲染一个具名模板；当不需要任何替换时（即模板中不包含 `{{` 标记），
    直接回退返回原始模板字符串。

    这是针对静态文本模板的一种性能优化。
    */
    pub fn render_or_raw(&self, name: &str, variables: &[(&str, &str)]) -> Result<String> {
        // 从锁中毒中恢复 —— 数据依然有效
        let guard = self.templates.read().unwrap_or_else(|e| e.into_inner());
        let template = guard
            .get(name)
            .ok_or_else(|| ReactError::Other(format!("Prompt template '{}' not found", name)))?;

        if !template.contains("{{") {
            return Ok(template.clone());
        }

        let vars: HashMap<String, String> = variables
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let elements = parse_template(template);
        Ok(render_elements(&elements, &vars))
    }
}

impl Default for PromptTemplateManager {
    /// 等价于 `PromptTemplateManager::new()`。
    fn default() -> Self {
        Self::new()
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_simple_variable_substitution() {
        let manager = PromptTemplateManager::new();
        manager.register("greeting", "Hello, {{name}}!");
        let result = manager.render("greeting", &[("name", "Alice")]).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_multiple_variables() {
        let manager = PromptTemplateManager::new();
        manager.register("intro", "I am {{name}}, a {{role}} at {{company}}.");
        let result = manager
            .render(
                "intro",
                &[("name", "Bob"), ("role", "engineer"), ("company", "Acme")],
            )
            .unwrap();
        assert_eq!(result, "I am Bob, a engineer at Acme.");
    }

    #[test]
    fn test_default_value() {
        let manager = PromptTemplateManager::new();
        manager.register("fallback", "Hello, {{name:Guest}}!");
        // 不提供 'name' 时，默认使用 "Guest"
        let result = manager.render("fallback", &[]).unwrap();
        assert_eq!(result, "Hello, Guest!");
        // 提供 'name' 时，使用提供的值
        let result = manager.render("fallback", &[("name", "Alice")]).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_missing_variable_empty() {
        let manager = PromptTemplateManager::new();
        manager.register("missing", "Hello, {{name}}!");
        let result = manager.render("missing", &[]).unwrap();
        assert_eq!(result, "Hello, !");
    }

    #[test]
    fn test_conditional_block_present() {
        let manager = PromptTemplateManager::new();
        manager.register(
            "cond",
            "Base text. {{#if detail}}Details: {{detail}}.{{#endif}} End.",
        );
        let result = manager
            .render("cond", &[("detail", "important info")])
            .unwrap();
        assert_eq!(result, "Base text. Details: important info. End.");
    }

    #[test]
    fn test_conditional_block_absent() {
        let manager = PromptTemplateManager::new();
        manager.register(
            "cond",
            "Base text. {{#if detail}}Details: {{detail}}.{{#endif}} End.",
        );
        let result = manager.render("cond", &[]).unwrap();
        assert_eq!(result, "Base text.  End.");
    }

    #[test]
    fn test_conditional_with_else() {
        let manager = PromptTemplateManager::new();
        manager.register(
            "cond_else",
            "{{#if premium}}Premium features enabled.{{#else}}Standard features only.{{#endif}}",
        );
        // 存在 premium
        let result = manager.render("cond_else", &[("premium", "true")]).unwrap();
        assert_eq!(result, "Premium features enabled.");
        // 不存在 premium
        let result = manager.render("cond_else", &[]).unwrap();
        assert_eq!(result, "Standard features only.");
    }

    #[test]
    fn test_conditional_empty_value_acts_as_absent() {
        let manager = PromptTemplateManager::new();
        manager.register("cond_empty", "{{#if var}}Present{{#else}}Absent{{#endif}}");
        // 空字符串值会触发 else 分支
        let result = manager.render("cond_empty", &[("var", "")]).unwrap();
        assert_eq!(result, "Absent");
    }

    #[test]
    fn test_nested_conditional() {
        let manager = PromptTemplateManager::new();
        manager.register(
            "nested",
            "{{#if a}}A is present. {{#if b}}B too.{{#else}}B missing.{{#endif}}{{#else}}A missing.{{#endif}}",
        );
        // 两者都存在
        let result = manager
            .render("nested", &[("a", "yes"), ("b", "yes")])
            .unwrap();
        assert_eq!(result, "A is present. B too.");
        // 仅 A 存在
        let result = manager.render("nested", &[("a", "yes")]).unwrap();
        assert_eq!(result, "A is present. B missing.");
        // A 不存在
        let result = manager.render("nested", &[]).unwrap();
        assert_eq!(result, "A missing.");
    }

    #[test]
    fn test_template_not_found() {
        let manager = PromptTemplateManager::new();
        let result = manager.render("nonexistent", &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_register_overwrite() {
        let manager = PromptTemplateManager::new();
        manager.register("key", "Version 1");
        manager.register("key", "Version 2");
        let result = manager.render("key", &[]).unwrap();
        assert_eq!(result, "Version 2");
    }

    #[test]
    fn test_remove_template() {
        let manager = PromptTemplateManager::new();
        manager.register("key", "Value");
        assert!(manager.remove("key"));
        assert!(!manager.contains("key"));
        assert!(!manager.remove("key")); // 已经被移除过
    }

    #[test]
    fn test_template_names() {
        let manager = PromptTemplateManager::new();
        manager.register("a", "A");
        manager.register("b", "B");
        manager.register("c", "C");
        let names = manager.template_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"c".to_string()));
    }

    #[test]
    fn test_render_template_direct() {
        let manager = PromptTemplateManager::new();
        let result = manager.render_template("Hello {{who}}!", &[("who", "World")]);
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_render_or_raw_static_template() {
        let manager = PromptTemplateManager::new();
        manager.register("static", "No variables here.");
        let result = manager.render_or_raw("static", &[]).unwrap();
        assert_eq!(result, "No variables here.");
    }

    #[test]
    fn test_render_or_raw_dynamic_template() {
        let manager = PromptTemplateManager::new();
        manager.register("dynamic", "Hello, {{name}}!");
        let result = manager
            .render_or_raw("dynamic", &[("name", "Alice")])
            .unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_arc_shared_manager() {
        let manager = Arc::new(PromptTemplateManager::new());
        manager.register("shared", "Hello, {{name}}!");

        // 可以从多个引用中使用
        let m1 = Arc::clone(&manager);
        let m2 = Arc::clone(&manager);
        let r1 = m1.render("shared", &[("name", "A")]).unwrap();
        let r2 = m2.render("shared", &[("name", "B")]).unwrap();
        assert_eq!(r1, "Hello, A!");
        assert_eq!(r2, "Hello, B!");
    }

    #[test]
    fn test_get_template() {
        let manager = PromptTemplateManager::new();
        manager.register("key", "Hello {{name}}");
        assert_eq!(
            manager.get_template("key"),
            Some("Hello {{name}}".to_string())
        );
        assert_eq!(manager.get_template("missing"), None);
    }

    #[test]
    fn test_whitespace_in_tags() {
        let manager = PromptTemplateManager::new();
        manager.register("ws", "Hello, {{ name }}!");
        let result = manager.render("ws", &[("name", "Alice")]).unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_literal_with_no_markers() {
        let manager = PromptTemplateManager::new();
        manager.register("plain", "Just plain text.");
        let result = manager.render("plain", &[]).unwrap();
        assert_eq!(result, "Just plain text.");
    }
}
