/*!
EchoAgent 插件所使用的扁平化 `plugin.json` 清单文件。

Portable Agent Plugins 的元数据与 EchoAgent 的本地配置共享
同一份根文档。各组件的位置由软件包布局固定，因此不会在清单中重复声明。
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Agent Plugins 1.0 清单模式的规范标识符。
pub const AGENT_PLUGIN_SCHEMA_V1: &str =
    "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";

/// 根 `plugin.json` 文档。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginManifest {
    /// 选择 Agent Plugins 的校验与解释契约。
    #[serde(rename = "$schema")]
    pub schema: String,

    /// 可移植的插件标识符。
    pub name: String,

    /// 可选的 UI 显示名称。缺省时回退为 `name`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,

    /// 插件版本。Agent Plugins 推荐使用语义化版本（SemVer），但不作强制要求。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// 简短的人类可读描述。
    #[serde(default)]
    pub description: String,

    /// 可选的作者元数据。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<PluginAuthor>,

    /// 文档或主页地址。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,

    /// 源代码仓库地址。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,

    /// 许可证字符串；标准推荐使用 SPDX 标识符。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// 用于搜索与发现的关键词。
    #[serde(default)]
    pub keywords: Vec<String>,

    /// 当配置完整时是否默认启动为启用状态。
    #[serde(default = "default_true")]
    pub default_enabled: bool,

    /// 由宿主应用管理的、用户可配置的值。
    #[serde(default)]
    pub config: HashMap<String, PluginUserConfigEntry>,

    /// 已安装插件之间可选的依赖顺序。
    #[serde(default)]
    pub dependencies: Vec<PluginDependency>,

    /**
    Agent Plugins 1.0 将未知的顶层字段视为非致命错误。保留它们，
    以便调用方可以在不丢失原始数据的情况下上报诊断信息。
    */
    #[serde(flatten)]
    unknown_fields: HashMap<String, serde_json::Value>,
}

/// 可移植的作者元数据。在 Agent Plugins 1.0 中所有成员都是可选的。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginAuthor {
    /// 作者姓名。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 作者邮箱。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// 作者主页或联系地址。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// 为 `default_enabled` 提供 serde 默认值：`true`。
fn default_true() -> bool {
    true
}

/// EchoAgent 用户可配置值的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginUserConfigType {
    /// 字符串类型（当 `multiple` 为 true 时表示字符串数组）。
    String,
    /// 数值类型，可配合 `min`/`max` 使用。
    Number,
    /// 布尔类型。
    Boolean,
    /// 目录路径类型；校验时要求路径存在且为目录。
    Directory,
    /// 文件路径类型；校验时要求路径存在且为文件。
    File,
}

/// 一个由宿主应用管理的配置值定义。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginUserConfigEntry {
    /// 该配置项的值类型。
    #[serde(rename = "type")]
    pub value_type: PluginUserConfigType,
    /// 展示给用户的标题。
    pub title: String,
    /// 该配置项的描述文本。
    #[serde(default)]
    pub description: String,
    /// 是否为敏感值（如密钥），供宿主应用决定是否遮蔽显示。
    #[serde(default)]
    pub sensitive: bool,
    /// 该值是否为必填项。
    #[serde(default)]
    pub required: bool,
    /// 默认值。
    #[serde(default)]
    pub default: Option<serde_json::Value>,
    /// 仅对 `String` 类型有效：是否允许多个值（字符串数组）。
    #[serde(default)]
    pub multiple: bool,
    /// 仅对 `Number` 类型有效：允许的最小值。
    #[serde(default)]
    pub min: Option<f64>,
    /// 仅对 `Number` 类型有效：允许的最大值。
    #[serde(default)]
    pub max: Option<f64>,
}

/// 对另一个已安装插件的可选依赖声明。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PluginDependency {
    /// 仅指定插件名，不限定版本。
    Simple(String),
    /// 指定插件名及版本约束（SemVer 版本要求表达式）。
    Versioned {
        /// 依赖的插件名称。
        name: String,
        /// 版本约束表达式。
        version: String,
    },
}

impl PluginDependency {
    /// 返回该依赖对应的插件名称。
    pub fn name(&self) -> &str {
        match self {
            Self::Simple(name) => name,
            Self::Versioned { name, .. } => name,
        }
    }

    /// 返回该依赖的版本约束表达式（若有指定版本）。
    pub fn version_constraint(&self) -> Option<&str> {
        match self {
            Self::Simple(_) => None,
            Self::Versioned { version, .. } => Some(version),
        }
    }

    /**
    判断给定的已安装版本是否满足该依赖的版本约束。

    若依赖未指定版本约束，则总是返回 `Ok(true)`。若指定了约束但未提供
    已安装版本，或版本字符串无法解析，则返回错误。
    */
    pub fn satisfies(&self, installed_version: Option<&str>) -> Result<bool, String> {
        let Some(constraint) = self.version_constraint() else {
            return Ok(true);
        };
        let version = installed_version
            .ok_or_else(|| format!("dependency '{}' has no installed version", self.name()))?;
        let requirement = semver::VersionReq::parse(constraint)
            .map_err(|error| format!("invalid version constraint '{constraint}': {error}"))?;
        let parsed = semver::Version::parse(version)
            .map_err(|error| format!("invalid installed version '{version}': {error}"))?;
        Ok(requirement.matches(&parsed))
    }
}

/// 一条致命的清单校验错误。
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// 出错的字段路径。
    pub field: String,
    /// 错误说明信息。
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    /// 以 `字段: 说明` 的格式渲染错误信息。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.field, self.message)
    }
}

impl PluginManifest {
    /// 解析一份根 Agent Plugins 清单文档。
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|error| format!("Failed to parse plugin.json: {error}"))
    }

    /// 从文件加载一份根 Agent Plugins 清单文档。
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|error| format!("Failed to read manifest file {}: {error}", path.display()))?;
        Self::from_json(&content)
    }

    /// 返回符合规范的客户端应当上报并忽略的非致命未知字段列表。
    pub fn unknown_top_level_fields(&self) -> Vec<&str> {
        let mut fields = self
            .unknown_fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        fields.sort_unstable();
        fields
    }

    /// 校验可移植元数据，以及宿主应用的根级配置字段。
    pub fn validate(&self) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        if self.schema != AGENT_PLUGIN_SCHEMA_V1 {
            errors.push(ValidationError {
                field: "$schema".to_string(),
                message: format!(
                    "Unsupported Agent Plugins schema '{}'; expected '{AGENT_PLUGIN_SCHEMA_V1}'",
                    self.schema
                ),
            });
        }
        if !is_agent_plugin_name(&self.name) {
            errors.push(ValidationError {
                field: "name".to_string(),
                message: "Plugin name must be 1-64 lowercase ASCII letters, digits, hyphens, or periods; begin and end alphanumeric; and contain neither '--' nor '..'"
                    .to_string(),
            });
        }

        self.validate_plugin_fields(&mut errors);
        errors
    }

    /// 该清单是否通过全部校验（无任何校验错误）。
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// 返回用于展示的版本标签；未指定版本时返回 `"unspecified"`。
    pub fn version_label(&self) -> &str {
        self.version.as_deref().unwrap_or("unspecified")
    }

    /// 返回用于展示的名称；未指定 `display_name` 时回退为 `name`。
    pub fn display_name(&self) -> &str {
        self.display_name.as_deref().unwrap_or(&self.name)
    }

    /**
    将调用方提供的配置值与清单中声明的默认值合并，并进行校验。

    提供了未声明键的条目会被记为错误；成功时返回合并后的完整配置集合。
    */
    pub fn resolve_user_config(
        &self,
        provided: &HashMap<String, serde_json::Value>,
    ) -> Result<HashMap<String, serde_json::Value>, Vec<ValidationError>> {
        let mut resolved = self.user_config_defaults();
        let mut errors = provided
            .keys()
            .filter(|key| !self.config.contains_key(*key))
            .map(|key| ValidationError {
                field: format!("config.{key}"),
                message: "Unknown plugin configuration key".to_string(),
            })
            .collect::<Vec<_>>();
        resolved.extend(
            provided
                .iter()
                .filter(|(key, value)| self.config.contains_key(*key) && !value.is_null())
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        errors.extend(self.validate_user_config(&resolved));
        if errors.is_empty() {
            Ok(resolved)
        } else {
            Err(errors)
        }
    }

    /// 收集清单中每个配置项声明的默认值。
    pub fn user_config_defaults(&self) -> HashMap<String, serde_json::Value> {
        self.config
            .iter()
            .filter_map(|(key, entry)| {
                entry
                    .default
                    .as_ref()
                    .map(|value| (key.clone(), value.clone()))
            })
            .collect()
    }

    /// 校验给定的用户配置值集合是否符合清单中各配置项的定义。
    pub fn validate_user_config(
        &self,
        values: &HashMap<String, serde_json::Value>,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        for key in values.keys() {
            if !self.config.contains_key(key) {
                errors.push(ValidationError {
                    field: format!("config.{key}"),
                    message: "Unknown plugin configuration key".to_string(),
                });
            }
        }
        for (key, entry) in &self.config {
            let value = values.get(key);
            if value.is_none_or(serde_json::Value::is_null) {
                if entry.required {
                    errors.push(ValidationError {
                        field: format!("config.{key}"),
                        message: "Required plugin configuration value is missing".to_string(),
                    });
                }
                continue;
            }
            if let Some(value) = value
                && let Some(message) = validate_config_value(entry, value)
            {
                errors.push(ValidationError {
                    field: format!("config.{key}"),
                    message,
                });
            }
        }
        errors
    }

    /**
    校验插件自身声明的字段：配置键命名、`multiple`/`min`/`max` 的适用性、
    默认值的合法性，以及依赖名称的合法性。
    */
    fn validate_plugin_fields(&self, errors: &mut Vec<ValidationError>) {
        for (key, entry) in &self.config {
            if !is_valid_identifier(key) {
                errors.push(ValidationError {
                    field: format!("config.{key}"),
                    message: "Config keys must start with a letter or underscore and contain only ASCII letters, digits, or underscores"
                        .to_string(),
                });
            }
            if entry.multiple && entry.value_type != PluginUserConfigType::String {
                errors.push(ValidationError {
                    field: format!("config.{key}.multiple"),
                    message: "'multiple' is only valid for type 'string'".to_string(),
                });
            }
            if (entry.min.is_some() || entry.max.is_some())
                && entry.value_type != PluginUserConfigType::Number
            {
                errors.push(ValidationError {
                    field: format!("config.{key}"),
                    message: "'min' and 'max' are only valid for type 'number'".to_string(),
                });
            }
            if let (Some(minimum), Some(maximum)) = (entry.min, entry.max)
                && minimum > maximum
            {
                errors.push(ValidationError {
                    field: format!("config.{key}"),
                    message: "'min' must not be greater than 'max'".to_string(),
                });
            }
            if let Some(default) = entry.default.as_ref()
                && let Some(message) = validate_config_value(entry, default)
            {
                errors.push(ValidationError {
                    field: format!("config.{key}.default"),
                    message,
                });
            }
        }
        for dependency in &self.dependencies {
            if !is_agent_plugin_name(dependency.name()) {
                errors.push(ValidationError {
                    field: "dependencies".to_string(),
                    message: format!(
                        "Dependency name '{}' is not a valid Agent Plugin name",
                        dependency.name()
                    ),
                });
            }
        }
    }
}

/**
校验单个配置值是否符合其配置项定义（类型、必填性、取值范围、路径存在性等）。
校验通过返回 `None`，否则返回描述错误的字符串。
*/
fn validate_config_value(
    entry: &PluginUserConfigEntry,
    value: &serde_json::Value,
) -> Option<String> {
    match entry.value_type {
        PluginUserConfigType::String if entry.multiple => {
            let Some(values) = value.as_array() else {
                return Some("Expected an array of strings".to_string());
            };
            if values.iter().any(|item| !item.is_string()) {
                return Some("Expected an array containing only strings".to_string());
            }
            if entry.required && values.is_empty() {
                return Some("Required value must not be empty".to_string());
            }
        }
        PluginUserConfigType::String => {
            let Some(text) = value.as_str() else {
                return Some("Expected a string".to_string());
            };
            if entry.required && text.is_empty() {
                return Some("Required value must not be empty".to_string());
            }
        }
        PluginUserConfigType::Number => {
            let Some(number) = value.as_f64() else {
                return Some("Expected a number".to_string());
            };
            if let Some(minimum) = entry.min
                && number < minimum
            {
                return Some(format!("Value must be at least {minimum}"));
            }
            if let Some(maximum) = entry.max
                && number > maximum
            {
                return Some(format!("Value must be at most {maximum}"));
            }
        }
        PluginUserConfigType::Boolean => {
            if !value.is_boolean() {
                return Some("Expected a boolean".to_string());
            }
        }
        PluginUserConfigType::Directory | PluginUserConfigType::File => {
            let Some(path) = value.as_str() else {
                return Some("Expected a filesystem path string".to_string());
            };
            let exists = match entry.value_type {
                PluginUserConfigType::Directory => Path::new(path).is_dir(),
                PluginUserConfigType::File => Path::new(path).is_file(),
                _ => false,
            };
            if !exists {
                return Some(match entry.value_type {
                    PluginUserConfigType::Directory => format!("Directory does not exist: {path}"),
                    PluginUserConfigType::File => format!("File does not exist: {path}"),
                    _ => String::new(),
                });
            }
        }
    }
    None
}

/**
校验插件名称是否符合 Agent Plugins 规范：1–64 个小写 ASCII 字母、
数字、连字符或句点，首尾必须为字母数字，且不能包含 `--` 或 `..`。
*/
fn is_agent_plugin_name(name: &str) -> bool {
    let len = name.chars().count();
    (1..=64).contains(&len)
        && name
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphanumeric())
        && name
            .chars()
            .last()
            .is_some_and(|character| character.is_ascii_alphanumeric())
        && name.chars().all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '-'
                || character == '.'
        })
        && !name.contains("--")
        && !name.contains("..")
}

/**
校验字符串是否为合法标识符：以字母或下划线开头，
且只包含 ASCII 字母、数字或下划线。
*/
fn is_valid_identifier(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个带有默认字段、并合并了 `extra` 中额外字段的测试用清单。
    fn manifest(extra: serde_json::Value) -> Result<PluginManifest, String> {
        let mut document = serde_json::json!({
            "$schema": AGENT_PLUGIN_SCHEMA_V1,
            "name": "example.plugin",
            "version": "1.2.0"
        });
        let fields = extra
            .as_object()
            .ok_or_else(|| "test manifest fields must be an object".to_string())?;
        let object = document
            .as_object_mut()
            .ok_or_else(|| "test manifest must be an object".to_string())?;
        object.extend(fields.clone());
        PluginManifest::from_json(
            &serde_json::to_string(&document).map_err(|error| error.to_string())?,
        )
    }

    /// 验证可解析扁平化插件清单并正确返回显示名与版本。
    #[test]
    fn parses_flat_plugin_manifest() -> Result<(), String> {
        let parsed = manifest(serde_json::json!({
            "displayName": "Example Plugin"
        }))?;
        assert!(parsed.is_valid());
        assert_eq!(parsed.display_name(), "Example Plugin");
        assert_eq!(parsed.version_label(), "1.2.0");
        Ok(())
    }

    /// 验证缺失或不受支持的 `$schema` 会被拒绝。
    #[test]
    fn rejects_missing_or_unsupported_schema() -> Result<(), String> {
        let parsed = PluginManifest::from_json(r#"{"name":"example"}"#)
            .err()
            .unwrap_or_default();
        assert!(parsed.contains("$schema"));

        let parsed = PluginManifest::from_json(
            r#"{"$schema":"https://example.invalid/plugin.json","name":"example"}"#,
        )?;
        assert!(
            parsed
                .validate()
                .iter()
                .any(|error| error.field == "$schema")
        );
        Ok(())
    }

    /// 验证插件名支持句点，但拒绝连续分隔符与大写字母。
    #[test]
    fn supports_standard_name_periods_and_rejects_double_separators() {
        assert!(is_agent_plugin_name("acme.tools"));
        assert!(!is_agent_plugin_name("acme..tools"));
        assert!(!is_agent_plugin_name("acme--tools"));
        assert!(!is_agent_plugin_name("Uppercase"));
    }

    /// 验证未知顶层字段不是致命错误并可被上报。
    #[test]
    fn unknown_top_level_fields_are_non_fatal() -> Result<(), String> {
        let parsed = PluginManifest::from_json(&format!(
            r#"{{"$schema":"{AGENT_PLUGIN_SCHEMA_V1}","name":"example","future":true}}"#
        ))?;
        assert!(parsed.is_valid());
        assert_eq!(parsed.unknown_top_level_fields(), vec!["future"]);
        Ok(())
    }

    /// 验证根级配置项的 min/max 约束会被校验。
    #[test]
    fn validates_root_configuration() -> Result<(), String> {
        let parsed = manifest(serde_json::json!({
            "config": {
                "retries": {
                    "type": "number",
                    "title": "Retries",
                    "min": 5,
                    "max": 2
                }
            }
        }))?;
        let errors = parsed.validate();
        assert!(errors.iter().any(|error| error.field.contains("retries")));
        Ok(())
    }

    /// 验证依赖版本约束在缺少已安装版本时返回错误。
    #[test]
    fn dependency_constraints_handle_missing_versions() {
        let dependency = PluginDependency::Versioned {
            name: "base.tools".to_string(),
            version: ">=1.0.0".to_string(),
        };
        assert!(dependency.satisfies(Some("1.2.0")).unwrap_or(false));
        assert!(dependency.satisfies(None).is_err());
    }

    /// 验证用户配置的默认值合并与类型校验。
    #[test]
    fn resolves_and_validates_user_configuration() -> Result<(), String> {
        let parsed = manifest(serde_json::json!({
            "config": {
                "endpoint": {
                    "type": "string",
                    "title": "Endpoint",
                    "required": true,
                    "default": "https://example.com"
                }
            }
        }))?;
        let resolved = parsed
            .resolve_user_config(&HashMap::new())
            .map_err(|errors| {
                errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; ")
            })?;
        assert_eq!(
            resolved.get("endpoint").and_then(serde_json::Value::as_str),
            Some("https://example.com")
        );
        let invalid = HashMap::from([(
            "endpoint".to_string(),
            serde_json::json!(["https://example.com"]),
        )]);
        assert!(!parsed.validate_user_config(&invalid).is_empty());
        Ok(())
    }
}
