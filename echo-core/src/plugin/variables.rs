/*!
插件环境变量 —— 组件配置中的路径替换。

以下三个变量可在 skill 内容、agent 内容、hook 命令、monitor 命令
以及 MCP/LSP 服务器配置中进行替换：

| 变量 | 说明 |
|----------|-------------|
| `${ECHO_PLUGIN_ROOT}` | 插件安装目录的绝对路径 |
| `${ECHO_PLUGIN_DATA}` | 持久化数据目录，更新后仍保留 |
| `${ECHO_PROJECT_DIR}` | 项目根目录 |

操作系统环境变量（`${ENV_VAR}`）同样会被替换。

Agent Plugins 1.0 MCP 使用更窄的可移植契约：`PLUGIN_ROOT` 与 `PLUGIN_DATA`
由严格的 `mcp.json` 适配器提供，且仅在标准定义的 stdio 字段中展开。
*/

use std::collections::HashMap;
use std::path::PathBuf;

/// 用于路径替换的已解析插件变量。
#[derive(Debug, Clone)]
pub struct PluginVariables {
    /// 插件安装目录的绝对路径。
    pub plugin_root: PathBuf,
    /// 插件的持久化数据目录。
    pub plugin_data: PathBuf,
    /// 项目根目录。
    pub project_dir: PathBuf,
    /// 来自插件配置的额外用户自定义变量。
    pub user_config: HashMap<String, String>,
}

impl PluginVariables {
    /// 为插件创建一组新的变量。
    pub fn new(plugin_root: PathBuf, plugin_data: PathBuf, project_dir: PathBuf) -> Self {
        Self {
            plugin_root,
            plugin_data,
            project_dir,
            user_config: HashMap::new(),
        }
    }

    /// 设置用户配置值。
    pub fn with_user_config(mut self, config: HashMap<String, String>) -> Self {
        self.user_config = config;
        self
    }

    /// 从持久化的 JSON 值设置用户配置（非字符串值会转为 JSON 文本）。
    pub fn with_json_user_config(mut self, config: &HashMap<String, serde_json::Value>) -> Self {
        self.user_config = config
            .iter()
            .map(|(key, value)| {
                let text = match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                (key.clone(), text)
            })
            .collect();
        self
    }

    /// 覆盖持久化数据目录。
    pub fn with_plugin_data(mut self, plugin_data: PathBuf) -> Self {
        self.plugin_data = plugin_data;
        self
    }

    /**
    替换字符串中的所有变量。

    替换 `${ECHO_PLUGIN_ROOT}`、`${ECHO_PLUGIN_DATA}`、
    `${ECHO_PROJECT_DIR}`、`${user_config.KEY}` 以及 `${ENV_VAR}`。
    */
    pub fn substitute(&self, input: &str) -> String {
        let mut result = input.to_string();

        // 内置变量
        result = result.replace("${ECHO_PLUGIN_ROOT}", &self.plugin_root.to_string_lossy());
        result = result.replace("${ECHO_PLUGIN_DATA}", &self.plugin_data.to_string_lossy());
        result = result.replace("${ECHO_PROJECT_DIR}", &self.project_dir.to_string_lossy());

        // 用户配置变量：${user_config.KEY}
        for (key, value) in &self.user_config {
            let placeholder = format!("${{user_config.{key}}}");
            result = result.replace(&placeholder, value);
        }

        // 环境变量：${ENV_VAR}
        result = substitute_env_vars(&result);

        result
    }

    /**
    将插件内的相对路径解析为绝对路径。

    路径应以 `./` 开头，并相对于 `plugin_root` 解析。
    */
    pub fn resolve_path(&self, relative: &str) -> PathBuf {
        let stripped = relative.strip_prefix("./").unwrap_or(relative);
        self.plugin_root.join(stripped)
    }

    /// 确保数据目录存在。
    pub fn ensure_data_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.plugin_data)?;
        Ok(())
    }
}

/**
将 `${ENV_VAR}` 形式的占位符替换为环境变量值。

未知变量保持原样（不会被移除）。
*/
fn substitute_env_vars(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '$' && chars.peek() == Some(&'{') {
            chars.next(); // 消费 '{'
            let mut var_name = String::new();
            let mut closed = false;
            for ch in chars.by_ref() {
                if ch == '}' {
                    closed = true;
                    break;
                }
                var_name.push(ch);
            }
            if closed && !var_name.starts_with("ECHO_") && !var_name.starts_with("user_config.") {
                // 尝试作为环境变量解析
                match std::env::var(&var_name) {
                    Ok(val) => result.push_str(&val),
                    Err(_) => {
                        // 未找到时保持原样
                        result.push_str(&format!("${{{var_name}}}"));
                    }
                }
            } else if !closed {
                // 未闭合的 ${，将已消费的内容原样写回
                result.push('$');
                result.push('{');
                result.push_str(&var_name);
            } else {
                // 已处理过的变量（ECHO_ 或 user_config.）
                result.push_str(&format!("${{{var_name}}}"));
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_builtin_vars() {
        let vars = PluginVariables {
            plugin_root: PathBuf::from("/home/user/.echo-agent/plugins/my-plugin"),
            plugin_data: PathBuf::from("/home/user/.echo-agent/plugins/data/my-plugin"),
            project_dir: PathBuf::from("/home/user/my-project"),
            user_config: HashMap::new(),
        };

        let input = "run ${ECHO_PLUGIN_ROOT}/scripts/start.sh";
        let result = vars.substitute(input);
        assert!(result.contains("/home/user/.echo-agent/plugins/my-plugin/scripts/start.sh"));
        assert!(!result.contains("${ECHO_PLUGIN_ROOT}"));
    }

    #[test]
    fn test_substitute_user_config() {
        let mut config = HashMap::new();
        config.insert(
            "api_endpoint".to_string(),
            "http://localhost:9090".to_string(),
        );

        let vars = PluginVariables {
            plugin_root: PathBuf::from("/tmp/plugin"),
            plugin_data: PathBuf::from("/tmp/data"),
            project_dir: PathBuf::from("/tmp/project"),
            user_config: config,
        };

        let input = "connect to ${user_config.api_endpoint}";
        let result = vars.substitute(input);
        assert_eq!(result, "connect to http://localhost:9090");
    }

    #[test]
    fn test_resolve_path() {
        let vars = PluginVariables::new(
            PathBuf::from("/home/user/.echo-agent/plugins/test"),
            PathBuf::from("/home/user/.echo-agent/plugins/data/test"),
            PathBuf::from("/home/user/project"),
        );

        assert_eq!(
            vars.resolve_path("./skills/my-skill"),
            PathBuf::from("/home/user/.echo-agent/plugins/test/skills/my-skill")
        );
    }

    #[test]
    fn test_substitute_env_vars() {
        unsafe { std::env::set_var("TEST_ECHO_VAR", "hello") };
        let result = substitute_env_vars("value is ${TEST_ECHO_VAR}");
        assert_eq!(result, "value is hello");
        unsafe { std::env::remove_var("TEST_ECHO_VAR") };
    }

    #[test]
    fn test_unknown_env_var_preserved() {
        let result = substitute_env_vars("${UNKNOWN_VAR_XYZ_12345}");
        assert_eq!(result, "${UNKNOWN_VAR_XYZ_12345}");
    }
}
