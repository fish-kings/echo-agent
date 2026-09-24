//! mcp.json 配置文件加载器
//!
//! 支持与 Claude Desktop / Cursor / VS Code 等主流 Agent 工具兼容的
//! `mcp.json` 配置格式，可直接复用现有的 MCP 服务端配置。
//!
//! ## 文件格式
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "filesystem": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
//!       "env": {
//!         "OPTIONAL_VAR": "value"
//!       }
//!     },
//!     "github": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-github"],
//!       "env": {
//!         "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"
//!       }
//!     },
//!     "remote-api": {
//!       "url": "http://localhost:8080/mcp",
//!       "headers": {
//!         "Authorization": "Bearer token"
//!       }
//!     },
//!     "legacy-sse": {
//!       "url": "http://localhost:3000",
//!       "transport": "sse"
//!     },
//!     "disabled-server": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-postgres", "postgres://localhost/db"],
//!       "disabled": true
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::server_config::{McpServerConfig, TransportConfig};
use crate::redaction::{collection_is_redacted, url as redact_url};
use echo_core::error::{McpError, ReactError, Result};

/// Agent Plugins 1.0 MCP schema 的规范标识符
pub const AGENT_PLUGIN_MCP_SCHEMA_V1: &str =
    "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";

// ── 配置文件结构 ──────────────────────────────────────────────────────────────

/// mcp.json 文件的顶层结构
///
/// 与 Claude Desktop / Cursor 的 `mcp.json` 格式完全兼容。
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct McpConfigFile {
    /** 可选的 schema 标识符。普通的 EchoAgent MCP 文件不需要它；
    Agent Plugin 包会使用 [`McpConfigFile::parse_agent_plugin`]。
    */
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,

    /// 服务端配置映射（key 为服务端名称，在同一 Agent 中唯一）
    #[serde(rename = "mcpServers", default)]
    pub mcp_servers: HashMap<String, McpServerEntry>,
}

impl fmt::Debug for McpConfigFile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpConfigFile")
            .field("schema", &self.schema)
            .field("mcp_servers", &self.mcp_servers)
            .finish()
    }
}

/// 单个 MCP 服务端的配置项
///
/// 支持两种模式：
/// - **stdio**：提供 `command`，可选 `args` 和 `env`
/// - **HTTP**：提供 `url`，可选 `headers` 和 `transport`
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
pub struct McpServerEntry {
    /// Agent Plugins 1.0 使用的显式传输类型区分字段
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub server_type: Option<String>,

    // ── stdio 模式 ────────────────────────────────────────────────────────────
    /// 启动服务端的命令（如 `"npx"`、`"uvx"`、`"python"`）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    /// 命令参数列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,

    /// 额外注入子进程的环境变量（不影响当前进程环境）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,

    /// stdio 进程的可选工作目录
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,

    // ── HTTP 模式 ─────────────────────────────────────────────────────────────
    /// HTTP 服务端 URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// HTTP 请求头（如 `Authorization: Bearer <token>`）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,

    /// HTTP 传输类型（仅在提供 `url` 时有效）
    ///
    /// - `"sse"`：旧版 HTTP+SSE（适用于旧版 MCP SDK），
    ///   在 `{url}/sse` 建立 SSE 连接，向动态获取的端点 POST 请求
    /// - 默认（不指定或其他值）：MCP Streamable HTTP，直接 POST 到端点 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,

    // ── 通用选项 ──────────────────────────────────────────────────────────────
    /// 设为 `true` 时跳过该服务端（默认为 `false`）
    #[serde(default)]
    pub disabled: bool,
}

impl fmt::Debug for McpServerEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpServerEntry")
            .field("server_type", &self.server_type)
            .field("command", &self.command)
            .field("args", &collection_is_redacted(self.args.is_empty()))
            .field("env", &collection_is_redacted(self.env.is_empty()))
            .field("cwd", &self.cwd)
            .field("url", &self.url.as_deref().map(redact_url))
            .field("headers", &collection_is_redacted(self.headers.is_empty()))
            .field("transport", &self.transport)
            .field("disabled", &self.disabled)
            .finish()
    }
}

impl McpServerEntry {
    /// 将此配置项转换为 [`McpServerConfig`]
    pub fn to_server_config(&self, name: &str) -> Result<McpServerConfig> {
        // 检查是否被禁用
        if self.disabled {
            return Err(ReactError::Mcp(Box::new(McpError::ConnectionFailed(
                format!("服务端 '{}' 已禁用（disabled: true）", name),
            ))));
        }

        if let Some(command) = &self.command {
            validate_stdio_command(command)?;

            let env: Vec<(String, String)> = self
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(McpServerConfig {
                name: name.to_string(),
                transport: TransportConfig::Stdio {
                    command: command.clone(),
                    args: self.args.clone(),
                    env,
                    cwd: self.cwd.as_ref().map(PathBuf::from),
                },
            })
        } else if let Some(url) = &self.url {
            let transport = match self.server_type.as_deref().or(self.transport.as_deref()) {
                Some("sse") => TransportConfig::Sse {
                    base_url: url.clone(),
                    headers: self.headers.clone(),
                },
                _ => TransportConfig::Http {
                    base_url: url.clone(),
                    headers: self.headers.clone(),
                },
            };
            Ok(McpServerConfig {
                name: name.to_string(),
                transport,
            })
        } else {
            Err(ReactError::Mcp(Box::new(McpError::ConnectionFailed(
                format!(
                    "服务端 '{}' 配置无效：stdio 模式需提供 'command'，HTTP 模式需提供 'url'",
                    name
                ),
            ))))
        }
    }
}

impl McpConfigFile {
    /// 从 JSON 字符串解析配置
    pub fn parse(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| {
            ReactError::Mcp(Box::new(McpError::ProtocolError(format!(
                "mcp.json 格式解析失败: {}",
                e
            ))))
        })
    }

    /** 解析封闭的 Agent Plugins 1.0 `mcp.json` 格式。

    顶层错误会为整个插件包禁用 MCP；无效的服务端条目会被跳过并记录到
    `diagnostics` 中，从而保留该标准要求的“按条目隔离失败”语义。
    */
    pub fn parse_agent_plugin(
        s: &str,
        plugin_root: &Path,
        plugin_data: &Path,
    ) -> Result<AgentPluginMcpLoad> {
        let raw: AgentPluginMcpDocument = serde_json::from_str(s).map_err(|error| {
            protocol_error(format!("Agent Plugin mcp.json is invalid: {error}"))
        })?;
        if raw.schema != AGENT_PLUGIN_MCP_SCHEMA_V1 {
            return Err(protocol_error(format!(
                "Unsupported Agent Plugins MCP schema '{}'; expected '{AGENT_PLUGIN_MCP_SCHEMA_V1}'",
                raw.schema
            )));
        }

        let plugin_root = canonical_directory(plugin_root, "PLUGIN_ROOT")?;
        let plugin_data = canonical_directory(plugin_data, "PLUGIN_DATA")?;
        let mut diagnostics = Vec::new();
        let mut mcp_servers = HashMap::new();
        let mut entries = raw.mcp_servers.into_iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));

        for (name, value) in entries {
            match parse_agent_plugin_server(&name, value, &plugin_root, &plugin_data) {
                Ok(entry) => {
                    mcp_servers.insert(name, entry);
                }
                Err(error) => diagnostics.push(format!("MCP server '{name}' skipped: {error}")),
            }
        }

        Ok(AgentPluginMcpLoad {
            config: Self {
                schema: Some(AGENT_PLUGIN_MCP_SCHEMA_V1.to_string()),
                mcp_servers,
            },
            diagnostics,
        })
    }

    /// 从文件路径加载配置
    ///
    /// # 示例
    /// ```rust,no_run
    /// use echo_integration::mcp::config_loader::McpConfigFile;
    ///
    /// let config = McpConfigFile::from_file("mcp.json")?;
    /// println!("共 {} 个服务端", config.mcp_servers.len());
    /// # Ok::<(), echo_core::error::ReactError>(())
    /// ```
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            ReactError::Mcp(Box::new(McpError::ConnectionFailed(format!(
                "读取配置文件失败 ({}): {}",
                path.display(),
                e
            ))))
        })?;
        Self::parse(&content)
    }

    /// 将所有**启用**的服务端转换为 [`McpServerConfig`] 列表
    pub fn to_server_configs(&self) -> Result<Vec<McpServerConfig>> {
        let mut configs = Vec::new();
        for (name, entry) in &self.mcp_servers {
            if entry.disabled {
                tracing::debug!("MCP: 跳过已禁用的服务端 '{}'", name);
                continue;
            }
            configs.push(entry.to_server_config(name)?);
        }
        Ok(configs)
    }

    /// 返回启用的服务端数量
    pub fn enabled_count(&self) -> usize {
        self.mcp_servers.values().filter(|e| !e.disabled).count()
    }
}

/// 解析 Agent Plugin 固定根目录下 `mcp.json` 组件的结果
#[derive(Debug, Clone)]
pub struct AgentPluginMcpLoad {
    /// 解析成功、已归一化的 MCP 配置
    pub config: McpConfigFile,
    /// 被跳过的无效服务端条目的诊断信息列表
    pub diagnostics: Vec<String>,
}

/// Agent Plugin `mcp.json` 文件的顶层原始结构（反序列化用）
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentPluginMcpDocument {
    /// schema 标识符，必须等于 [`AGENT_PLUGIN_MCP_SCHEMA_V1`]
    #[serde(rename = "$schema")]
    schema: String,
    /// 服务端配置映射（值先保留为原始 JSON，稍后按类型分发解析）
    #[serde(rename = "mcpServers")]
    mcp_servers: HashMap<String, serde_json::Value>,
}

/// Agent Plugin stdio 类型服务端的原始配置
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentPluginStdioServer {
    /// 传输类型标记，固定为 `stdio`
    #[serde(rename = "type")]
    server_type: AgentPluginStdioType,
    /// 启动命令
    command: String,
    /// 命令参数列表
    #[serde(default)]
    args: Vec<String>,
    /// 注入子进程的环境变量
    #[serde(default)]
    env: HashMap<String, String>,
    /// 子进程的工作目录
    #[serde(default)]
    cwd: Option<String>,
}

/// Agent Plugin stdio 服务端的传输类型标记
#[derive(Debug, Deserialize)]
enum AgentPluginStdioType {
    /// stdio 传输
    #[serde(rename = "stdio")]
    Stdio,
}

/// Agent Plugin 远程（HTTP）类型服务端的原始配置
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentPluginRemoteServer {
    /// 传输类型标记：`streamable-http` 或 `sse`
    #[serde(rename = "type")]
    server_type: AgentPluginRemoteType,
    /// 服务端 URL
    url: String,
    /// HTTP 请求头
    #[serde(default)]
    headers: HashMap<String, String>,
}

/// Agent Plugin 远程服务端的传输类型标记
#[derive(Debug, Clone, Copy, Deserialize)]
enum AgentPluginRemoteType {
    /// MCP Streamable HTTP 传输
    #[serde(rename = "streamable-http")]
    StreamableHttp,
    /// 旧版 HTTP+SSE 传输
    #[serde(rename = "sse")]
    Sse,
}

/// 根据 `type` 字段将一个原始 JSON 服务端条目解析并转换为 [`McpServerEntry`]，
/// 同时展开 `${PLUGIN_ROOT}` / `${PLUGIN_DATA}` 变量、校验命令与 URL 的安全性。
fn parse_agent_plugin_server(
    name: &str,
    value: serde_json::Value,
    plugin_root: &Path,
    plugin_data: &Path,
) -> std::result::Result<McpServerEntry, String> {
    let server_type = value
        .as_object()
        .and_then(|object| object.get("type"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "missing string field 'type'".to_string())?;
    match server_type {
        "stdio" => {
            let raw: AgentPluginStdioServer = serde_json::from_value(value)
                .map_err(|error| format!("invalid stdio configuration: {error}"))?;
            let _ = raw.server_type;
            validate_agent_plugin_command(&raw.command)?;
            if raw.env.contains_key("PLUGIN_ROOT") || raw.env.contains_key("PLUGIN_DATA") {
                return Err("env must not override PLUGIN_ROOT or PLUGIN_DATA".to_string());
            }
            let args = raw
                .args
                .into_iter()
                .map(|argument| expand_agent_plugin_variables(&argument, plugin_root, plugin_data))
                .collect();
            let mut env = raw
                .env
                .into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        expand_agent_plugin_variables(&value, plugin_root, plugin_data),
                    )
                })
                .collect::<HashMap<_, _>>();
            env.insert(
                "PLUGIN_ROOT".to_string(),
                plugin_root.to_string_lossy().into_owned(),
            );
            env.insert(
                "PLUGIN_DATA".to_string(),
                plugin_data.to_string_lossy().into_owned(),
            );
            let cwd = resolve_agent_plugin_cwd(raw.cwd.as_deref(), plugin_root, plugin_data)?;
            Ok(McpServerEntry {
                server_type: Some("stdio".to_string()),
                command: Some(raw.command),
                args,
                env,
                cwd: Some(cwd.to_string_lossy().into_owned()),
                url: None,
                headers: HashMap::new(),
                transport: None,
                disabled: false,
            })
        }
        "streamable-http" | "sse" => {
            let raw: AgentPluginRemoteServer = serde_json::from_value(value)
                .map_err(|error| format!("invalid remote configuration: {error}"))?;
            validate_agent_plugin_url(&raw.url)?;
            let transport = match raw.server_type {
                AgentPluginRemoteType::StreamableHttp => "streamable-http",
                AgentPluginRemoteType::Sse => "sse",
            };
            Ok(McpServerEntry {
                server_type: Some(transport.to_string()),
                command: None,
                args: Vec::new(),
                env: HashMap::new(),
                cwd: None,
                url: Some(raw.url),
                headers: raw.headers,
                transport: None,
                disabled: false,
            })
        }
        other => Err(format!("unsupported transport type '{other}'")),
    }
    .map_err(|error| format!("{name}: {error}"))
}

/// 将给定路径解析为绝对规范路径，并确认它是一个存在的目录；`label` 用于错误信息中标识该路径的用途
fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let canonical = std::fs::canonicalize(path).map_err(|error| {
        protocol_error(format!(
            "{label} directory '{}' is unavailable: {error}",
            path.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(protocol_error(format!(
            "{label} path '{}' is not a directory",
            canonical.display()
        )));
    }
    Ok(canonical)
}

/// 对输入字符串做单遍扫描，将其中的 `${PLUGIN_ROOT}` / `${PLUGIN_DATA}` 占位符
/// 替换为对应的实际路径（按 Unicode 字符边界安全处理，不做递归展开）
fn expand_agent_plugin_variables(input: &str, plugin_root: &Path, plugin_data: &Path) -> String {
    let source = input.chars().collect::<Vec<_>>();
    let root_marker = "${PLUGIN_ROOT}".chars().collect::<Vec<_>>();
    let data_marker = "${PLUGIN_DATA}".chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(input.len());
    let mut position = 0usize;
    while position < source.len() {
        if chars_match_at(&source, position, &root_marker) {
            output.push_str(&plugin_root.to_string_lossy());
            position = position.saturating_add(root_marker.len());
        } else if chars_match_at(&source, position, &data_marker) {
            output.push_str(&plugin_data.to_string_lossy());
            position = position.saturating_add(data_marker.len());
        } else if let Some(character) = source.get(position) {
            output.push(*character);
            position = position.saturating_add(1);
        } else {
            break;
        }
    }
    output
}

/// 判断 `pattern` 是否从 `source` 的 `position` 位置开始完全匹配
fn chars_match_at(source: &[char], position: usize, pattern: &[char]) -> bool {
    pattern.iter().enumerate().all(|(offset, expected)| {
        position
            .checked_add(offset)
            .and_then(|index| source.get(index))
            == Some(expected)
    })
}

/** 解析并校验 Agent Plugin 服务端声明的工作目录。

`cwd` 必须为空、`${PLUGIN_ROOT}`/`${PLUGIN_DATA}`（可带子路径）或以 `./` 开头的
相对路径；解析结果必须是插件根目录或数据目录内实际存在的目录，防止通过 `..`
或绝对路径逃逸出插件的沙箱边界。
*/
fn resolve_agent_plugin_cwd(
    cwd: Option<&str>,
    plugin_root: &Path,
    plugin_data: &Path,
) -> std::result::Result<PathBuf, String> {
    let (base, suffix) = match cwd {
        None => return Ok(plugin_root.to_path_buf()),
        Some("${PLUGIN_ROOT}") => (plugin_root, ""),
        Some("${PLUGIN_DATA}") => (plugin_data, ""),
        Some(value) if value.starts_with("${PLUGIN_ROOT}/") => {
            (plugin_root, value.trim_start_matches("${PLUGIN_ROOT}/"))
        }
        Some(value) if value.starts_with("${PLUGIN_DATA}/") => {
            (plugin_data, value.trim_start_matches("${PLUGIN_DATA}/"))
        }
        Some(value) if value.starts_with("./") => (plugin_root, value.trim_start_matches("./")),
        Some(value) => {
            return Err(format!(
                "cwd '{value}' must begin with './', '${{PLUGIN_ROOT}}', or '${{PLUGIN_DATA}}'"
            ));
        }
    };
    if Path::new(suffix).components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err("cwd must remain inside its selected plugin directory".to_string());
    }
    let candidate = base.join(suffix);
    let canonical = std::fs::canonicalize(&candidate)
        .map_err(|error| format!("cwd '{}' is unavailable: {error}", candidate.display()))?;
    if !canonical.starts_with(base) || !canonical.is_dir() {
        return Err(format!(
            "cwd '{}' must be an existing directory inside the plugin package or data directory",
            candidate.display()
        ));
    }
    Ok(canonical)
}

/// 校验 Agent Plugin 服务端命令的形式：不能为空、不能包含插件变量占位符，
/// 且必须是裸可执行文件名或以 `./` 开头且不逃出插件根目录的相对路径
fn validate_agent_plugin_command(command: &str) -> std::result::Result<(), String> {
    if command.is_empty() {
        return Err("command must not be empty".to_string());
    }
    if command.contains("${PLUGIN_ROOT}") || command.contains("${PLUGIN_DATA}") {
        return Err("command does not support plugin variable expansion".to_string());
    }
    if command.starts_with("./") {
        if Path::new(command).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err("plugin-relative command must remain inside the plugin root".to_string());
        }
        return Ok(());
    }
    if Path::new(command).is_absolute() || command.contains('/') || command.contains('\\') {
        return Err("command must be a bare executable name or begin with './'".to_string());
    }
    Ok(())
}

/// 校验 Agent Plugin 远程服务端 URL 的安全性：必须是合法的 http/https URL，
/// 不含用户信息或 URL fragment，且非回环地址必须使用 https
fn validate_agent_plugin_url(value: &str) -> std::result::Result<(), String> {
    let url = url::Url::parse(value).map_err(|error| format!("invalid URL: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("URL scheme must be http or https".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URL must not contain user information".to_string());
    }
    if url.fragment().is_some() {
        return Err("URL must not contain a fragment".to_string());
    }
    let loopback = match url.host() {
        Some(url::Host::Ipv4(address)) => address.is_loopback(),
        Some(url::Host::Ipv6(address)) => address.is_loopback(),
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        None => false,
    };
    if url.scheme() != "https" && !loopback {
        return Err("non-loopback MCP URLs must use https".to_string());
    }
    Ok(())
}

/// 将错误信息包装为 [`ReactError::Mcp`] 中的协议错误（[`McpError::ProtocolError`]）
fn protocol_error(message: String) -> ReactError {
    ReactError::Mcp(Box::new(McpError::ProtocolError(message)))
}

/** 校验通用 MCP 命令的基本形式。

命令会被直接启动，不经过 shell。因此该框架只校验存在一个可执行文件标记，
具体的信任策略交由嵌入该框架的本地应用自行决定。
*/
pub fn validate_stdio_command(command: &str) -> Result<()> {
    if command.trim().is_empty() {
        return Err(protocol_error("MCP stdio command is empty".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建一对临时的插件根目录与数据目录，供测试使用
    fn plugin_directories() -> std::result::Result<(tempfile::TempDir, PathBuf, PathBuf), String> {
        let temporary = tempfile::tempdir().map_err(|error| error.to_string())?;
        let root = temporary.path().join("plugin");
        let data = temporary.path().join("data");
        std::fs::create_dir_all(root.join("workspace")).map_err(|error| error.to_string())?;
        std::fs::create_dir_all(&data).map_err(|error| error.to_string())?;
        Ok((temporary, root, data))
    }

    /// 验证 Agent Plugin mcp.json 能被正确解析，且变量展开仅作用于 stdio 服务端的字段
    #[test]
    fn parses_agent_plugin_mcp_and_expands_only_stdio_values() -> std::result::Result<(), String> {
        let (_temporary, root, data) = plugin_directories()?;
        let document = serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_V1,
            "mcpServers": {
                "local": {
                    "type": "stdio",
                    "command": "node",
                    "args": ["${PLUGIN_ROOT}/server.js", "你好"],
                    "env": {"CACHE": "${PLUGIN_DATA}/cache"},
                    "cwd": "./workspace"
                },
                "remote": {
                    "type": "streamable-http",
                    "url": "http://localhost:8080/mcp",
                    "headers": {"X-Literal": "${PLUGIN_ROOT}"}
                }
            }
        });
        let loaded = McpConfigFile::parse_agent_plugin(&document.to_string(), &root, &data)
            .map_err(|error| error.to_string())?;
        assert!(loaded.diagnostics.is_empty());
        let local = loaded
            .config
            .mcp_servers
            .get("local")
            .ok_or_else(|| "local server missing".to_string())?;
        let canonical_root = std::fs::canonicalize(&root)
            .map_err(|error| error.to_string())?
            .to_string_lossy()
            .into_owned();
        assert!(
            local
                .args
                .iter()
                .any(|argument| argument.ends_with("/server.js"))
        );
        assert_eq!(local.args.get(1).map(String::as_str), Some("你好"));
        assert_eq!(
            local.env.get("PLUGIN_ROOT").map(String::as_str),
            Some(canonical_root.as_str())
        );
        let remote = loaded
            .config
            .mcp_servers
            .get("remote")
            .ok_or_else(|| "remote server missing".to_string())?;
        assert_eq!(
            remote.headers.get("X-Literal").map(String::as_str),
            Some("${PLUGIN_ROOT}")
        );
        Ok(())
    }

    /// 验证无效的服务端条目会被单独跳过并记录诊断信息，不影响其他有效条目
    #[test]
    fn isolates_invalid_agent_plugin_mcp_entries() -> std::result::Result<(), String> {
        let (_temporary, root, data) = plugin_directories()?;
        let document = serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_V1,
            "mcpServers": {
                "valid": {"type": "stdio", "command": "node"},
                "invalid": {"type": "stdio", "command": "node", "future": true},
                "insecure": {"type": "streamable-http", "url": "http://example.com/mcp"}
            }
        });
        let loaded = McpConfigFile::parse_agent_plugin(&document.to_string(), &root, &data)
            .map_err(|error| error.to_string())?;
        assert!(loaded.config.mcp_servers.contains_key("valid"));
        assert!(!loaded.config.mcp_servers.contains_key("invalid"));
        assert!(!loaded.config.mcp_servers.contains_key("insecure"));
        assert_eq!(loaded.diagnostics.len(), 2);
        Ok(())
    }

    /// 验证顶层结构不合法（存在未知字段）时整体解析会返回错误
    #[test]
    fn rejects_invalid_agent_plugin_mcp_top_level() -> std::result::Result<(), String> {
        let (_temporary, root, data) = plugin_directories()?;
        let document = serde_json::json!({
            "$schema": AGENT_PLUGIN_MCP_SCHEMA_V1,
            "mcpServers": {},
            "future": true
        });
        assert!(McpConfigFile::parse_agent_plugin(&document.to_string(), &root, &data).is_err());
        Ok(())
    }

    /// 验证变量展开是单遍扫描（不会对展开结果递归再展开）且对 Unicode 字符安全
    #[test]
    fn variable_expansion_is_single_pass_and_unicode_safe() {
        let root = Path::new("/tmp/${PLUGIN_DATA}/插件");
        let data = Path::new("/tmp/data");
        assert_eq!(
            expand_agent_plugin_variables("前缀/${PLUGIN_ROOT}", root, data),
            "前缀//tmp/${PLUGIN_DATA}/插件"
        );
    }

    /// 验证配置的 Debug 输出会对环境变量和请求头中的敏感值做脱敏处理
    #[test]
    fn parsed_config_debug_redacts_environment_and_headers() -> std::result::Result<(), String> {
        let config = McpConfigFile::parse(
            r#"{
                "mcpServers": {
                    "local": {
                        "command": "node",
                        "env": {"TOKEN": "stdio-secret"}
                    },
                    "remote": {
                        "url": "https://example.invalid/mcp",
                        "headers": {"Authorization": "Bearer header-secret"}
                    }
                }
            }"#,
        )
        .map_err(|error| error.to_string())?;

        let debug = format!("{config:?}");
        for secret in ["stdio-secret", "header-secret"] {
            assert!(!debug.contains(secret), "debug leaked secret: {debug}");
        }
        Ok(())
    }
}
