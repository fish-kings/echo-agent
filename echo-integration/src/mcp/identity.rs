use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

/**
拥有某个 MCP 服务端连接的稳定归属方。

`Direct` 保留了历史上的公开 API。插件身份以准备好的插件快照所提供的
规范插件 ID 保存；管理器绝不会通过解析投影后的工具名或资源名来推导归属。
*/
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum McpServerOwner {
    /// 由应用直接配置的服务端
    Direct,
    /// 由插件提供的服务端，携带规范插件 ID
    Plugin(String),
}

impl McpServerOwner {
    /// 以插件 ID 构造插件归属方。
    pub fn plugin(id: impl Into<String>) -> Self {
        Self::Plugin(id.into())
    }

    /// 是否为直接配置的归属方。
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct)
    }

    /// 返回稳定名称：插件归属方返回插件 ID，直接归属方返回 `None`。
    pub fn stable_name(&self) -> Option<&str> {
        match self {
            Self::Direct => None,
            Self::Plugin(id) => Some(id),
        }
    }
}

impl fmt::Display for McpServerOwner {
    /// 输出 `direct` 或 `plugin:<id>`。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => formatter.write_str("direct"),
            Self::Plugin(id) => write!(formatter, "plugin:{id}"),
        }
    }
}

/// 某个 MCP 服务端在其归属方命名空间内的规范身份。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct McpServerId {
    /// 服务端的归属方
    pub owner: McpServerOwner,
    /// 在归属方命名空间内的本地名称
    pub local_name: String,
}

impl McpServerId {
    /// 构造直接配置的服务端身份。
    pub fn direct(local_name: impl Into<String>) -> Self {
        Self {
            owner: McpServerOwner::Direct,
            local_name: local_name.into(),
        }
    }

    /// 构造插件提供的服务端身份。
    pub fn plugin(owner: impl Into<String>, local_name: impl Into<String>) -> Self {
        Self {
            owner: McpServerOwner::plugin(owner),
            local_name: local_name.into(),
        }
    }

    /**
    生成可往返解析的选择器字符串。

    直接归属方通常直接使用本地名称（若名称以 `plugin:` 或 `direct:` 开头则编码为
    `direct:<base64>` 以免歧义）；插件归属方编码为 `plugin:<base64>:<base64>`。
    */
    pub fn selector(&self) -> String {
        match &self.owner {
            McpServerOwner::Direct
                if self.local_name.starts_with("plugin:")
                    || self.local_name.starts_with("direct:") =>
            {
                format!("direct:{}", encode_segment(&self.local_name))
            }
            McpServerOwner::Direct => self.local_name.clone(),
            McpServerOwner::Plugin(plugin) => format!(
                "plugin:{}:{}",
                encode_segment(plugin),
                encode_segment(&self.local_name)
            ),
        }
    }

    /// 解析 [`Self::selector`] 生成的选择器；无前缀的字符串视为直接配置的本地名称。
    pub fn parse_selector(selector: &str) -> Option<Self> {
        let mut parts = selector.split(':');
        match parts.next()? {
            "direct" => {
                let local_name = decode_segment(parts.next()?)?;
                if parts.next().is_some() {
                    return None;
                }
                Some(Self::direct(local_name))
            }
            "plugin" => {
                let plugin = decode_segment(parts.next()?)?;
                let local_name = decode_segment(parts.next()?)?;
                if parts.next().is_some() {
                    return None;
                }
                Some(Self::plugin(plugin, local_name))
            }
            _ => Some(Self::direct(selector.to_string())),
        }
    }
}

impl fmt::Display for McpServerId {
    /// 输出 `<归属方>:<本地名称>`。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.owner, self.local_name)
    }
}

/// 以 URL 安全、无填充的 base64 编码选择器片段。
fn encode_segment(value: &str) -> String {
    URL_SAFE_NO_PAD.encode(value.as_bytes())
}

/// 解码选择器片段；非法 base64 或非 UTF-8 时返回 `None`。
fn decode_segment(value: &str) -> Option<String> {
    let bytes = URL_SAFE_NO_PAD.decode(value).ok()?;
    String::from_utf8(bytes).ok()
}

/**
有损的工具名投影；当字符清洗可能把不同的规范名称合并为同一个时，附加摘要后缀。

摘要基于完整的 Unicode 身份计算，因此规范身份仍可在元数据中获取。
*/
pub fn plugin_tool_projection(plugin: &str, server: &str, tool: &str) -> String {
    let base = format!(
        "mcp__plugin_{}_{}__{}",
        slug(plugin),
        slug(server),
        slug(tool)
    );
    if [plugin, server, tool].iter().all(|value| {
        value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
    }) {
        return base;
    }
    let mut hasher = Sha256::new();
    hasher.update(plugin.as_bytes());
    hasher.update([0]);
    hasher.update(server.as_bytes());
    hasher.update([0]);
    hasher.update(tool.as_bytes());
    let digest = hex_digest(&hasher.finalize());
    format!("{base}__{}", digest.chars().take(12).collect::<String>())
}

/// 将非 ASCII 字母数字、`_`、`-` 的字符替换为 `_`；空结果返回 `unnamed`。
fn slug(value: &str) -> String {
    let mut result = String::new();
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
            result.push(character);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() {
        "unnamed".to_string()
    } else {
        result
    }
}

/// 将字节序列格式化为小写十六进制字符串。
fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证插件选择器可无损往返 Unicode 与标点字符。
    #[test]
    fn plugin_selector_round_trips_unicode_and_punctuation() {
        let id = McpServerId::plugin("供应商/α", "数据:server");
        assert_eq!(McpServerId::parse_selector(&id.selector()), Some(id));
    }

    /// 验证直连选择器保留旧版线上名称。
    #[test]
    fn direct_selector_keeps_legacy_wire_name() {
        let id = McpServerId::direct("filesystem");
        assert_eq!(id.selector(), "filesystem");
        assert_eq!(McpServerId::parse_selector("filesystem"), Some(id));
    }

    /// 验证直连选择器不会被误解析为插件选择器。
    #[test]
    fn direct_selector_cannot_be_misparsed_as_plugin() {
        let plugin = McpServerId::plugin("owner", "server");
        let direct = McpServerId::direct(plugin.selector());
        assert_ne!(direct.selector(), plugin.selector());
        assert_eq!(
            McpServerId::parse_selector(&direct.selector()),
            Some(direct)
        );
        assert_eq!(
            McpServerId::parse_selector(&plugin.selector()),
            Some(plugin)
        );
    }

    /// 验证有损的插件投射带有可区分的摘要。
    #[test]
    fn lossy_plugin_projection_has_distinguishing_digest() {
        let left = plugin_tool_projection("a/b", "same", "tool");
        let right = plugin_tool_projection("a_b", "same", "tool");
        assert_ne!(left, right);
    }

    /// 验证组件边界不会导致工具投射名称冲突。
    #[test]
    fn component_boundaries_cannot_collapse_tool_projection() {
        let left = plugin_tool_projection("a", "b_c", "tool");
        let right = plugin_tool_projection("a_b", "c", "tool");
        assert_ne!(left, right);
    }
}
