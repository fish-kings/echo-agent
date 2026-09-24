//! 插件能力类型 —— 描述插件能够提供什么。
//!
//! 每个变体都对应一种组件类型，在插件加载期间会被接入相应的子系统。

use serde::{Deserialize, Serialize};

/// 插件所提供的内容。
///
/// 能力在插件清单中声明，并在加载阶段被解析。每种能力类型
/// 都对应一个具体的子系统：
///
/// | 能力 | 目标子系统 |
/// |-----------|-----------------|
/// | `Skill` | `SkillRegistry` |
/// | `Hook` | `HookRegistry` |
/// | `McpServer` | `McpManager` |
/// | `LspServer` | `LspManager` |
/// | `Agent` | `SubagentRegistry` |
/// | `Tool` | `ToolManager`（原生代码插件，未来支持） |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginCapability {
    /// 基于文件的技能（SKILL.md 文件）—— 注册到 `SkillRegistry`。
    Skill,
    /// Hook 定义（hooks.yaml）—— 注册到 `HookRegistry`。
    Hook,
    /// MCP 服务器配置（`mcp.json`）—— 通过 `McpManager` 连接。
    McpServer,
    /// LSP 服务器配置 —— 通过 `LspManager` 启动。
    LspServer,
    /// Agent 定义文件（agents/*.md）—— 注册到 `SubagentRegistry`。
    Agent,
    /// 原生代码工具 —— 注册到 `ToolManager`（需要 entry_point，未来支持）。
    Tool,
}

impl PluginCapability {
    /// 返回人类可读的展示名称。
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Skill => "Skills",
            Self::Hook => "Hooks",
            Self::McpServer => "MCP Servers",
            Self::LspServer => "LSP Servers",
            Self::Agent => "Agents",
            Self::Tool => "Tools",
        }
    }
}

impl std::str::FromStr for PluginCapability {
    type Err = String;

    /// 从字符串解析出对应的插件能力（大小写不敏感，支持单复数别名）。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "skill" | "skills" => Ok(Self::Skill),
            "hook" | "hooks" => Ok(Self::Hook),
            "mcp" | "mcp_server" | "mcpserver" | "mcp_servers" => Ok(Self::McpServer),
            "lsp" | "lsp_server" | "lspserver" | "lsp_servers" => Ok(Self::LspServer),
            "agent" | "agents" => Ok(Self::Agent),
            "tool" | "tools" => Ok(Self::Tool),
            _ => Err(format!("unknown plugin capability: {value}")),
        }
    }
}

impl std::fmt::Display for PluginCapability {
    /// 将能力格式化为其展示名称。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}
