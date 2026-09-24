//! MCP（Model Context Protocol，模型上下文协议）门面模块。
//!
//! 重新导出 `echo_integration::mcp` 的公共 API。
//! 权威实现位于 `echo_integration`。

/// 直接从 `echo_integration::mcp` 重新导出。
pub mod integration {
    pub use echo_integration::mcp::*;
}

pub use echo_integration::mcp::{
    AGENT_PLUGIN_MCP_SCHEMA_V1, AgentPluginMcpLoad, LIST_MCP_RESOURCE_TEMPLATES_TOOL,
    LIST_MCP_RESOURCES_TOOL, MCP_RESOURCE_TOOL_NAMES, McpClient, McpConfigFile, McpManager,
    McpServer, McpServerConfig, McpServerEntry, McpServerId, McpServerOwner, McpTargetChange,
    McpTargetReceipt, McpToolAdapter, READ_MCP_RESOURCE_TOOL, TransportConfig,
    build_mcp_resource_tools, build_mcp_resource_tools_by_id,
};
pub use echo_integration::mcp::{
    McpContent, McpPrompt, McpPromptGetResult, McpResource, McpResourceReadResult,
    McpResourceTemplate, McpResourceTemplatesListResult, McpTool, McpToolCallResult,
    ServerCapabilities,
};
