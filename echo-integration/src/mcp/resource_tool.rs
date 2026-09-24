use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use futures::future::{BoxFuture, join_all};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{McpClient, McpResource, McpResourceTemplate, McpServerId};
use echo_core::error::Result;
use echo_core::tools::{Tool, ToolFailureCategory, ToolParameters, ToolResult, ToolRiskLevel};

/// “列出 MCP 资源”工具的名称。
pub const LIST_MCP_RESOURCES_TOOL: &str = "list_mcp_resources";
/// “列出 MCP 资源模板”工具的名称。
pub const LIST_MCP_RESOURCE_TEMPLATES_TOOL: &str = "list_mcp_resource_templates";
/// “读取 MCP 资源”工具的名称。
pub const READ_MCP_RESOURCE_TOOL: &str = "read_mcp_resource";
/// 本模块注册的全部 MCP 资源工具名称集合。
pub const MCP_RESOURCE_TOOL_NAMES: [&str; 3] = [
    LIST_MCP_RESOURCES_TOOL,
    LIST_MCP_RESOURCE_TEMPLATES_TOOL,
    READ_MCP_RESOURCE_TOOL,
];

/// 分页返回结果时每页包含的最大条目数。
const PAGE_SIZE: usize = 50;

/// 支持资源能力的已连接 MCP 客户端目录，按服务器选择器（selector）索引。
#[derive(Clone)]
struct McpResourceDirectory {
    /// 从服务器选择器到对应客户端的只读映射，使用 `Arc` 便于跨工具克隆共享。
    clients: Arc<BTreeMap<String, ResourceClient>>,
}

/// 目录中记录的单个 MCP 资源客户端及其身份标识。
#[derive(Clone)]
struct ResourceClient {
    /// 该客户端所属的 MCP 服务器标识（可能是直连服务器，也可能是插件提供的服务器）。
    id: McpServerId,
    /// 指向底层 MCP 客户端实例的共享引用。
    client: Arc<McpClient>,
}

impl McpResourceDirectory {
    /// 从已连接的客户端集合中筛选出支持资源能力的客户端，构建目录。
    fn new(clients: HashMap<McpServerId, Arc<McpClient>>) -> Self {
        let clients = clients
            .into_iter()
            .filter(|(_, client)| client.supports_resources())
            .map(|(id, client)| (id.selector(), ResourceClient { id, client }))
            .collect::<BTreeMap<_, _>>();
        Self {
            clients: Arc::new(clients),
        }
    }

    /// 判断目录中是否没有任何支持资源能力的客户端。
    fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    /// 按可选的服务器选择器筛选客户端：指定服务器时只返回该服务器（未连接则报错），
    /// 未指定时返回目录中所有已连接的客户端。
    fn select(
        &self,
        server: Option<&str>,
    ) -> std::result::Result<Vec<(McpServerId, Arc<McpClient>)>, String> {
        match server {
            Some(server) => self
                .clients
                .get(server)
                .map(|entry| vec![(entry.id.clone(), Arc::clone(&entry.client))])
                .ok_or_else(|| format!("MCP resource server '{server}' is not connected")),
            None => Ok(self
                .clients
                .values()
                .map(|entry| (entry.id.clone(), Arc::clone(&entry.client)))
                .collect()),
        }
    }
}

/// 构建三个标准的、可跨服务器使用的 MCP 资源工具。
///
/// 返回的工具共享一份连接管理器当前客户端的不可变快照。
/// 当连接拓扑发生变化时，调用方需要重新生成并替换这些工具。
pub fn build_mcp_resource_tools(clients: HashMap<String, Arc<McpClient>>) -> Vec<Box<dyn Tool>> {
    build_mcp_resource_tools_by_id(
        clients
            .into_iter()
            .map(|(name, client)| (McpServerId::direct(name), client))
            .collect(),
    )
}

/// 基于带归属信息（owner-qualified）的 MCP 客户端标识构建资源工具。
pub fn build_mcp_resource_tools_by_id(
    clients: HashMap<McpServerId, Arc<McpClient>>,
) -> Vec<Box<dyn Tool>> {
    let directory = McpResourceDirectory::new(clients);
    if directory.is_empty() {
        return Vec::new();
    }
    vec![
        Box::new(ListMcpResourcesTool {
            directory: directory.clone(),
        }),
        Box::new(ListMcpResourceTemplatesTool {
            directory: directory.clone(),
        }),
        Box::new(ReadMcpResourceTool { directory }),
    ]
}

/// 分页游标的内部表示，编码后作为不透明字符串暴露给调用方。
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PageCursor {
    /// 结果集中下一页应从此偏移量开始读取。
    offset: usize,
}

/// 将分页偏移量编码为一个不透明的、URL 安全的游标字符串。
fn encode_cursor(offset: usize) -> std::result::Result<String, String> {
    serde_json::to_vec(&PageCursor { offset })
        .map(|bytes| URL_SAFE_NO_PAD.encode(bytes))
        .map_err(|error| format!("failed to encode MCP resource cursor: {error}"))
}

/// 解码调用方传入的游标字符串，得到对应的分页偏移量；未提供游标时视为从头开始（偏移量为 0）。
fn decode_cursor(cursor: Option<&str>) -> std::result::Result<usize, String> {
    let Some(cursor) = cursor else {
        return Ok(0);
    };
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|error| format!("invalid MCP resource cursor: {error}"))?;
    serde_json::from_slice::<PageCursor>(&bytes)
        .map(|cursor| cursor.offset)
        .map_err(|error| format!("invalid MCP resource cursor payload: {error}"))
}

/// 从参数中读取一个可选的字符串参数：缺失时返回 `None`，存在但为空白或非字符串则报错。
fn optional_string(
    parameters: &ToolParameters,
    name: &str,
) -> std::result::Result<Option<String>, String> {
    let Some(value) = parameters.get(name) else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .ok_or_else(|| format!("parameter '{name}' must be a string"))?
        .trim();
    if value.is_empty() {
        return Err(format!("parameter '{name}' must not be empty"));
    }
    Ok(Some(value.to_string()))
}

/// 从参数中读取一个必填的字符串参数；缺失时返回“缺少必填参数”的错误。
fn required_string(parameters: &ToolParameters, name: &str) -> std::result::Result<String, String> {
    optional_string(parameters, name)?.ok_or_else(|| format!("missing required parameter '{name}'"))
}

/// 单条资源列表项，附带其所属的服务器选择器，用于跨服务器聚合展示。
#[derive(Serialize)]
struct ResourceEntry {
    /// 提供该资源的 MCP 服务器选择器。
    server: String,
    /// 原始的 MCP 资源数据，序列化时展开（flatten）合并到外层对象。
    #[serde(flatten)]
    resource: McpResource,
}

/// 单条资源模板列表项，附带其所属的服务器选择器，用于跨服务器聚合展示。
#[derive(Serialize)]
struct ResourceTemplateEntry {
    /// 提供该资源模板的 MCP 服务器选择器。
    server: String,
    /// 原始的 MCP 资源模板数据，序列化时展开（flatten）合并到外层对象。
    #[serde(flatten)]
    template: McpResourceTemplate,
}

/// 在跨服务器聚合列表时，记录某个服务器调用失败的信息。
#[derive(Serialize)]
struct ServerError {
    /// 调用失败的 MCP 服务器选择器。
    server: String,
    /// 失败原因的描述信息。
    message: String,
}

/// 将完整的条目列表裁剪为一页（最多 `PAGE_SIZE` 条），并构建包含分页元数据
/// （总数、本页返回数、是否被截断、下一页游标等）的工具结果。
fn paginated_result<T: Serialize>(
    key: &str,
    items: Vec<T>,
    errors: Vec<ServerError>,
    offset: usize,
) -> std::result::Result<ToolResult, String> {
    let total = items.len();
    if offset > total {
        return Err(format!(
            "MCP resource cursor offset {offset} exceeds result count {total}"
        ));
    }
    let page = items
        .into_iter()
        .skip(offset)
        .take(PAGE_SIZE)
        .collect::<Vec<_>>();
    let returned = page.len();
    let end = offset.saturating_add(returned);
    let next_cursor = if end < total {
        Some(encode_cursor(end)?)
    } else {
        None
    };

    let mut payload = Map::new();
    payload.insert(
        key.to_string(),
        serde_json::to_value(page)
            .map_err(|error| format!("failed to serialize MCP resources: {error}"))?,
    );
    payload.insert(
        "next_cursor".to_string(),
        next_cursor
            .as_ref()
            .map_or(Value::Null, |cursor| Value::String(cursor.clone())),
    );
    if !errors.is_empty() {
        payload.insert(
            "errors".to_string(),
            serde_json::to_value(errors)
                .map_err(|error| format!("failed to serialize MCP resource errors: {error}"))?,
        );
    }

    let mut result = ToolResult::success_json(Value::Object(payload))
        .with_truncated(next_cursor.is_some())
        .with_meta("tool_source", "mcp")
        .with_meta("page.total", total.to_string())
        .with_meta("page.returned", returned.to_string())
        .with_meta("page.truncated", next_cursor.is_some().to_string());
    if let Some(cursor) = next_cursor {
        result = result.with_meta("page.next_cursor", cursor);
    }
    Ok(result)
}

/// 构建“列出资源”与“列出资源模板”两个工具共用的 JSON Schema 参数定义
/// （可选的 `server` 与 `cursor` 参数）。
fn list_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "MCP server name. Omit to list entries from every connected resource server."
            },
            "cursor": {
                "type": "string",
                "description": "Opaque cursor from a previous call; omit for the first page."
            }
        },
        "additionalProperties": false
    })
}

/// “列出 MCP 资源”工具：跨所有（或指定）已连接的资源服务器，聚合分页返回资源列表。
pub struct ListMcpResourcesTool {
    /// 已连接、支持资源能力的 MCP 客户端目录。
    directory: McpResourceDirectory,
}

impl Tool for ListMcpResourcesTool {
    /// 返回该工具的注册名称。
    fn name(&self) -> &str {
        LIST_MCP_RESOURCES_TOOL
    }

    /// 返回展示给模型的工具说明文本。
    fn description(&self) -> &str {
        "List resources provided by connected MCP servers. Prefer these application-provided resources over web search when they can supply the needed context."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        list_parameters()
    }

    /// 声明该工具为只读操作。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 执行资源列表查询：可选按 `server` 过滤，支持通过 `cursor` 分页；
    /// 未指定服务器时并发查询所有已连接服务器并合并结果，单个服务器失败仅记录为错误项，
    /// 而指定服务器查询失败时直接返回失败结果。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let server = match optional_string(&parameters, "server") {
                Ok(server) => server,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let cursor = match optional_string(&parameters, "cursor")
                .and_then(|cursor| decode_cursor(cursor.as_deref()))
            {
                Ok(cursor) => cursor,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let clients = match self.directory.select(server.as_deref()) {
                Ok(clients) => clients,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let targeted = server.is_some();
            let requests = clients.into_iter().map(|(name, client)| async move {
                let result = client.list_resources().await;
                (name, result)
            });
            let mut resources = Vec::new();
            let mut errors = Vec::new();
            for (name, response) in join_all(requests).await {
                match response {
                    Ok(entries) => {
                        resources.extend(entries.into_iter().map(|resource| ResourceEntry {
                            server: name.selector(),
                            resource,
                        }))
                    }
                    Err(error) if targeted => {
                        return Ok(ToolResult::failure(
                            ToolFailureCategory::Unavailable,
                            format!("MCP server '{name}' resources/list failed: {error}"),
                        ));
                    }
                    Err(error) => errors.push(ServerError {
                        server: name.selector(),
                        message: error.to_string(),
                    }),
                }
            }
            resources.sort_by(|left, right| {
                left.server
                    .cmp(&right.server)
                    .then_with(|| left.resource.uri.cmp(&right.resource.uri))
            });
            Ok(paginated_result("resources", resources, errors, cursor)
                .unwrap_or_else(ToolResult::invalid_arguments))
        })
    }
}

/// “列出 MCP 资源模板”工具：跨所有（或指定）已连接的资源服务器，聚合分页返回参数化资源模板列表。
pub struct ListMcpResourceTemplatesTool {
    /// 已连接、支持资源能力的 MCP 客户端目录。
    directory: McpResourceDirectory,
}

impl Tool for ListMcpResourceTemplatesTool {
    /// 返回该工具的注册名称。
    fn name(&self) -> &str {
        LIST_MCP_RESOURCE_TEMPLATES_TOOL
    }

    /// 返回展示给模型的工具说明文本。
    fn description(&self) -> &str {
        "List parameterized resource templates provided by connected MCP servers. Prefer these application-provided templates over web search when appropriate."
    }

    /// 返回该工具的 JSON Schema 参数定义。
    fn parameters(&self) -> Value {
        list_parameters()
    }

    /// 声明该工具为只读操作。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 执行资源模板列表查询：可选按 `server` 过滤，支持通过 `cursor` 分页；
    /// 未指定服务器时并发查询所有已连接服务器并合并结果，单个服务器失败仅记录为错误项，
    /// 而指定服务器查询失败时直接返回失败结果。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let server = match optional_string(&parameters, "server") {
                Ok(server) => server,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let cursor = match optional_string(&parameters, "cursor")
                .and_then(|cursor| decode_cursor(cursor.as_deref()))
            {
                Ok(cursor) => cursor,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let clients = match self.directory.select(server.as_deref()) {
                Ok(clients) => clients,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let targeted = server.is_some();
            let requests = clients.into_iter().map(|(name, client)| async move {
                let result = client.list_resource_templates().await;
                (name, result)
            });
            let mut templates = Vec::new();
            let mut errors = Vec::new();
            for (name, response) in join_all(requests).await {
                match response {
                    Ok(entries) => templates.extend(entries.into_iter().map(|template| {
                        ResourceTemplateEntry {
                            server: name.selector(),
                            template,
                        }
                    })),
                    Err(error) if targeted => {
                        return Ok(ToolResult::failure(
                            ToolFailureCategory::Unavailable,
                            format!("MCP server '{name}' resources/templates/list failed: {error}"),
                        ));
                    }
                    Err(error) => errors.push(ServerError {
                        server: name.selector(),
                        message: error.to_string(),
                    }),
                }
            }
            templates.sort_by(|left, right| {
                left.server
                    .cmp(&right.server)
                    .then_with(|| left.template.uri_template.cmp(&right.template.uri_template))
            });
            Ok(
                paginated_result("resource_templates", templates, errors, cursor)
                    .unwrap_or_else(ToolResult::invalid_arguments),
            )
        })
    }
}

/// “读取 MCP 资源”工具：按服务器名和资源 URI 精确读取单个资源的内容。
pub struct ReadMcpResourceTool {
    /// 已连接、支持资源能力的 MCP 客户端目录。
    directory: McpResourceDirectory,
}

impl Tool for ReadMcpResourceTool {
    /// 返回该工具的注册名称。
    fn name(&self) -> &str {
        READ_MCP_RESOURCE_TOOL
    }

    /// 返回展示给模型的工具说明文本。
    fn description(&self) -> &str {
        "Read a specific resource from a connected MCP server using its exact server name and resource URI."
    }

    /// 返回该工具的 JSON Schema 参数定义（必填的 `server` 与 `uri`）。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "MCP server name exactly as returned by list_mcp_resources."
                },
                "uri": {
                    "type": "string",
                    "description": "Resource URI exactly as returned by list_mcp_resources."
                }
            },
            "required": ["server", "uri"],
            "additionalProperties": false
        })
    }

    /// 声明该工具为只读操作。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 执行单个资源的读取：校验服务器已连接后调用其 `resources/read`，
    /// 并在返回的 JSON 负载中附加 `server`、`uri` 字段方便调用方核对来源。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let server = match required_string(&parameters, "server") {
                Ok(server) => server,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let uri = match required_string(&parameters, "uri") {
                Ok(uri) => uri,
                Err(error) => return Ok(ToolResult::invalid_arguments(error)),
            };
            let client = match self.directory.clients.get(&server) {
                Some(entry) => Arc::clone(&entry.client),
                None => {
                    return Ok(ToolResult::invalid_arguments(format!(
                        "MCP resource server '{server}' is not connected"
                    )));
                }
            };
            let resource = match client.read_resource(&uri).await {
                Ok(resource) => resource,
                Err(error) => {
                    return Ok(ToolResult::failure(
                        ToolFailureCategory::Unavailable,
                        format!("MCP server '{server}' resources/read failed: {error}"),
                    ));
                }
            };
            let payload = match serde_json::to_value(resource) {
                Ok(Value::Object(mut payload)) => {
                    payload.insert("server".to_string(), Value::String(server.clone()));
                    payload.insert("uri".to_string(), Value::String(uri.clone()));
                    Value::Object(payload)
                }
                Ok(payload) => payload,
                Err(error) => {
                    return Ok(ToolResult::error(format!(
                        "failed to serialize MCP resource: {error}"
                    )));
                }
            };
            Ok(ToolResult::success_json(payload)
                .with_meta("tool_source", "mcp")
                .with_meta("mcp_server", server)
                .with_meta("mcp_resource_uri", uri))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::McpTransport;
    use crate::mcp::types::{
        JsonRpcError, JsonRpcNotification, JsonRpcNotificationReceiver, JsonRpcRequest,
        JsonRpcResponse,
    };

    /// 测试用的假 MCP 传输层：对 `resources/list`、`resources/templates/list`、
    /// `resources/read` 请求返回预设的成功响应，用于验证资源工具的正常路径。
    struct ResourceTransport;

    /// 测试用的假 MCP 传输层：对任意请求都返回错误响应，用于验证资源工具的失败处理路径。
    struct FailingResourceTransport;

    impl McpTransport for ResourceTransport {
        /// 根据请求方法名返回预设的分页/读取响应；不支持的方法返回 JSON-RPC 错误。
        fn send(&self, request: JsonRpcRequest) -> BoxFuture<'_, Result<JsonRpcResponse>> {
            Box::pin(async move {
                let result = match request.method.as_str() {
                    "resources/list" => {
                        let cursor = request
                            .params
                            .as_ref()
                            .and_then(|params| params.get("cursor"))
                            .and_then(Value::as_str);
                        if cursor == Some("resources-2") {
                            json!({
                                "resources": [{
                                    "uri": "schema://orders",
                                    "name": "orders",
                                    "title": "Orders",
                                    "mimeType": "application/schema+json",
                                    "icons": []
                                }]
                            })
                        } else {
                            json!({
                                "resources": [{
                                    "uri": "file:///project/说明.txt",
                                    "name": "说明.txt",
                                    "description": "UTF-8 resource",
                                    "mimeType": "text/plain",
                                    "icons": []
                                }],
                                "nextCursor": "resources-2"
                            })
                        }
                    }
                    "resources/templates/list" => {
                        let cursor = request
                            .params
                            .as_ref()
                            .and_then(|params| params.get("cursor"))
                            .and_then(Value::as_str);
                        if cursor == Some("templates-2") {
                            json!({
                                "resourceTemplates": [{
                                    "uriTemplate": "schema://{table}",
                                    "name": "database schema",
                                    "icons": []
                                }]
                            })
                        } else {
                            json!({
                                "resourceTemplates": [{
                                    "uriTemplate": "file:///{path}",
                                    "name": "project file",
                                    "title": "Project File",
                                    "icons": []
                                }],
                                "nextCursor": "templates-2"
                            })
                        }
                    }
                    "resources/read" => json!({
                        "contents": [
                            {
                                "uri": "file:///project/说明.txt",
                                "mimeType": "text/plain",
                                "text": "你好，MCP",
                                "audience": ["assistant"]
                            },
                            {
                                "uri": "file:///project/pixel.png",
                                "mimeType": "image/png",
                                "blob": "iVBORw0KGgo="
                            }
                        ]
                    }),
                    method => {
                        return Ok(JsonRpcResponse {
                            jsonrpc: "2.0".to_string(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32601,
                                message: format!("unsupported method '{method}'"),
                                data: None,
                            }),
                        });
                    }
                };
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                })
            })
        }

        /// 测试用空实现：不做任何事，直接返回成功。
        fn notify(&self, _notification: JsonRpcNotification) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        /// 测试用空实现：不做任何事，直接返回成功。
        fn close(&self) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        /// 测试传输层不支持通知通道，始终返回 `None`。
        fn notification_rx(&self) -> Option<Arc<dyn JsonRpcNotificationReceiver>> {
            None
        }
    }

    impl McpTransport for FailingResourceTransport {
        /// 对任意请求都返回“资源后端不可用”的 JSON-RPC 错误响应。
        fn send(&self, request: JsonRpcRequest) -> BoxFuture<'_, Result<JsonRpcResponse>> {
            Box::pin(async move {
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32603,
                        message: "resource backend unavailable".to_string(),
                        data: None,
                    }),
                })
            })
        }

        /// 测试用空实现：不做任何事，直接返回成功。
        fn notify(&self, _notification: JsonRpcNotification) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        /// 测试用空实现：不做任何事，直接返回成功。
        fn close(&self) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }

        /// 测试传输层不支持通知通道，始终返回 `None`。
        fn notification_rx(&self) -> Option<Arc<dyn JsonRpcNotificationReceiver>> {
            None
        }
    }

    /// 构建一个连接单个测试服务器（名为 `context`）的资源工具集合，供各用例复用。
    fn resource_tools() -> Vec<Box<dyn Tool>> {
        let client = McpClient::with_test_transport("context", Arc::new(ResourceTransport));
        build_mcp_resource_tools_by_id(HashMap::from([(McpServerId::direct("context"), client)]))
    }

    /// 在给定工具集合中按名称查找工具并执行，找不到时返回错误。
    async fn execute_named(
        tools: &[Box<dyn Tool>],
        name: &str,
        parameters: ToolParameters,
    ) -> std::result::Result<ToolResult, String> {
        let tool = tools
            .iter()
            .find(|tool| tool.name() == name)
            .ok_or_else(|| format!("missing tool '{name}'"))?;
        tool.execute(parameters)
            .await
            .map_err(|error| error.to_string())
    }

    /// 验证游标编码/解码可以正确往返，且非法游标字符串会被拒绝。
    #[test]
    fn cursor_round_trip_and_validation() -> std::result::Result<(), String> {
        let cursor = encode_cursor(51)?;
        assert_eq!(decode_cursor(Some(&cursor))?, 51);
        assert!(decode_cursor(Some("not-a-cursor")).is_err());
        Ok(())
    }

    /// 验证分页结果会被限制在 `PAGE_SIZE` 以内，并正确报告是否存在下一页游标。
    #[test]
    fn pagination_is_bounded_and_reports_continuation() -> std::result::Result<(), String> {
        let items = (0..51).collect::<Vec<_>>();
        let result = paginated_result("resources", items, Vec::new(), 0)?;
        assert!(result.success);
        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("page.returned").map(String::as_str),
            Some("50")
        );
        let next = result
            .metadata
            .get("page.next_cursor")
            .ok_or_else(|| "missing next cursor".to_string())?;
        assert_eq!(decode_cursor(Some(next))?, 50);
        Ok(())
    }

    /// 验证当没有任何支持资源能力的客户端时，不会注册任何资源工具。
    #[test]
    fn empty_directory_registers_no_tools() {
        assert!(build_mcp_resource_tools_by_id(HashMap::new()).is_empty());
    }

    /// 验证三个标准资源工具均已注册、均为只读，并能分别正常列出资源、列出资源模板、读取资源内容。
    #[tokio::test]
    async fn canonical_tools_list_templates_and_read_contents() -> std::result::Result<(), String> {
        let tools = resource_tools();
        let mut names = tools
            .iter()
            .map(|tool| tool.name().to_string())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            vec![
                LIST_MCP_RESOURCE_TEMPLATES_TOOL.to_string(),
                LIST_MCP_RESOURCES_TOOL.to_string(),
                READ_MCP_RESOURCE_TOOL.to_string(),
            ]
        );
        assert!(
            tools
                .iter()
                .all(|tool| tool.risk_level() == ToolRiskLevel::ReadOnly)
        );

        let resources =
            execute_named(&tools, LIST_MCP_RESOURCES_TOOL, ToolParameters::new()).await?;
        assert!(resources.success);
        let resource_count = resources
            .data
            .as_ref()
            .and_then(|data| data.get("resources"))
            .and_then(Value::as_array)
            .map(Vec::len);
        assert_eq!(resource_count, Some(2));
        assert!(resources.output.contains("说明.txt"));

        let templates = execute_named(
            &tools,
            LIST_MCP_RESOURCE_TEMPLATES_TOOL,
            ToolParameters::new(),
        )
        .await?;
        let template_count = templates
            .data
            .as_ref()
            .and_then(|data| data.get("resource_templates"))
            .and_then(Value::as_array)
            .map(Vec::len);
        assert_eq!(template_count, Some(2));

        let resource = execute_named(
            &tools,
            READ_MCP_RESOURCE_TOOL,
            ToolParameters::from([
                ("server".to_string(), Value::String("context".to_string())),
                (
                    "uri".to_string(),
                    Value::String("file:///project/说明.txt".to_string()),
                ),
            ]),
        )
        .await?;
        assert!(resource.success);
        assert!(resource.output.contains("你好，MCP"));
        assert!(resource.output.contains("iVBORw0KGgo="));
        assert!(resource.output.contains("audience"));
        assert_eq!(
            resource.metadata.get("mcp_server").map(String::as_str),
            Some("context")
        );
        Ok(())
    }

    /// 验证指定一个未连接的服务器名时，会返回“参数错误”类别的失败结果。
    #[tokio::test]
    async fn unknown_server_is_an_argument_error() -> std::result::Result<(), String> {
        let tools = resource_tools();
        let result = execute_named(
            &tools,
            LIST_MCP_RESOURCES_TOOL,
            ToolParameters::from([("server".to_string(), Value::String("missing".to_string()))]),
        )
        .await?;
        assert!(!result.success);
        assert_eq!(
            result.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::InvalidArguments)
        );
        Ok(())
    }

    /// 验证不指定服务器时，聚合列表会保留健康服务器的结果并将失败服务器记为错误项；
    /// 而明确指定失败服务器时，则直接返回“不可用”的失败结果。
    #[tokio::test]
    async fn all_server_listing_keeps_successes_and_targeted_failure_is_explicit()
    -> std::result::Result<(), String> {
        let healthy = McpClient::with_test_transport("healthy", Arc::new(ResourceTransport));
        let failing = McpClient::with_test_transport("failing", Arc::new(FailingResourceTransport));
        let tools = build_mcp_resource_tools_by_id(HashMap::from([
            (McpServerId::direct("healthy"), healthy),
            (McpServerId::direct("failing"), failing),
        ]));

        let aggregate =
            execute_named(&tools, LIST_MCP_RESOURCES_TOOL, ToolParameters::new()).await?;
        assert!(aggregate.success);
        assert_eq!(
            aggregate
                .data
                .as_ref()
                .and_then(|data| data.get("resources"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
        assert_eq!(
            aggregate
                .data
                .as_ref()
                .and_then(|data| data.get("errors"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );

        let targeted = execute_named(
            &tools,
            LIST_MCP_RESOURCES_TOOL,
            ToolParameters::from([("server".to_string(), Value::String("failing".to_string()))]),
        )
        .await?;
        assert!(!targeted.success);
        assert_eq!(
            targeted.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::Unavailable)
        );
        Ok(())
    }

    /// 验证当两个不同插件提供的服务器暴露相同的资源 URI 时，各自的选择器仍保持唯一区分，
    /// 聚合列表中不会互相覆盖，且各自的资源都能被正确读取。
    #[tokio::test]
    async fn same_resource_uri_from_two_plugin_owners_keeps_distinct_selectors()
    -> std::result::Result<(), String> {
        let first_id = McpServerId::plugin("插件/A", "资料:shared");
        let second_id = McpServerId::plugin("plugin-b", "资料:shared");
        let tools = build_mcp_resource_tools_by_id(HashMap::from([
            (
                first_id.clone(),
                McpClient::with_test_transport("资料:shared", Arc::new(ResourceTransport)),
            ),
            (
                second_id.clone(),
                McpClient::with_test_transport("资料:shared", Arc::new(ResourceTransport)),
            ),
        ]));
        let aggregate =
            execute_named(&tools, LIST_MCP_RESOURCES_TOOL, ToolParameters::new()).await?;
        let resources = aggregate
            .data
            .as_ref()
            .and_then(|data| data.get("resources"))
            .and_then(Value::as_array)
            .ok_or_else(|| "resource list missing".to_string())?;
        assert_eq!(resources.len(), 4);
        let selectors = resources
            .iter()
            .filter_map(|entry| entry.get("server").and_then(Value::as_str))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(selectors.len(), 2);
        assert!(selectors.contains(first_id.selector().as_str()));
        assert!(selectors.contains(second_id.selector().as_str()));

        for id in [first_id, second_id] {
            let resource = execute_named(
                &tools,
                READ_MCP_RESOURCE_TOOL,
                ToolParameters::from([
                    ("server".to_string(), Value::String(id.selector())),
                    (
                        "uri".to_string(),
                        Value::String("file:///project/说明.txt".to_string()),
                    ),
                ]),
            )
            .await?;
            assert!(resource.success);
        }
        Ok(())
    }
}
