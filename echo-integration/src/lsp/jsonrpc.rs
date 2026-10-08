//! 用于 LSP 通信的 JSON-RPC 2.0 消息类型。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 请求。
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    /// 协议版本，固定为 `"2.0"`
    pub jsonrpc: String,
    /// 请求 ID，用于匹配响应
    pub id: u64,
    /// 调用的方法名
    pub method: String,
    /// 方法参数（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// 创建一个 JSON-RPC 2.0 请求。
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 通知（没有 id，也不期望响应）。
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcNotification {
    /// 协议版本，固定为 `"2.0"`
    pub jsonrpc: String,
    /// 通知的方法名
    pub method: String,
    /// 通知参数（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcNotification {
    /// 创建一个 JSON-RPC 2.0 通知。
    pub fn new(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 响应。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    /// 协议版本
    pub jsonrpc: String,
    /// 对应请求的 ID（服务端主动发送的消息可能没有）
    pub id: Option<u64>,
    /// 成功时的结果
    #[serde(default)]
    pub result: Option<Value>,
    /// 失败时的错误对象
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 错误对象。
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    /// 错误码
    pub code: i64,
    /// 错误描述
    pub message: String,
    /// 附加错误数据（可选）
    #[serde(default)]
    pub data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    /// 以 `JSON-RPC error <code>: <message>` 格式输出。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

/**
将 JSON-RPC 消息编码为带 Content-Length 分帧的 LSP 消息。

LSP 使用如下格式：
```text
Content-Length: <length>\r\n
\r\n
<json body>
```
*/
pub fn encode_message(msg: &impl Serialize) -> Result<Vec<u8>, serde_json::Error> {
    let body = serde_json::to_vec(msg)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    let mut bytes = Vec::with_capacity(header.len() + body.len());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

/// 从一行头部中解析 Content-Length 的值。
pub fn parse_content_length(line: &str) -> Option<usize> {
    let line = line.trim();
    let (name, value) = line.split_once(':')?;
    name.eq_ignore_ascii_case("content-length")
        .then(|| value.trim().parse().ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 JSON-RPC 消息编码（含 Content-Length 头）。
    #[test]
    fn test_encode_message() -> Result<(), String> {
        let req = JsonRpcRequest::new(1, "initialize", None);
        let bytes = encode_message(&req).map_err(|error| error.to_string())?;
        let msg = String::from_utf8(bytes).map_err(|error| error.to_string())?;
        assert!(msg.starts_with("Content-Length:"));
        assert!(msg.contains("\"method\":\"initialize\""));
        Ok(())
    }

    /// 验证 Content-Length 头解析。
    #[test]
    fn test_parse_content_length() {
        assert_eq!(parse_content_length("Content-Length: 42"), Some(42));
        assert_eq!(parse_content_length("content-length: 100"), Some(100));
        assert_eq!(parse_content_length("Other-Header: foo"), None);
        assert_eq!(parse_content_length("Content-Length: 中文"), None);
        assert_eq!(parse_content_length("Ｃontent-Length: 4"), None);
    }

    /// 验证成功响应解析。
    #[test]
    fn test_parse_response() -> Result<(), String> {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"capabilities":{}}}"#;
        let resp: JsonRpcResponse =
            serde_json::from_str(json).map_err(|error| error.to_string())?;
        assert_eq!(resp.id, Some(1));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
        Ok(())
    }

    /// 验证错误响应解析。
    #[test]
    fn test_parse_error_response() -> Result<(), String> {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse =
            serde_json::from_str(json).map_err(|error| error.to_string())?;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.map(|error| error.code), Some(-32601));
        Ok(())
    }
}
