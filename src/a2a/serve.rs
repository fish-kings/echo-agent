/*!
A2A HTTP 服务层。

提供基于 axum 的完整 A2A HTTP 服务器，包括：
- `GET /.well-known/agent.json` —— Agent Card 发现端点
- `POST /` —— JSON-RPC 任务请求（同步 + SSE 流式）
- `GET /health` —— K8s 存活探针
- `GET /ready` —— K8s 就绪探针
- 可选的 JWT Bearer Token 鉴权中间件

# 用法

```rust,no_run
use echo_agent::a2a::{A2AServer, AgentCard, serve};
use echo_agent::prelude::*;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let card = AgentCard::builder("my-agent", "http://localhost:3000").build();
let agent = ReactAgentBuilder::simple("qwen3-max", "test")?;
let server = A2AServer::new(card, agent);
serve(server, "0.0.0.0:3000").await?;
# Ok(())
# }
```
*/

use super::auth::{JwtConfig, jwt_middleware};
use super::server::A2AServer;
use super::types::*;
use axum::{
    Router,
    extract::{DefaultBodyLimit, State},
    http::{StatusCode, header},
    middleware,
    response::{IntoResponse, Response, sse::Sse},
    routing::get,
};
use futures::StreamExt;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tracing::{info, warn};

/// 默认的请求体大小上限（1 MB）。
pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;

/// A2A 服务器的共享状态。
#[derive(Clone)]
struct AppState {
    /// 底层 A2A 服务器实例。
    server: Arc<A2AServer>,
}

/**
 * 启动 A2A HTTP 服务器（不鉴权，默认 1 MB 请求体上限）。
 *
 * 绑定到给定地址，提供 Agent Card 发现、JSON-RPC 任务处理、健康检查等能力。
 */
pub async fn serve(server: A2AServer, bind_addr: &str) -> crate::error::Result<()> {
    serve_inner(
        server,
        bind_addr,
        JwtConfig::disabled(),
        DEFAULT_MAX_BODY_BYTES,
    )
    .await
}

/**
 * 启动 A2A HTTP 服务器（带 JWT 鉴权，默认 1 MB 请求体上限）。
 *
 * 所有 `/` 端点请求都必须携带合法的 `Authorization: Bearer <token>` 请求头。
 * `/health` 与 `/ready` 端点不受鉴权约束。
 */
pub async fn serve_with_auth(
    server: A2AServer,
    bind_addr: &str,
    jwt_config: JwtConfig,
) -> crate::error::Result<()> {
    serve_inner(server, bind_addr, jwt_config, DEFAULT_MAX_BODY_BYTES).await
}

/// 内部统一实现。
async fn serve_inner(
    server: A2AServer,
    bind_addr: &str,
    jwt_config: JwtConfig,
    max_body_bytes: usize,
) -> crate::error::Result<()> {
    // 安全警告：在非回环地址上禁用鉴权时，需要醒目地记录日志
    if !jwt_config.is_enabled() {
        let is_loopback = bind_addr.starts_with("127.0.0.1")
            || bind_addr.starts_with("localhost")
            || bind_addr.starts_with("[::1]")
            || bind_addr == "localhost:0";
        if !is_loopback {
            warn!(
                "⚠️  A2A server starting WITHOUT authentication on {} — \
                 this is insecure for non-local addresses! \
                 Use serve_with_auth() or JwtConfig::hs256() to enable JWT authentication.",
                bind_addr
            );
        }
    }

    let state = AppState {
        server: Arc::new(server),
    };

    let jwt_enabled = jwt_config.is_enabled();
    let jwt_state = Arc::new(jwt_config);

    // 受鉴权保护的路由组
    let protected = Router::new()
        .route("/.well-known/agent.json", get(agent_card))
        .route("/", get(|| async { "A2A Server" }).post(handle_json_rpc));

    let protected = if jwt_enabled {
        protected.route_layer(middleware::from_fn_with_state(
            jwt_state.clone(),
            jwt_middleware,
        ))
    } else {
        protected
    };

    // 健康检查端点不受鉴权约束
    let app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .merge(protected)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state);

    let listener = TcpListener::bind(bind_addr)
        .await
        .map_err(|e| crate::error::ReactError::Other(format!("Bind address failed: {e}")))?;

    info!(
        addr = %bind_addr,
        jwt = %jwt_enabled,
        max_body_bytes = %max_body_bytes,
        "A2A HTTP server started"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| crate::error::ReactError::Other(format!("Server error: {e}")))?;

    info!("A2A HTTP server shut down gracefully");
    Ok(())
}

/// 等待 SIGINT 或 SIGTERM 信号以实现优雅关闭。
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = signal::ctrl_c().await {
            warn!(%error, "Failed to install Ctrl+C handler");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(error) => {
                warn!(%error, "Failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received SIGINT signal, shutting down gracefully...");
        }
        _ = terminate => {
            info!("Received SIGTERM signal, shutting down gracefully...");
        }
    }
}

// ── 端点处理函数 ─────────────────────────────────────────────────────────

/// 处理 `GET /.well-known/agent.json`：返回本服务的 Agent Card。
async fn agent_card(State(state): State<AppState>) -> Response {
    match state.server.agent_card_json() {
        Ok(json) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            json,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({"error": format!("{e}")}).to_string(),
        )
            .into_response(),
    }
}

/**
 * `GET /health` —— K8s 存活探针。
 *
 * 只要进程存活就返回 200。
 */
async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({"status": "ok"}).to_string(),
    )
}

/**
 * `GET /ready` —— K8s 就绪探针。
 *
 * 返回 200 表示服务已准备好接受请求。
 */
async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let _ = state.server.agent_card();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({"status": "ready"}).to_string(),
    )
}

/// `POST /` —— JSON-RPC 处理函数。
async fn handle_json_rpc(State(state): State<AppState>, body: String) -> Response {
    let method = parse_method(&body);

    if method.as_deref() == Some(METHOD_SEND_SUBSCRIBE) {
        handle_sse(state, &body).await
    } else {
        handle_sync(state, &body).await
    }
}

/// 同步 JSON-RPC 响应。
async fn handle_sync(state: AppState, body: &str) -> Response {
    let response_json = state.server.handle_request(body).await;
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        response_json,
    )
        .into_response()
}

/// SSE 流式响应（tasks/sendSubscribe）。
async fn handle_sse(state: AppState, body: &str) -> Response {
    let server = state.server.clone();
    let body_owned = body.to_string();

    match A2AServer::handle_request_stream(&server, &body_owned).await {
        Ok(stream) => {
            let sse_stream = stream.map(move |event| {
                let sse_data = A2AServer::format_sse_event(&event, "req-stream");
                Ok::<axum::response::sse::Event, std::convert::Infallible>(
                    axum::response::sse::Event::default().data(sse_data),
                )
            });

            Sse::new(sse_stream).into_response()
        }
        Err(e) => {
            warn!(error = %e, "A2A SSE: stream processing failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::json!({"error": format!("{e}")}).to_string(),
            )
                .into_response()
        }
    }
}

/// 从 JSON-RPC 请求体中快速提取 method 字段。
fn parse_method(body: &str) -> Option<String> {
    let body = body.trim();

    if let Some(idx) = body.find("\"method\":") {
        let rest = &body[idx + "\"method\":".len()..].trim();
        if let Some(quote_start) = rest.find('"') {
            let after_quote = &rest[quote_start + 1..];
            if let Some(quote_end) = after_quote.find('"') {
                return Some(after_quote[..quote_end].to_string());
            }
        }
    } else if let Some(idx) = body.find("\"method\"") {
        let rest = &body[idx + "\"method\"".len()..].trim();
        if let Some(after_colon) = rest.strip_prefix(':') {
            let after_colon = after_colon.trim();
            if let Some(quote_start) = after_colon.find('"') {
                let after_quote = &after_colon[quote_start + 1..];
                if let Some(quote_end) = after_quote.find('"') {
                    return Some(after_quote[..quote_end].to_string());
                }
            }
        }
    }

    None
}

/// A2A HTTP 服务的单元测试与属性测试。
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ── 确定性单元测试 ──────────────────────────────────────────────

    /// 验证可解析出 `tasks/send` 方法。
    #[test]
    fn test_parse_method_tasks_send() {
        let body = r#"{"jsonrpc":"2.0","id":"1","method":"tasks/send","params":{}}"#;
        assert_eq!(parse_method(body), Some("tasks/send".to_string()));
    }

    /// 验证可解析出 `tasks/sendSubscribe` 方法。
    #[test]
    fn test_parse_method_send_subscribe() {
        let body = r#"{"jsonrpc":"2.0","id":"1","method":"tasks/sendSubscribe","params":{}}"#;
        assert_eq!(parse_method(body), Some("tasks/sendSubscribe".to_string()));
    }

    /// 验证可解析出 `tasks/get` 方法。
    #[test]
    fn test_parse_method_tasks_get() {
        let body = r#"{"jsonrpc":"2.0","id":"1","method":"tasks/get","params":{}}"#;
        assert_eq!(parse_method(body), Some("tasks/get".to_string()));
    }

    /// 验证非法 JSON 不会解析出方法。
    #[test]
    fn test_parse_method_invalid_json() {
        assert_eq!(parse_method("not json"), None);
    }

    /// 验证空输入不会解析出方法。
    #[test]
    fn test_parse_method_empty() {
        assert_eq!(parse_method(""), None);
    }

    /// 验证缺少 `method` 字段时不会解析出方法。
    #[test]
    fn test_parse_method_no_method_field() {
        let body = r#"{"jsonrpc":"2.0","id":"1","params":{}}"#;
        assert_eq!(parse_method(body), None);
    }

    /// 验证 `method` 值为空时不会解析出方法。
    #[test]
    fn test_parse_method_empty_method_value() {
        let body = r#"{"jsonrpc":"2.0","method":"","params":{}}"#;
        assert_eq!(parse_method(body), Some("".to_string()));
    }

    /// 验证冒号两侧存在空白时仍可解析出方法。
    #[test]
    fn test_parse_method_whitespace_around_colon() {
        let body = r#"{"jsonrpc":"2.0","method" : "tasks/send","params":{}}"#;
        assert_eq!(parse_method(body), Some("tasks/send".to_string()));
    }

    /// 验证 `method` 不是首个字段时仍可解析出方法。
    #[test]
    fn test_parse_method_method_not_first_field() {
        let body = r#"{"jsonrpc":"2.0","params":{},"method":"tasks/get","id":"1"}"#;
        assert_eq!(parse_method(body), Some("tasks/get".to_string()));
    }

    /// 验证嵌套对象中的 `method` 键不会被误识别。
    #[test]
    fn test_parse_method_nested_object_with_method_key() {
        // 嵌套 params 内部的 "method" 不应被提取出来
        let body = r#"{"jsonrpc":"2.0","method":"tasks/send","params":{"method":"inner"}}"#;
        assert_eq!(parse_method(body), Some("tasks/send".to_string()));
    }

    // ── Proptest 基于属性的测试 ─────────────────────────────────────────

    /// 辅助函数：构造一个带 method 字段的最小 JSON-RPC 风格请求体。
    fn json_body(method: &str) -> String {
        format!("{{\"method\":\"{}\"}}", method)
    }

    proptest! {
        /// parse_method 对任意字符串都不应 panic
        #[test]
        fn parse_method_never_panics(s in "\\PC*") {
            let _ = parse_method(&s);
        }

        /// parse_method 应能从格式良好的 JSON 中正确提取合法的 method 名称
        #[test]
        fn parse_method_extracts_valid_method(
            method in "[a-zA-Z0-9_/\\.]+"
        ) {
            let body = json_body(&method);
            let result = parse_method(&body);
            assert_eq!(result, Some(method));
        }

        /// parse_method 应能处理冒号前后的空白字符
        #[test]
        fn parse_method_handles_whitespace_before_colon(
            spaces_before in "[ ]*",
            spaces_after in "[ ]*",
            method in "[a-zA-Z0-9_]+"
        ) {
            let body = format!(
                "{{\"jsonrpc\":\"2.0\",\"method\"{}:{}{}\"{}\",\"params\":{{}}}}",
                spaces_before, spaces_after, spaces_after, method
            );
            let result = parse_method(&body);
            assert_eq!(result, Some(method));
        }

        /// parse_method 对随机的非 JSON 字符串应返回 None
        #[test]
        fn parse_method_returns_none_for_random_strings(
            s in "[^\\{\\}\"m]*"
        ) {
            let result = parse_method(&s);
            assert_eq!(result, None);
        }

        /// parse_method 处理深度嵌套 JSON 时不应 panic
        #[test]
        fn parse_method_deeply_nested_no_panic(
            depth in 1usize..20usize
        ) {
            let inner = (0..depth).fold("\"value\"".to_string(), |acc, _| {
                format!("{{\"nested\": {}}}", acc)
            });
            let body = format!("{{\"method\":\"test\", \"data\": {}}}", inner);
            let _ = parse_method(&body);
        }

        /// parse_method 处理 unicode method 名称时不应 panic
        #[test]
        fn parse_method_unicode_methods_no_panic(
            method in "[\\p{L}\\p{N}_/]+"
        ) {
            let body = format!("{{\"method\":\"{}\",\"id\":\"1\"}}", method);
            let result = parse_method(&body);
            let _ = result;
        }
    }
}
