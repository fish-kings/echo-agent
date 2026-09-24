/*!
A2A JWT 鉴权中间件。

为 A2A HTTP 端点提供 JWT Bearer Token 校验。

# 用法

```rust,no_run
use echo_agent::a2a::{A2AServer, AgentCard, serve_with_auth, JwtConfig};
use echo_agent::prelude::*;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let card = AgentCard::builder("my-agent", "http://localhost:3000").build();
let agent = ReactAgentBuilder::simple("qwen3-max", "test")?;
let server = A2AServer::new(card, agent);

let jwt_config = JwtConfig::hs256("my-secret-key")
    .with_issuer("echo-agent")
    .with_audience("a2a-clients");

serve_with_auth(server, "0.0.0.0:3000", jwt_config).await?;
# Ok(())
# }
```
*/

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── JWT 配置 ────────────────────────────────────────────────────────────────

/// JWT 鉴权配置。
#[derive(Clone)]
pub struct JwtConfig {
    /// 校验方式（None 表示鉴权被禁用）。
    verification: Option<JwtVerification>,
    /// 签发者（issuer）校验（None 表示不校验）。
    issuer: Option<String>,
    /// 受众（audience）校验（None 表示不校验）。
    audience: Option<String>,
}

/// JWT 签名校验方式。
#[derive(Clone)]
enum JwtVerification {
    /// HS256 对称密钥校验。
    Hs256(DecodingKey),
    /// RS256 非对称公钥校验。
    Rs256(DecodingKey),
}

impl JwtVerification {
    /// 返回该校验方式对应的签名算法。
    fn algorithm(&self) -> Algorithm {
        match self {
            Self::Hs256(_) => Algorithm::HS256,
            Self::Rs256(_) => Algorithm::RS256,
        }
    }

    /// 返回该校验方式使用的解码密钥。
    fn decoding_key(&self) -> &DecodingKey {
        match self {
            Self::Hs256(key) | Self::Rs256(key) => key,
        }
    }
}

/// 无效的 JWT 校验配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwtConfigError {
    /// 错误描述信息。
    message: String,
}

impl std::fmt::Display for JwtConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for JwtConfigError {}

impl JwtConfig {
    /**
    使用 HS256 对称密钥创建配置。

    # 参数

    * `secret` —— 签名密钥（至少 32 字符，推荐 64 字符及以上）
    */
    pub fn hs256(secret: impl Into<String>) -> Self {
        let secret = secret.into();
        Self {
            verification: Some(JwtVerification::Hs256(DecodingKey::from_secret(
                secret.as_bytes(),
            ))),
            issuer: None,
            audience: None,
        }
    }

    /**
    使用 RS256 公钥创建配置。

    # 参数

    * `public_key` —— PEM 格式的 RSA 公钥
    */
    pub fn rs256(public_key: impl Into<String>) -> Result<Self, JwtConfigError> {
        let public_key = public_key.into();
        let decoding_key =
            DecodingKey::from_rsa_pem(public_key.as_bytes()).map_err(|error| JwtConfigError {
                message: format!("invalid RSA public key: {error}"),
            })?;
        Ok(Self {
            verification: Some(JwtVerification::Rs256(decoding_key)),
            issuer: None,
            audience: None,
        })
    }

    /// 设置签发者（issuer）校验。
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = Some(issuer.into());
        self
    }

    /// 设置受众（audience）校验。
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = Some(audience.into());
        self
    }

    /// 禁用鉴权（放行所有请求）。
    pub fn disabled() -> Self {
        Self {
            verification: None,
            issuer: None,
            audience: None,
        }
    }

    /// 鉴权是否已启用。
    pub fn is_enabled(&self) -> bool {
        self.verification.is_some()
    }
}

impl std::fmt::Debug for JwtConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtConfig")
            .field("enabled", &self.is_enabled())
            .field(
                "algorithm",
                &self.verification.as_ref().map(JwtVerification::algorithm),
            )
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("verification_key", &"[redacted]")
            .finish()
    }
}

// ── JWT 声明（Claims）─────────────────────────────────────────────────────────

/// JWT 声明（标准字段 + 自定义扩展）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// 签发者（issuer）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// 主体（sub，通常是用户/客户端 ID）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    /// 受众（audience）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// 过期时间（Unix 时间戳）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<usize>,
    /// 生效起始时间（not before）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbf: Option<usize>,
    /// 签发时间（issued at）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iat: Option<usize>,
    /// 令牌唯一 ID。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
    /// 自定义字段（其余所有 claims）。
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl JwtClaims {
    /// 获取主体（sub）的引用。
    pub fn subject(&self) -> Option<&str> {
        self.sub.as_deref()
    }
}

// ── 中间件 ────────────────────────────────────────────────────────────────

/**
JWT 鉴权中间件。

从 `Authorization: Bearer <token>` 请求头中提取并校验 JWT。校验成功后，
声明（claims）会被注入到请求扩展（extensions）中，下游处理函数可以通过
[`JwtClaims`] 提取它们。

# 错误响应

- 401 `{"error": "missing Authorization header"}` —— 缺少 Authorization 请求头
- 401 `{"error": "invalid token"}` —— 令牌无效或已过期
*/
pub async fn jwt_middleware(
    State(config): State<Arc<JwtConfig>>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    // 鉴权被禁用时，直接放行
    if !config.is_enabled() {
        return next.run(req).await;
    }

    // 提取 Bearer token
    let token = match extract_bearer_token(req.headers()) {
        Some(t) => t,
        None => {
            return unauthorized_response("missing Authorization header");
        }
    };

    // 校验 JWT
    match validate_token(&config, token) {
        Ok(claims) => {
            // 注入声明（claims），供下游处理函数使用
            req.extensions_mut().insert(claims);
            next.run(req).await
        }
        Err(e) => {
            tracing::warn!(error = %e, "JWT validation failed");
            unauthorized_response("invalid token")
        }
    }
}

/// 从请求头中提取 Bearer token。
fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<&str> {
    let header_value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    header_value.strip_prefix("Bearer ")
}

/// 校验 JWT 并返回其中的声明（Claims）。
fn validate_token(config: &JwtConfig, token: &str) -> Result<JwtClaims, String> {
    let verification = config
        .verification
        .as_ref()
        .ok_or_else(|| "JWT authentication is disabled".to_string())?;
    let mut validation = Validation::new(verification.algorithm());

    if let Some(ref issuer) = config.issuer {
        validation.set_issuer(&[issuer.as_str()]);
    }
    if let Some(ref audience) = config.audience {
        validation.set_audience(&[audience.as_str()]);
    }

    // 设置合理的校验选项
    validation.validate_exp = true;
    validation.validate_nbf = true;
    validation.leeway = 30; // 允许 30 秒的时钟偏差

    let token_data = decode::<JwtClaims>(token, verification.decoding_key(), &validation)
        .map_err(|e| format!("JWT validation error: {e}"))?;

    Ok(token_data.claims)
}

/// 401 未授权响应。
fn unauthorized_response(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::json!({"error": message}).to_string(),
    )
        .into_response()
}

// ── 辅助函数 ─────────────────────────────────────────────────────────

/**
从请求扩展（extensions）中提取 JWT 声明（校验成功后被注入）。

# 用法示例

```rust,ignore
async fn my_handler(Extension(claims): Extension<JwtClaims>) -> impl IntoResponse {
    format!("Hello, {}", claims.subject().unwrap_or("unknown"))
}
```
*/
pub fn get_claims<B>(req: &Request<B>) -> Option<&JwtClaims>
where
    B: std::fmt::Debug + Send + Sync + 'static,
{
    req.extensions().get::<JwtClaims>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use jsonwebtoken::{EncodingKey, Header, encode};

    const RSA_PRIVATE_KEY: &[u8] = include_bytes!("../../tests/fixtures/rsa-private.pem");
    const RSA_PUBLIC_KEY: &str = include_str!("../../tests/fixtures/rsa-public.pem");

    fn claims() -> JwtClaims {
        JwtClaims {
            iss: Some("echo-agent".to_string()),
            sub: Some("client-123".to_string()),
            aud: Some("a2a-clients".to_string()),
            exp: Some(4_102_444_800),
            nbf: None,
            iat: None,
            jti: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn test_extract_bearer_token_valid() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer my.jwt.token"),
        );
        assert_eq!(extract_bearer_token(&headers), Some("my.jwt.token"));
    }

    #[test]
    fn test_extract_bearer_token_missing() {
        let headers = axum::http::HeaderMap::new();
        assert_eq!(extract_bearer_token(&headers), None);
    }

    #[test]
    fn test_extract_bearer_token_not_bearer() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert_eq!(extract_bearer_token(&headers), None);
    }

    #[test]
    fn test_jwt_config_disabled() {
        let config = JwtConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_jwt_config_debug_redacts_secret() {
        let config = JwtConfig::hs256("super-secret-key-1234567890");
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.contains("super-secret"));
        assert!(debug_str.contains("[redacted]"));
        assert!(debug_str.contains("HS256"));
    }

    #[test]
    fn hs256_validates_only_hs256_tokens() {
        let config = JwtConfig::hs256("a-test-secret-that-is-long-enough")
            .with_issuer("echo-agent")
            .with_audience("a2a-clients");
        let token_result = encode(
            &Header::new(Algorithm::HS256),
            &claims(),
            &EncodingKey::from_secret(b"a-test-secret-that-is-long-enough"),
        );
        assert!(token_result.is_ok());
        let Some(token) = token_result.ok() else {
            return;
        };
        assert!(validate_token(&config, &token).is_ok());

        let wrong_secret_token = encode(
            &Header::new(Algorithm::HS256),
            &claims(),
            &EncodingKey::from_secret(b"a-different-test-secret-long-enough"),
        );
        assert!(wrong_secret_token.is_ok());
        let Some(wrong_secret_token) = wrong_secret_token.ok() else {
            return;
        };
        assert!(validate_token(&config, &wrong_secret_token).is_err());
    }

    #[test]
    fn rs256_validates_only_matching_rsa_tokens() {
        let config_result = JwtConfig::rs256(RSA_PUBLIC_KEY).map(|config| {
            config
                .with_issuer("echo-agent")
                .with_audience("a2a-clients")
        });
        assert!(config_result.is_ok());
        let Some(config) = config_result.ok() else {
            return;
        };
        let encoding_key_result = EncodingKey::from_rsa_pem(RSA_PRIVATE_KEY);
        assert!(encoding_key_result.is_ok());
        let Some(encoding_key) = encoding_key_result.ok() else {
            return;
        };
        let token_result = encode(&Header::new(Algorithm::RS256), &claims(), &encoding_key);
        assert!(token_result.is_ok());
        let Some(token) = token_result.ok() else {
            return;
        };
        assert!(validate_token(&config, &token).is_ok());

        let hmac_token = encode(
            &Header::new(Algorithm::HS256),
            &claims(),
            &EncodingKey::from_secret(RSA_PUBLIC_KEY.as_bytes()),
        );
        assert!(hmac_token.is_ok());
        let Some(hmac_token) = hmac_token.ok() else {
            return;
        };
        assert!(validate_token(&config, &hmac_token).is_err());
    }

    #[test]
    fn rs256_rejects_invalid_pem_at_configuration_time() {
        assert!(JwtConfig::rs256("not a PEM key").is_err());
    }

    #[test]
    fn test_jwt_claims_subject() {
        let claims = JwtClaims {
            iss: None,
            sub: Some("client-123".to_string()),
            aud: None,
            exp: None,
            nbf: None,
            iat: None,
            jti: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(claims.subject(), Some("client-123"));
    }

    #[test]
    fn test_jwt_claims_no_subject() {
        let claims = JwtClaims {
            iss: None,
            sub: None,
            aud: None,
            exp: None,
            nbf: None,
            iat: None,
            jti: None,
            extra: serde_json::Map::new(),
        };
        assert_eq!(claims.subject(), None);
    }
}
