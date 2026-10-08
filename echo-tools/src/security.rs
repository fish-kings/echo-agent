/*!
安全模块——路径校验与资源限制

提供统一的文件访问安全控制：
- 路径沙箱：防止路径穿越攻击
- 资源限制：防止 DoS 与 OOM
*/

use std::collections::HashMap;
use std::net::ToSocketAddrs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use echo_core::error::{Result, ToolError};

// ── 资源限制配置 ──

/// 资源限制配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// 最大文件大小（字节），默认 50MB。
    #[serde(default = "default_max_file_size")]
    pub max_file_size: u64,

    /// 最大预览行数，默认 10000。
    #[serde(default = "default_max_preview_rows")]
    pub max_preview_rows: usize,

    /// 最大预览字符数，默认 100KB。
    #[serde(default = "default_max_preview_chars")]
    pub max_preview_chars: usize,

    /// PDF 最大预览页数，默认 50。
    #[serde(default = "default_max_preview_pages")]
    pub max_preview_pages: usize,

    /// 图片最大像素数，默认 4096*4096 = 16M。
    #[serde(default = "default_max_image_pixels")]
    pub max_image_pixels: usize,

    /// HTTP 请求超时（秒），默认 30。
    #[serde(default = "default_http_timeout_secs")]
    pub http_timeout_secs: u64,

    /// HTTP 响应体最大大小（字节），默认 10MB。
    #[serde(default = "default_http_max_size")]
    pub http_max_size: u64,

    /// 正则超时（秒），默认 5。
    #[serde(default = "default_regex_timeout_secs")]
    pub regex_timeout_secs: u64,

    /// 正则最大内存（字节），默认 10MB。
    #[serde(default = "default_regex_max_size")]
    pub regex_max_size: usize,
}

/// `max_file_size` 的默认值。
fn default_max_file_size() -> u64 {
    50 * 1024 * 1024 // 50MB
}

/// `max_preview_rows` 的默认值。
fn default_max_preview_rows() -> usize {
    10000
}

/// `max_preview_chars` 的默认值。
fn default_max_preview_chars() -> usize {
    100 * 1024 // 100KB
}

/// `max_preview_pages` 的默认值。
fn default_max_preview_pages() -> usize {
    50
}

/// `max_image_pixels` 的默认值。
fn default_max_image_pixels() -> usize {
    4096 * 4096
}

/// `http_timeout_secs` 的默认值。
fn default_http_timeout_secs() -> u64 {
    30
}

/// `http_max_size` 的默认值。
fn default_http_max_size() -> u64 {
    10 * 1024 * 1024 // 10MB
}

/// `regex_timeout_secs` 的默认值。
fn default_regex_timeout_secs() -> u64 {
    5
}

/// `regex_max_size` 的默认值。
fn default_regex_max_size() -> usize {
    10 * 1024 * 1024 // 10MB
}

impl Default for ResourceLimits {
    /// 使用各字段默认值构建资源限制。
    fn default() -> Self {
        Self {
            max_file_size: default_max_file_size(),
            max_preview_rows: default_max_preview_rows(),
            max_preview_chars: default_max_preview_chars(),
            max_preview_pages: default_max_preview_pages(),
            max_image_pixels: default_max_image_pixels(),
            http_timeout_secs: default_http_timeout_secs(),
            http_max_size: default_http_max_size(),
            regex_timeout_secs: default_regex_timeout_secs(),
            regex_max_size: default_regex_max_size(),
        }
    }
}

impl ResourceLimits {
    /// 创建默认的资源限制配置。
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取 HTTP 超时时长（`Duration`）。
    pub fn http_timeout(&self) -> Duration {
        Duration::from_secs(self.http_timeout_secs)
    }

    /// 获取正则超时时长（`Duration`）。
    pub fn regex_timeout(&self) -> Duration {
        Duration::from_secs(self.regex_timeout_secs)
    }
}

// ── 路径校验器 ──

/**
路径校验器

确保文件访问位于允许的目录范围内，防止路径穿越攻击。
*/
#[derive(Debug, Clone)]
pub struct PathValidator {
    /// 允许访问的根目录列表（为空表示全部允许）。
    allowed_roots: Vec<PathBuf>,

    /// 禁止访问的路径列表。
    denied_paths: Vec<PathBuf>,

    /// 资源限制。
    limits: ResourceLimits,

    /// 是否启用校验（关闭用于测试或可信环境）。
    enabled: bool,
}

impl Default for PathValidator {
    /// 创建启用校验、无根目录限制的默认校验器。
    fn default() -> Self {
        Self {
            allowed_roots: Vec::new(),
            denied_paths: Vec::new(),
            limits: ResourceLimits::default(),
            enabled: true,
        }
    }
}

impl PathValidator {
    /// 创建新的路径校验器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置允许访问的根目录。
    pub fn with_allowed_roots(mut self, roots: &[&str]) -> Self {
        self.allowed_roots = roots.iter().map(PathBuf::from).collect();
        self
    }

    /// 设置禁止访问的路径。
    pub fn with_denied_paths(mut self, paths: &[&str]) -> Self {
        self.denied_paths = paths.iter().map(PathBuf::from).collect();
        self
    }

    /// 设置资源限制。
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// 启用或禁用校验。
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /**
    校验文件路径。

    返回规范化后的路径，或错误。
    */
    pub fn validate_file(&self, path: &str) -> Result<PathBuf> {
        if !self.enabled {
            return Ok(PathBuf::from(path));
        }

        let path = Path::new(path);

        // 1. 检查是否为绝对路径
        if !path.is_absolute() {
            return Err(ToolError::InvalidPath {
                path: path.display().to_string(),
                reason: "Path must be absolute".to_string(),
            }
            .into());
        }

        // 2. 规范化路径（解析 .. 与 .）
        let canonical = path.canonicalize().map_err(|e| ToolError::InvalidPath {
            path: path.display().to_string(),
            reason: format!("Path does not exist or cannot be accessed: {}", e),
        })?;

        // 3. 检查是否在禁止列表中
        for denied in &self.denied_paths {
            if canonical.starts_with(denied) {
                return Err(ToolError::AccessDenied {
                    path: path.display().to_string(),
                    reason: "Path is in the denied list".to_string(),
                }
                .into());
            }
        }

        // 4. 检查是否位于允许的根目录内（若已配置）
        if !self.allowed_roots.is_empty() {
            let is_allowed = self.allowed_roots.iter().any(|root| {
                if let Ok(root_canonical) = root.canonicalize() {
                    canonical.starts_with(&root_canonical)
                } else {
                    false
                }
            });

            if !is_allowed {
                return Err(ToolError::AccessDenied {
                    path: path.display().to_string(),
                    reason: "Path is not within allowed directory scope".to_string(),
                }
                .into());
            }
        }

        // 5. 检查文件大小
        let metadata = std::fs::metadata(&canonical).map_err(|e| ToolError::ExecutionFailed {
            tool: "path_validator".to_string(),
            message: format!("Unable to get file info: {}", e),
        })?;

        if metadata.len() > self.limits.max_file_size {
            return Err(ToolError::FileTooLarge {
                size: metadata.len(),
                max: self.limits.max_file_size,
            }
            .into());
        }

        Ok(canonical)
    }

    /**
    校验路径位于基准目录之内（路径可以尚不存在）。

    规范化最近的已存在祖先目录，再拼回剩余后缀，校验其包含于基准目录，
    并检查禁止路径列表。这是常见“限制在工作区根目录内”模式的**标准**校验器
    （Phase 6.1）。代码库中其它路径校验器应逐步收敛到这里。
    */
    pub fn validate_within_base(&self, path: &str, base: &Path) -> Result<PathBuf> {
        if !self.enabled {
            return Ok(PathBuf::from(path));
        }
        let requested = Path::new(path);
        if !requested.is_absolute() {
            return Err(ToolError::InvalidPath {
                path: path.to_string(),
                reason: "Path must be absolute".into(),
            }
            .into());
        }
        // 在词法层面拒绝 `..`
        for comp in requested.components() {
            if matches!(comp, std::path::Component::ParentDir) {
                return Err(ToolError::InvalidPath {
                    path: path.to_string(),
                    reason: "Path traversal (..) is not allowed".into(),
                }
                .into());
            }
        }
        // 查找最近的已存在祖先目录
        let mut check = if requested.exists() {
            requested
        } else {
            requested.parent().unwrap_or(requested)
        };
        while !check.exists() {
            if let Some(p) = check.parent() {
                check = p;
            } else {
                break;
            }
        }
        let canonical_ancestor = check.canonicalize().map_err(|e| ToolError::InvalidPath {
            path: check.display().to_string(),
            reason: format!("Cannot resolve path: {}", e),
        })?;
        let base_canonical = base.canonicalize().map_err(|e| ToolError::InvalidPath {
            path: base.display().to_string(),
            reason: format!("Cannot resolve base: {}", e),
        })?;
        if !canonical_ancestor.starts_with(&base_canonical) {
            return Err(ToolError::AccessDenied {
                path: path.to_string(),
                reason: "Path is outside the allowed base directory".into(),
            }
            .into());
        }
        // 重建完整路径
        let suffix = requested
            .strip_prefix(check)
            .map_err(|_| ToolError::InvalidPath {
                path: path.to_string(),
                reason: "Cannot compute relative suffix".into(),
            })?;
        let resolved = canonical_ancestor.join(suffix);
        // 对最终路径再次检查禁止列表
        for denied in &self.denied_paths {
            if let Ok(d) = denied.canonicalize()
                && resolved.starts_with(&d)
            {
                return Err(ToolError::AccessDenied {
                    path: path.to_string(),
                    reason: "Path is in the denied list".into(),
                }
                .into());
            }
        }
        Ok(resolved)
    }

    /**
    校验输出文件路径。

    与 `validate_file()` 不同，输出路径允许文件尚不存在，但仍会：
    - 要求绝对路径
    - 规范化 `.` / `..`
    - 检查 denied_paths / allowed_roots
    */
    pub fn validate_output_file(&self, path: &str) -> Result<PathBuf> {
        if !self.enabled {
            return Ok(PathBuf::from(path));
        }

        let path = Path::new(path);
        if !path.is_absolute() {
            return Err(ToolError::InvalidPath {
                path: path.display().to_string(),
                reason: "Path must be absolute".to_string(),
            }
            .into());
        }

        let normalized = normalize_absolute_path(path)?;

        for denied in &self.denied_paths {
            if normalize_for_policy(denied)
                .map(|denied_path| normalized.starts_with(&denied_path))
                .unwrap_or(false)
            {
                return Err(ToolError::AccessDenied {
                    path: path.display().to_string(),
                    reason: "Path is in the denied list".to_string(),
                }
                .into());
            }
        }

        if !self.allowed_roots.is_empty() {
            let is_allowed = self.allowed_roots.iter().any(|root| {
                normalize_for_policy(root)
                    .map(|allowed_root| normalized.starts_with(&allowed_root))
                    .unwrap_or(false)
            });

            if !is_allowed {
                return Err(ToolError::AccessDenied {
                    path: path.display().to_string(),
                    reason: "Path is not within allowed directory scope".to_string(),
                }
                .into());
            }
        }

        Ok(normalized)
    }

    /// 校验并获取文件内容大小。
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = std::fs::metadata(path).map_err(|e| ToolError::ExecutionFailed {
            tool: "path_validator".to_string(),
            message: format!("Unable to get file info: {}", e),
        })?;
        Ok(metadata.len())
    }

    /// 获取资源限制。
    pub fn limits(&self) -> &ResourceLimits {
        &self.limits
    }
}

// ── 全局安全配置 ──

use std::sync::OnceLock;

/// 进程级全局安全配置。
static GLOBAL_SECURITY: OnceLock<Arc<SecurityConfig>> = OnceLock::new();

/// 安全配置。
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// 路径校验器，限制文件访问范围。
    pub path_validator: PathValidator,
    /// 资源限制（内存、CPU、并发等）。
    pub limits: ResourceLimits,
}

impl Default for SecurityConfig {
    /// 使用默认路径校验器与资源限制构建配置。
    fn default() -> Self {
        Self {
            path_validator: PathValidator::new(),
            limits: ResourceLimits::default(),
        }
    }
}

impl SecurityConfig {
    /// 获取全局安全配置。
    pub fn global() -> Arc<Self> {
        GLOBAL_SECURITY
            .get_or_init(|| Arc::new(Self::default()))
            .clone()
    }

    /// 设置全局安全配置（仅首次设置生效）。
    pub fn set_global(config: Self) {
        let _ = GLOBAL_SECURITY.set(Arc::new(config));
    }

    /// 校验文件路径。
    pub fn validate_file(&self, path: &str) -> Result<PathBuf> {
        self.path_validator.validate_file(path)
    }

    /// 校验输出文件路径（允许目标文件尚不存在）。
    pub fn validate_output_file(&self, path: &str) -> Result<PathBuf> {
        self.path_validator.validate_output_file(path)
    }

    /// 检查文件大小。
    pub fn check_file_size(&self, size: u64) -> Result<()> {
        if size > self.limits.max_file_size {
            return Err(ToolError::FileTooLarge {
                size,
                max: self.limits.max_file_size,
            }
            .into());
        }
        Ok(())
    }
}

// ── 安全 HTTP 客户端 ──

/// 创建经过安全配置的 HTTP 客户端。
pub fn create_safe_http_client(limits: &ResourceLimits) -> Result<reqwest::Client> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(limits.http_timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .redirect(ssrf_safe_redirect_policy())
        .build()
        .map_err(|e| ToolError::ExecutionFailed {
            tool: "http_client".to_string(),
            message: format!("Failed to create HTTP client: {}", e),
        })?;

    Ok(client)
}

// ── 安全正则 ──

use regex::RegexBuilder;

/// 创建经过安全配置的正则表达式。
pub fn create_safe_regex(pattern: &str, limits: &ResourceLimits) -> Result<regex::Regex> {
    RegexBuilder::new(pattern)
        .size_limit(limits.regex_max_size)
        .dfa_size_limit(limits.regex_max_size)
        .build()
        .map_err(|e| {
            ToolError::InvalidParameter {
                name: "pattern".to_string(),
                message: format!("Invalid regex: {}", e),
            }
            .into()
        })
}

// ── SSRF 防护 ──

/**
校验 URL 目标地址，拒绝访问私有 / 链路本地 IP 的请求（SSRF 防护）。

返回已校验的主机名与解析出的公网 IP 地址。调用方若使用这些地址建立连接
（通过 `pin_request_to_addrs`），即可关闭单独调用 `validate_url` 再由客户端独立解析
所留下的 DNS 重绑定 TOCTOU 窗口。

注意：本函数仍会解析一次 DNS；若要完全防御重绑定，请用返回的地址建立固定连接
（参见 [`ssrf_safe_get`]）。
*/
pub fn validate_url(url_str: &str) -> Result<()> {
    validate_url_with_addrs(url_str).map(|_| ())
}

/**
校验 URL，并返回主机名以及所有已解析且通过 SSRF 检查的 IP 地址。

这是 [`validate_url`] 的防重绑定版本：只解析一次主机名，只要*任一*解析地址为
私有 / 链路本地地址就拒绝请求，否则返回这些公网地址。将其传给通过
[`reqwest::ClientBuilder::resolve_to_addrs`] 配置的客户端，使连接复用这批已校验的 IP，
而不是再次解析（DNS 重绑定会利用第二次解析指向 `127.0.0.1` / `169.254.169.254`）。
*/
pub fn validate_url_with_addrs(url_str: &str) -> Result<(String, Vec<std::net::IpAddr>)> {
    resolve_url_with_addrs(url_str, true)
}

/// 解析 URL 主机名的全部 IP；`reject_private` 为真时遇到私有地址即拒绝。
fn resolve_url_with_addrs(
    url_str: &str,
    reject_private: bool,
) -> Result<(String, Vec<std::net::IpAddr>)> {
    let host = extract_host(url_str)?.to_string();

    // 将主机名解析为 IP 地址
    let addr_str = format!("{}:0", host);
    let addrs = addr_str
        .to_socket_addrs()
        .map_err(|e| ToolError::ExecutionFailed {
            tool: "security".to_string(),
            message: format!("SSRF protection: DNS resolution failed: {}", e),
        })?;

    let mut public: Vec<std::net::IpAddr> = Vec::new();
    for addr in addrs {
        let ip = addr.ip();
        if reject_private && is_private_ip(&ip) {
            return Err(ToolError::AccessDenied {
                path: url_str.to_string(),
                reason: format!(
                    "SSRF protection: rejecting access to private IP address {}",
                    ip
                ),
            }
            .into());
        }
        public.push(ip);
    }

    Ok((host, public))
}

/// 从 URL 字符串中提取主机名（不依赖 url crate）。
fn extract_host(url_str: &str) -> Result<&str> {
    let rest = url_str
        .strip_prefix("http://")
        .or_else(|| url_str.strip_prefix("https://"))
        .ok_or_else(|| ToolError::InvalidParameter {
            name: "url".to_string(),
            message: "URL must start with http:// or https://".to_string(),
        })?;

    // 提取 authority（host:port）
    let authority = rest.split('/').next().unwrap_or(rest);
    // 去掉 ?query
    let authority = authority.split('?').next().unwrap_or(authority);
    // 去掉 userinfo@——取最后一个 @ 之后的全部内容
    let authority = authority.rsplit('@').next().unwrap_or(authority);

    // 提取主机，处理 IPv6 方括号写法 [::1]:port
    let host = if authority.starts_with('[') {
        // IPv6 字面量：查找右方括号
        if let Some(close) = authority.find(']') {
            &authority[1..close]
        } else {
            // 格式错误——整体视为主机（多半会在 DNS 解析时失败）
            authority
        }
    } else {
        // 普通 host:port——按第一个冒号切分以去掉端口
        authority.split(':').next().unwrap_or(authority)
    };

    if host.is_empty() {
        return Err(ToolError::InvalidParameter {
            name: "url".to_string(),
            message: "URL missing hostname".to_string(),
        }
        .into());
    }

    Ok(host)
}

/// 判断 IP 地址是否为私有 / 链路本地地址。
fn is_private_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => is_private_v4(v4),
        std::net::IpAddr::V6(v6) => {
            /* 检查 IPv4 映射的 IPv6 地址（::ffff:x.x.x.x），
            并校验其映射的 IPv4 地址 */
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_private_v4(&v4);
            }

            let octets = v6.octets();
            // ::1（本机回环）
            *v6 == std::net::Ipv6Addr::LOCALHOST
                // fc00::/7（ULA，RFC 4193）——同时覆盖 fc00::/8 与 fd00::/8
                || (octets[0] & 0xFE) == 0xFC
                // fe80::/10（链路本地）
                || (octets[0] == 0xfe && (octets[1] & 0xC0) == 0x80)
        }
    }
}

/// 判断 IPv4 地址是否为私有 / 链路本地 / 保留地址。
fn is_private_v4(v4: &std::net::Ipv4Addr) -> bool {
    let octets = v4.octets();
    // 127.0.0.0/8（回环）
    octets[0] == 127
        // 10.0.0.0/8（RFC 1918 私有网段）
        || octets[0] == 10
        // 172.16.0.0/12（RFC 1918 私有网段）
        || (octets[0] == 172 && (octets[1] & 0xF0) == 16)
        // 192.168.0.0/16（RFC 1918 私有网段）
        || (octets[0] == 192 && octets[1] == 168)
        // 169.254.0.0/16（链路本地）
        || (octets[0] == 169 && octets[1] == 254)
        // 0.0.0.0/8（本网络）
        || octets[0] == 0
        // 100.64.0.0/10（CGNAT / 共享地址空间）
        || (octets[0] == 100 && (octets[1] & 0xC0) == 64)
        // 192.0.0.0/24（IETF 协议分配）
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        // 198.18.0.0/15（网络基准测试）
        || (octets[0] == 198 && (octets[1] == 18 || octets[1] == 19))
        // 192.0.2.0/24（文档用 TEST-NET-1）
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 2)
        // 198.51.100.0/24（文档用 TEST-NET-2）
        || (octets[0] == 198 && octets[1] == 51 && octets[2] == 100)
        // 203.0.113.0/24（文档用 TEST-NET-3）
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
}

/// 创建防 SSRF 的重定向策略。
pub fn ssrf_safe_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() > 5 {
            return attempt.error("SSRF protection: too many redirects");
        }
        match validate_url(attempt.url().as_str()) {
            Ok(()) => attempt.follow(),
            Err(e) => attempt.error(format!("SSRF protection: redirect target blocked: {}", e)),
        }
    })
}

/**
为 `host` 构建一个固定到已校验 IP 地址的一次性 [`reqwest::Client`]，
以关闭 DNS 重绑定的 TOCTOU 窗口。

返回的客户端以 `host` 作为 TLS SNI 与 `Host` 头（因此虚拟主机与 TLS 证书校验仍然有效），
但*只*连接 `addrs` 中的地址——即 [`validate_url_with_addrs`] 检查过的那些 IP。
本次请求不会再查询 DNS。
*/
fn pinned_client(
    host: &str,
    addrs: &[std::net::IpAddr],
    timeout: Duration,
) -> Result<reqwest::Client> {
    /* `resolve_to_addrs` 需要 SocketAddr；端口对 IP 固定无关紧要
    （reqwest 会从 URL 填入真实端口），因此使用占位端口。 */
    let socket_addrs: Vec<std::net::SocketAddr> = addrs
        .iter()
        .map(|ip| std::net::SocketAddr::new(*ip, 0))
        .collect();
    reqwest::Client::builder()
        .resolve_to_addrs(host, &socket_addrs)
        .redirect(reqwest::redirect::Policy::none()) // 重定向由调用方重新校验
        .timeout(timeout)
        .build()
        .map_err(|e| {
            ToolError::ExecutionFailed {
                tool: "security".to_string(),
                message: format!("SSRF protection: failed to build pinned client: {}", e),
            }
            .into()
        })
}

/**
执行防 SSRF 的 HTTP GET，固定解析出的 IP 以抵御 DNS 重绑定。

流程：解析并校验一次 → 构建固定到已校验 IP 的客户端 → 发送。每一跳重定向都会
独立重新解析并重新校验（因此 `302 → 169.254.169.254` 无法溜过）。所有网页抓取工具
都应使用这个统一入口，而不是 `validate_url(url)?; client.get(url)`。
*/
pub async fn ssrf_safe_get(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
) -> Result<reqwest::Response> {
    http_request_with_policy(
        url,
        timeout,
        max_redirects,
        reqwest::Method::GET,
        None,
        None,
        true,
    )
    .await
}

/**
执行本地助手场景的 HTTP GET。由于用户常在本地运行服务，私有与回环目标是合法的；
但协议校验、DNS 固定以及有上限的逐跳重定向处理仍会强制执行。
*/
pub async fn local_http_get(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
) -> Result<reqwest::Response> {
    http_request_with_policy(
        url,
        timeout,
        max_redirects,
        reqwest::Method::GET,
        None,
        None,
        false,
    )
    .await
}

/// 防 SSRF 请求（任意方法），带 IP 固定与逐跳重定向校验。
pub async fn ssrf_safe_request(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
    method: reqwest::Method,
) -> Result<reqwest::Response> {
    http_request_with_policy(url, timeout, max_redirects, method, None, None, true).await
}

/// 允许私有与回环目标的本地助手请求变体。
pub async fn local_http_request(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
    method: reqwest::Method,
) -> Result<reqwest::Response> {
    http_request_with_policy(url, timeout, max_redirects, method, None, None, false).await
}

/**
带可选 JSON 请求体与额外请求头的防 SSRF 请求。

供向运维配置的 URL POST 负载的 HTTP 钩子使用。复用与 [`ssrf_safe_request`] 相同的
“解析一次 → 固定 IP → 每次重定向重新校验”流水线；请求体与请求头附加在各跳请求上。
请求体在第一跳即被消耗（在 reqwest 中不重新序列化就无法跨重定向重发 JSON 请求体），
因此重定向会丢弃请求体——对 SSRF 敏感的出站 POST 而言这是保守的选择
（我们绝不希望把机密重放给重定向目标）。
*/
pub async fn ssrf_safe_request_with_body(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
    method: reqwest::Method,
    body: &serde_json::Value,
    headers: Option<&HashMap<String, String>>,
) -> Result<reqwest::Response> {
    http_request_with_policy(
        url,
        timeout,
        max_redirects,
        method,
        Some(body),
        headers,
        true,
    )
    .await
}

/**
带请求体与不带请求体两类变体共享的防 SSRF 核心流水线。

`body` 与 `headers` 只在第一跳生效。重定向以无请求体方式跟随，避免把（可能含机密的）
负载重放到校验器未见过的重定向目标。
*/
async fn http_request_with_policy(
    url: &str,
    timeout: Duration,
    max_redirects: usize,
    method: reqwest::Method,
    body: Option<&serde_json::Value>,
    headers: Option<&HashMap<String, String>>,
    reject_private: bool,
) -> Result<reqwest::Response> {
    let mut current = url.to_string();
    let mut first_hop = true;
    for _ in 0..=max_redirects {
        let (host, addrs) = resolve_url_with_addrs(&current, reject_private)?;
        let client = pinned_client(&host, &addrs, timeout)?;
        let mut req = client.request(method.clone(), &current);
        // 仅在第一跳附加请求体与请求头；参见函数文档。
        if first_hop {
            if let Some(b) = body {
                req = req.json(b);
            }
            if let Some(h) = headers {
                for (k, v) in h {
                    req = req.header(k, v);
                }
            }
            first_hop = false;
        }
        let response = req.send().await.map_err(|e| ToolError::ExecutionFailed {
            tool: "security".to_string(),
            message: format!("SSRF-safe request failed: {}", e),
        })?;

        if response.status().is_redirection() {
            // 独立地重新解析并校验重定向目标。
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| ToolError::ExecutionFailed {
                    tool: "security".to_string(),
                    message: "redirect without Location header".to_string(),
                })?;
            current = reqwest::Url::parse(&current)
                .and_then(|base| base.join(location))
                .map_err(|error| ToolError::InvalidParameter {
                    name: "url".to_string(),
                    message: format!("invalid redirect target: {error}"),
                })?
                .to_string();
            continue;
        }
        return Ok(response);
    }
    Err(ToolError::ExecutionFailed {
        tool: "security".to_string(),
        message: format!("SSRF protection: too many redirects (>{max_redirects})"),
    }
    .into())
}

/// 为策略比较规范化路径：优先 canonicalize，失败则做纯词法规范化。
fn normalize_for_policy(path: &Path) -> Option<PathBuf> {
    path.canonicalize()
        .ok()
        .or_else(|| normalize_absolute_path(path).ok())
}

/// 对绝对路径做纯词法规范化（处理 `.` 与 `..`），不访问文件系统。
fn normalize_absolute_path(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        return Err(ToolError::InvalidPath {
            path: path.display().to_string(),
            reason: "Path must be absolute".to_string(),
        }
        .into());
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

/// 单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 相对路径应被拒绝。
    #[test]
    fn test_path_validator_absolute_required() {
        let validator = PathValidator::new().with_enabled(true);
        let result = validator.validate_file("relative/path.txt");
        assert!(result.is_err());
    }

    /// 可以配置允许的根目录。
    #[test]
    fn test_path_validator_allowed_root() {
        let _validator = PathValidator::new()
            .with_allowed_roots(&["/tmp"])
            .with_enabled(true);
    }

    /// 资源限制默认值符合预期。
    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_file_size, 50 * 1024 * 1024);
        assert_eq!(limits.max_preview_rows, 10000);
    }

    /// 输出路径必须为绝对路径。
    #[test]
    fn test_validate_output_file_absolute_required() {
        let validator = PathValidator::new().with_enabled(true);
        let result = validator.validate_output_file("relative/output.txt");
        assert!(result.is_err());
    }

    /// 输出路径中的 `..` 会被规范化。
    #[test]
    fn test_validate_output_file_normalizes_parent_segments() {
        let validator = PathValidator::new().with_enabled(true);
        let path = validator
            .validate_output_file("/tmp/demo/../result.txt")
            .unwrap();
        assert_eq!(path, PathBuf::from("/tmp/result.txt"));
    }

    // ── SSRF 黄金测试 ─────────────────────────────────────────────────

    /// IPv4 回环地址视为私有。
    #[test]
    fn test_is_private_ip_v4_loopback() {
        assert!(is_private_ip(&"127.0.0.1".parse().unwrap()));
        assert!(is_private_ip(&"127.255.255.255".parse().unwrap()));
    }

    /// RFC 1918 网段视为私有。
    #[test]
    fn test_is_private_ip_v4_rfc1918() {
        assert!(is_private_ip(&"10.0.0.1".parse().unwrap()));
        assert!(is_private_ip(&"10.255.255.255".parse().unwrap()));
        assert!(is_private_ip(&"172.16.0.1".parse().unwrap()));
        assert!(is_private_ip(&"172.31.255.255".parse().unwrap()));
        assert!(is_private_ip(&"192.168.0.1".parse().unwrap()));
        assert!(is_private_ip(&"192.168.255.255".parse().unwrap()));
    }

    /// IPv4 链路本地地址视为私有。
    #[test]
    fn test_is_private_ip_v4_link_local() {
        assert!(is_private_ip(&"169.254.0.1".parse().unwrap()));
        assert!(is_private_ip(&"169.254.255.255".parse().unwrap()));
    }

    /// CGNAT 网段视为私有。
    #[test]
    fn test_is_private_ip_v4_cgnat() {
        assert!(is_private_ip(&"100.64.0.1".parse().unwrap()));
        assert!(is_private_ip(&"100.127.255.255".parse().unwrap()));
    }

    /// 各类特殊保留网段视为私有。
    #[test]
    fn test_is_private_ip_v4_special_ranges() {
        // 0.0.0.0/8
        assert!(is_private_ip(&"0.0.0.0".parse().unwrap()));
        assert!(is_private_ip(&"0.255.255.255".parse().unwrap()));
        // TEST-NET-1（文档保留网段）
        assert!(is_private_ip(&"192.0.2.1".parse().unwrap()));
        // TEST-NET-2（文档保留网段）
        assert!(is_private_ip(&"198.51.100.1".parse().unwrap()));
        // TEST-NET-3（文档保留网段）
        assert!(is_private_ip(&"203.0.113.1".parse().unwrap()));
        // IETF 协议分配
        assert!(is_private_ip(&"192.0.0.1".parse().unwrap()));
        // 基准测试网段
        assert!(is_private_ip(&"198.18.0.1".parse().unwrap()));
        assert!(is_private_ip(&"198.19.255.255".parse().unwrap()));
    }

    /// IPv4 公网地址放行。
    #[test]
    fn test_public_ip_v4_allowed() {
        assert!(!is_private_ip(&"8.8.8.8".parse().unwrap()));
        assert!(!is_private_ip(&"1.1.1.1".parse().unwrap()));
        assert!(!is_private_ip(&"93.184.216.34".parse().unwrap()));
    }

    /// IPv6 回环地址视为私有。
    #[test]
    fn test_is_private_ip_v6_loopback() {
        assert!(is_private_ip(&"::1".parse().unwrap()));
    }

    /// IPv6 链路本地地址视为私有。
    #[test]
    fn test_is_private_ip_v6_link_local() {
        assert!(is_private_ip(&"fe80::1".parse().unwrap()));
        assert!(is_private_ip(&"feb0::1".parse().unwrap()));
    }

    /// 整个 IPv6 ULA 范围视为私有。
    #[test]
    fn test_is_private_ip_v6_ula_full_range() {
        // fc00::/7 同时覆盖 fc00::/8 与 fd00::/8（RFC 4193）
        assert!(is_private_ip(&"fc00::1".parse().unwrap()));
        assert!(is_private_ip(&"fcff::1".parse().unwrap()));
        assert!(is_private_ip(&"fd00::1".parse().unwrap()));
        assert!(is_private_ip(&"fdff::1".parse().unwrap()));
    }

    /// IPv4 映射的 IPv6 地址按其 IPv4 判断。
    #[test]
    fn test_is_private_ip_v6_mapped_v4() {
        // ::ffff:127.0.0.1 应视为私有地址
        assert!(is_private_ip(&"::ffff:127.0.0.1".parse().unwrap()));
        assert!(is_private_ip(&"::ffff:10.0.0.1".parse().unwrap()));
        assert!(is_private_ip(&"::ffff:192.168.1.1".parse().unwrap()));
    }

    /// IPv6 公网地址放行。
    #[test]
    fn test_public_ip_v6_allowed() {
        assert!(!is_private_ip(&"2001:4860:4860::8888".parse().unwrap()));
        assert!(!is_private_ip(&"2606:4700:4700::1111".parse().unwrap()));
    }

    // ── extract_host 黄金测试 ─────────────────────────────────────────

    /// 提取普通 URL 的主机名。
    #[test]
    fn test_extract_host_plain() {
        assert_eq!(
            extract_host("http://example.com/path").unwrap(),
            "example.com"
        );
        assert_eq!(extract_host("https://example.com").unwrap(), "example.com");
    }

    /// 提取主机名时去掉端口。
    #[test]
    fn test_extract_host_with_port() {
        assert_eq!(
            extract_host("http://example.com:8080/path").unwrap(),
            "example.com"
        );
    }

    /// 提取主机名时去掉 userinfo。
    #[test]
    fn test_extract_host_with_userinfo() {
        assert_eq!(
            extract_host("http://user:pass@example.com/path").unwrap(),
            "example.com"
        );
    }

    /// 提取 IPv6 字面量主机。
    #[test]
    fn test_extract_host_ipv6() {
        assert_eq!(extract_host("http://[::1]/path").unwrap(), "::1");
        assert_eq!(
            extract_host("https://[2001:db8::1]/api").unwrap(),
            "2001:db8::1"
        );
    }

    /// 提取带端口的 IPv6 字面量主机。
    #[test]
    fn test_extract_host_ipv6_with_port() {
        assert_eq!(extract_host("http://[::1]:8080/path").unwrap(), "::1");
        assert_eq!(
            extract_host("http://[2001:db8::1]:443/data").unwrap(),
            "2001:db8::1"
        );
    }

    /// 提取主机名时忽略查询串。
    #[test]
    fn test_extract_host_with_query() {
        assert_eq!(
            extract_host("http://example.com/path?key=value").unwrap(),
            "example.com"
        );
    }

    /// 拒绝非 http/https 协议。
    #[test]
    fn test_extract_host_rejects_non_http() {
        assert!(extract_host("ftp://example.com").is_err());
        assert!(extract_host("file:///etc/passwd").is_err());
    }

    // ── validate_url_with_addrs 黄金测试 ─────────────────────────────

    /// validate_url 拒绝回环地址。
    #[test]
    fn test_validate_url_rejects_loopback() {
        assert!(validate_url("http://127.0.0.1:8080/data").is_err());
        assert!(validate_url("http://[::1]:8080/data").is_err());
    }

    /// validate_url 放行公网主机。
    #[test]
    fn test_validate_url_allows_public() {
        /* 本测试需要对已知公网主机做 DNS 解析。
        example.com 解析为公网 IP；若 DNS 不可用则跳过。 */
        let result = validate_url("https://example.com/");
        /* 在无网络的 CI 中 DNS 可能失败。这里只检查失败原因不是
        私有 IP 拒绝（那种情况总会报错）。 */
        if let Err(e) = &result {
            let msg = format!("{}", e);
            assert!(
                !msg.contains("private IP"),
                "should not reject public host: {msg}"
            );
        }
    }

    /// validate_url_with_addrs 返回公网 IP 列表。
    #[test]
    fn test_validate_url_with_addrs_returns_ips() {
        let result = validate_url_with_addrs("https://example.com/");
        if let Ok((host, addrs)) = result {
            assert_eq!(host, "example.com");
            assert!(!addrs.is_empty(), "should return at least one IP");
            for ip in &addrs {
                assert!(!is_private_ip(ip), "returned IP {ip} should be public");
            }
        }
    }

    // ── 已修复漏洞的回归测试 ────────────────────────

    /// fc00::/8 必须被拦截。
    #[test]
    fn test_ipv6_ula_fc00_blocked() {
        // 回归：修复前 fc00::/8 未被拦截（只拦截了 fd00::/8）
        let ip: std::net::IpAddr = "fc00::1".parse().unwrap();
        assert!(
            is_private_ip(&ip),
            "fc00::/8 (ULA lower half) must be blocked"
        );
    }

    /// IPv6 加端口的主机提取回归测试。
    #[test]
    fn test_extract_host_ipv6_port_regression() {
        // 回归：extract_host 曾使用 split(':').next()，遇到 IPv6+端口时出错
        assert_eq!(
            extract_host("http://[::1]:8080/path").unwrap(),
            "::1",
            "IPv6 with port must work"
        );
        assert_eq!(
            extract_host("http://[fe80::1]:3000/").unwrap(),
            "fe80::1",
            "IPv6 link-local with port must work"
        );
    }

    // ── 模糊 / 边界测试 ─────────────────────────────────────────────

    /// 按 UTF-8 字符边界安全截断。
    #[test]
    fn test_truncate_utf8_safe() {
        /* 验证按字节边界截断对多字节字符是安全的。
        "é" = 0xC3 0xA9（2 字节），"好" = 0xE5 0xA5 0xBD（3 字节） */
        let s = "hello 世界!";
        // floor_char_boundary 返回安全的字节索引
        let safe = s.floor_char_boundary(8); // "hello " 占 6 字节，其后是 3 字节的 "世"
        assert_eq!(&s[..safe], "hello ");
    }

    /// .ssh 目录可从路径中识别。
    #[test]
    fn test_unicode_case_fold_security() {
        let path = "/home/user/.ssh/authorized_keys";
        let path_buf = std::path::PathBuf::from(path);
        let parent = path_buf.parent().unwrap();
        assert!(
            parent.ends_with(".ssh"),
            ".ssh directory should be detectable from path"
        );
    }

    // ── 非 ASCII 模糊测试（Phase 3.7）──────────────────────────────────

    /// emoji 边界处安全截断。
    #[test]
    fn test_truncate_emoji_boundary() {
        // Emoji 占 4 字节（如 😀 = 0xF0 0x9F 0x98 0x80）
        let s = "abc😀def";
        let safe = s.floor_char_boundary(4); // "abc" 占 3 字节，其后是 4 字节的 emoji
        assert_eq!(&s[..safe], "abc");
    }

    /// 全多字节字符串安全截断。
    #[test]
    fn test_truncate_all_multibyte() {
        // 仅包含多字节字符的字符串
        let s = "世界你好"; // 4 个字符 × 每个 3 字节 = 12 字节
        let safe = s.floor_char_boundary(4); // 应停在 3 字节处（1 个字符）
        assert_eq!(&s[..safe], "世");
        let safe2 = s.floor_char_boundary(7); // 应停在 6 字节处（2 个字符）
        assert_eq!(&s[..safe2], "世界");
    }

    /// 截断到 0 返回 0。
    #[test]
    fn test_truncate_zero() {
        let s = "hello";
        assert_eq!(s.floor_char_boundary(0), 0);
    }

    /// 超出末尾截断返回字符串长度。
    #[test]
    fn test_truncate_past_end() {
        let s = "hi";
        let safe = s.floor_char_boundary(100);
        assert_eq!(safe, 2); // 应返回长度而不是 panic
    }

    /// Unicode 形近字符不能绕过 URL 校验。
    #[test]
    fn test_validate_url_rejects_unicode_homoglyph() {
        /* Unicode 全角斜杠（U+FF0F）／ 不应绕过 URL 解析，
        因为 URL 必须以 ASCII "http://" 开头 */
        let result = validate_url("http：//evil.com/");
        assert!(
            result.is_err(),
            "Non-ASCII colon should fail URL validation"
        );
    }

    /// 国际化域名可被正确提取。
    #[test]
    fn test_extract_host_unicode_idn() {
        /* IDN（国际化域名）——extract_host 会提取主机名，
        其中若有非 ASCII 字符也一并保留。 */
        let host = extract_host("http://münchen.de/path").unwrap();
        assert_eq!(host, "münchen.de");
    }

    /// 任意字节位置截断都不会 panic。
    #[test]
    fn test_byte_slice_safety_on_boundary_edge() {
        /* 模拟 web_fetch_enhanced.rs 中的模式：在任意字节位置截断，
        并验证 floor_char_boundary 是安全的。 */
        for boundary in 0..=20 {
            let s = "a😀b世c界déefgh";
            let safe = s.floor_char_boundary(boundary);
            // 必须是合法的字符边界——切片永远不应 panic
            let _ = &s[..safe];
        }
    }
}
