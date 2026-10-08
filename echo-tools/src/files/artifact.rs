/*!
 * 工具输出制品（artifact）的分页读取实现。
 *
 * 提供经过校验、UTF-8 安全的分页读取能力，以及面向模型的 `read_artifact` 工具。
 */

use echo_core::error::ToolError;
use echo_core::tokenizer::{HeuristicTokenizer, Tokenizer};
use echo_core::tools::artifact::{ToolOutputArtifactConfig, ToolOutputArtifactRef};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolFailureCategory, ToolParameters, ToolResult, ToolRiskLevel};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// 单页默认的内容 token 上限。
const DEFAULT_ARTIFACT_CONTENT_TOKENS: usize = 3_500;
/// 单页允许的内容 token 硬上限。
const MAX_ARTIFACT_CONTENT_TOKENS: usize = 3_500;
/// 每次预读 / 哈希时使用的缓冲区字节数。
const ARTIFACT_READ_AHEAD_BYTES: usize = 64 * 1024;

/// 调用方为单个制品分页选择的边界。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactPageLimit {
    /// 按 UTF-8 字节数限制分页；单个 Unicode 标量值可能超过该限制。
    Bytes(usize),
    /// 使用框架的启发式分词器按 token 数限制分页。
    Tokens(usize),
}

/// 不可变工具输出制品中经过校验、UTF-8 安全的一页内容。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPage {
    /// 本页的文本内容。
    pub content: String,
    /// 读取下一页所用的不透明游标；已读完时为 `None`。
    pub next_cursor: Option<String>,
    /// 是否已读到制品末尾。
    pub complete: bool,
    /// 制品完整内容的 SHA-256（小写十六进制）。
    pub sha256: String,
    /// 本页在文件中的起始字节偏移（含）。
    pub start_byte: u64,
    /// 本页在文件中的结束字节偏移（不含）。
    pub end_byte: u64,
    /// 制品文件总字节数。
    pub total_bytes: u64,
}

/// 供制品分页消费者使用的稳定错误分类。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactReadErrorKind {
    /// 制品引用无效（路径越界、不是文件、哈希格式错误等）。
    InvalidReference,
    /// 游标无效（格式错误、越界或不在字符边界上）。
    InvalidCursor,
    /// 制品在引用生成后或读取期间被修改。
    Changed,
    /// 制品内容不是合法的 UTF-8。
    InvalidUtf8,
    /// 底层 I/O 错误。
    Io,
}

/// 失败即关闭（fail-closed）的制品校验或读取错误。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactReadError {
    /// 错误分类。
    kind: ArtifactReadErrorKind,
    /// 面向人类的错误描述。
    message: String,
}

impl ArtifactReadError {
    /// 返回错误分类。
    pub fn kind(&self) -> ArtifactReadErrorKind {
        self.kind
    }

    /// 以指定分类和消息构造错误。
    fn new(kind: ArtifactReadErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ArtifactReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ArtifactReadError {}

/// 分页游标：绑定偏移量与制品快照身份，防止跨快照复用。
#[derive(Clone, Debug, PartialEq, Eq)]
struct ArtifactCursor {
    /// 下一页的起始字节偏移。
    offset: u64,
    /// 签发游标时的文件字节数。
    file_bytes: u64,
    /// 签发游标时的文件修改时间（纳秒）。
    modified_nanos: u128,
    /// 文件身份摘要（大小、修改时间、inode 等）。
    file_id: String,
    /// 快照身份摘要（根目录、路径与内容哈希）。
    snapshot_id: String,
    /// 期望的制品内容 SHA-256。
    expected_sha256: String,
}

impl ArtifactCursor {
    /// 将游标编码为 `v4:` 前缀的字符串。
    fn encode(self) -> String {
        format!(
            "v4:{}:{}:{}:{}:{}:{}",
            self.offset,
            self.file_bytes,
            self.modified_nanos,
            self.file_id,
            self.snapshot_id,
            self.expected_sha256
        )
    }

    /// 解析 `v4:` 格式的游标字符串，字段缺失或多余均视为错误。
    fn parse(value: &str) -> Result<Self, String> {
        let mut parts = value.split(':');
        if parts.next() != Some("v4") {
            return Err("artifact cursor has an unsupported version".to_string());
        }
        let offset = parse_cursor_number(parts.next(), "offset")?;
        let file_bytes = parse_cursor_number(parts.next(), "file size")?;
        let modified_nanos = parse_cursor_number(parts.next(), "modified time")?;
        let file_id = parse_cursor_hash(parts.next(), "file identity")?;
        let snapshot_id = parse_cursor_hash(parts.next(), "snapshot identity")?;
        let expected_sha256 = parse_cursor_hash(parts.next(), "artifact SHA-256")?;
        if parts.next().is_some() {
            return Err("artifact cursor contains unexpected fields".to_string());
        }
        Ok(Self {
            offset,
            file_bytes,
            modified_nanos,
            file_id,
            snapshot_id,
            expected_sha256,
        })
    }
}

/// 解析游标中的 64 位十六进制哈希字段，并统一转为小写。
fn parse_cursor_hash(value: Option<&str>, label: &str) -> Result<String, String> {
    value
        .filter(|value| value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()))
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| format!("artifact cursor has an invalid {label}"))
}

/// 解析游标中的数值字段。
fn parse_cursor_number<T>(value: Option<&str>, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .ok_or_else(|| format!("artifact cursor is missing {label}"))?
        .parse::<T>()
        .map_err(|_| format!("artifact cursor has an invalid {label}"))
}

/// 从参数中读取正整数 `max_tokens`，缺省时使用默认值，并截断到硬上限。
fn positive_token_limit(parameters: &ToolParameters) -> Result<usize, String> {
    let Some(value) = parameters.get("max_tokens") else {
        return Ok(DEFAULT_ARTIFACT_CONTENT_TOKENS);
    };
    let raw = value
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or_else(|| "'max_tokens' must be a positive integer".to_string())?;
    let requested = usize::try_from(raw).map_err(|_| "'max_tokens' is too large".to_string())?;
    Ok(requested.min(MAX_ARTIFACT_CONTENT_TOKENS))
}

/// 将修改时间转换为自 UNIX 纪元以来的纳秒数，失败时返回 0。
fn modified_nanos(modified: SystemTime) -> u128 {
    modified
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

/// 规范化制品路径，并确保其位于配置的制品根目录内且为普通文件。
fn resolve_artifact_path(
    config: &ToolOutputArtifactConfig,
    requested_path: &Path,
) -> Result<PathBuf, ArtifactReadError> {
    let root = std::fs::canonicalize(&config.root_dir).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("cannot resolve artifact root: {error}"),
        )
    })?;
    let candidate = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        config.root_dir.join(requested_path)
    };
    let path = std::fs::canonicalize(&candidate).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("cannot resolve artifact '{}': {error}", candidate.display()),
        )
    })?;
    if !path.starts_with(&root) {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!(
                "artifact '{}' is outside the configured artifact root",
                path.display()
            ),
        ));
    }
    let metadata = std::fs::metadata(&path).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("cannot inspect artifact '{}': {error}", path.display()),
        )
    })?;
    if !metadata.is_file() {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("artifact '{}' is not a file", path.display()),
        ));
    }
    Ok(path)
}

/// 解码 UTF-8 字节；若末尾是被截断的不完整字符则丢弃该部分，其余非法序列报错。
fn decode_utf8_prefix(bytes: Vec<u8>) -> Result<String, String> {
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text),
        Err(error) if error.utf8_error().error_len().is_none() => {
            let valid_up_to = error.utf8_error().valid_up_to();
            let mut bytes = error.into_bytes();
            bytes.truncate(valid_up_to);
            String::from_utf8(bytes)
                .map_err(|nested| format!("artifact contains invalid UTF-8: {nested}"))
        }
        Err(error) => Err(format!("artifact contains invalid UTF-8: {error}")),
    }
}

/// 通过二分查找取出不超过 `max_tokens` 的最长字符前缀（至少保留一个字符）。
fn token_bounded_prefix(text: &str, max_tokens: usize) -> String {
    let tokenizer = HeuristicTokenizer;
    if tokenizer.count_tokens(text) <= max_tokens {
        return text.to_string();
    }

    let mut low = 0_usize;
    let mut high = text.chars().count();
    while low < high {
        let middle = low.saturating_add(high).saturating_add(1) / 2;
        let candidate = text.chars().take(middle).collect::<String>();
        if tokenizer.count_tokens(&candidate) <= max_tokens {
            low = middle;
        } else {
            high = middle.saturating_sub(1);
        }
    }
    let character_limit = if low == 0 && !text.is_empty() { 1 } else { low };
    text.chars().take(character_limit).collect()
}

/// 取出不超过 `max_bytes` 字节的最长字符前缀（至少保留一个字符）。
fn byte_bounded_prefix(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut content = String::new();
    for character in text.chars() {
        if !content.is_empty() && content.len().saturating_add(character.len_utf8()) > max_bytes {
            break;
        }
        content.push(character);
    }
    content
}

/// 由文件元数据推导出的文件身份，用于检测读取期间的变化。
#[derive(Clone, Debug, PartialEq, Eq)]
struct ArtifactFileIdentity {
    /// 文件字节数。
    file_bytes: u64,
    /// 文件修改时间（纳秒）。
    modified_nanos: u128,
    /// 综合大小、修改时间及平台相关元数据（设备号、inode、ctime）的摘要。
    file_id: String,
}

/// 根据文件元数据计算 [`ArtifactFileIdentity`]。
fn file_identity(metadata: &std::fs::Metadata) -> ArtifactFileIdentity {
    let file_bytes = metadata.len();
    let modified_nanos = metadata.modified().map(modified_nanos).unwrap_or(0);
    let mut hasher = Sha256::new();
    hasher.update(file_bytes.to_le_bytes());
    hasher.update(modified_nanos.to_le_bytes());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        hasher.update(metadata.dev().to_le_bytes());
        hasher.update(metadata.ino().to_le_bytes());
        hasher.update(metadata.ctime().to_le_bytes());
        hasher.update(metadata.ctime_nsec().to_le_bytes());
    }
    ArtifactFileIdentity {
        file_bytes,
        modified_nanos,
        file_id: format!("{:x}", hasher.finalize()),
    }
}

/// 以只读、不跟随符号链接 / 重解析点的方式打开制品，并确认其为普通文件。
fn open_artifact_readonly(path: &Path) -> Result<File, ArtifactReadError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    #[cfg(not(any(unix, windows)))]
    {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            "atomic no-follow artifact reads are unavailable on this platform",
        ));
    }
    let file = options.open(path).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot open artifact: {error}"),
        )
    })?;
    let metadata = file.metadata().map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot inspect opened artifact: {error}"),
        )
    })?;
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if metadata.file_attributes()
            & windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT
            != 0
        {
            return Err(ArtifactReadError::new(
                ArtifactReadErrorKind::InvalidReference,
                format!(
                    "refusing to read artifact reparse point '{}'",
                    path.display()
                ),
            ));
        }
    }
    if !metadata.is_file() {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("artifact '{}' is not a regular file", path.display()),
        ));
    }
    Ok(file)
}

/// 从当前位置读取文件剩余内容并计算 SHA-256；可选地累加哈希次数计数器（测试用）。
fn sha256_reader(
    file: &mut File,
    hash_passes: Option<&AtomicUsize>,
) -> Result<String, ArtifactReadError> {
    if let Some(counter) = hash_passes {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; ARTIFACT_READ_AHEAD_BYTES];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            ArtifactReadError::new(
                ArtifactReadErrorKind::Io,
                format!("cannot hash artifact: {error}"),
            )
        })?;
        if read == 0 {
            break;
        }
        if let Some(chunk) = buffer.get(..read) {
            hasher.update(chunk);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// 校验 SHA-256 字符串为 64 位十六进制并转为小写。
fn validate_sha256(value: &str) -> Result<String, ArtifactReadError> {
    if value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            "artifact reference has an invalid SHA-256",
        ))
    }
}

/// 由规范化根目录、制品路径和内容哈希计算快照身份摘要。
fn artifact_snapshot_id(
    config: &ToolOutputArtifactConfig,
    path: &Path,
    sha256: &str,
) -> Result<String, ArtifactReadError> {
    let root = std::fs::canonicalize(&config.root_dir).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            format!("cannot resolve artifact root: {error}"),
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(root.as_os_str().as_encoded_bytes());
    hasher.update([0]);
    hasher.update(path.as_os_str().as_encoded_bytes());
    hasher.update([0]);
    hasher.update(sha256.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// 计算单次读取的字节窗口大小；限制值为 0 时报错。
fn read_window_bytes(limit: ArtifactPageLimit) -> Result<usize, ArtifactReadError> {
    let requested = match limit {
        ArtifactPageLimit::Bytes(value) | ArtifactPageLimit::Tokens(value) => value,
    };
    if requested == 0 {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            "artifact page limit must be positive",
        ));
    }
    Ok(match limit {
        ArtifactPageLimit::Bytes(value) => value.saturating_add(3),
        ArtifactPageLimit::Tokens(_) => ARTIFACT_READ_AHEAD_BYTES,
    }
    .min(ARTIFACT_READ_AHEAD_BYTES))
}

/// 从头计算整个文件的哈希并与期望值比较，同时确认哈希期间文件身份未变化。
fn verify_artifact_digest(
    file: &mut File,
    expected_sha256: &str,
    identity: &ArtifactFileIdentity,
    hash_passes: Option<&AtomicUsize>,
) -> Result<(), ArtifactReadError> {
    file.seek(std::io::SeekFrom::Start(0)).map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot seek artifact for hashing: {error}"),
        )
    })?;
    let actual_sha256 = sha256_reader(file, hash_passes)?;
    let after_hash = file.metadata().map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot re-inspect artifact after hashing: {error}"),
        )
    })?;
    if file_identity(&after_hash) != *identity {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Changed,
            "artifact changed while its snapshot was being verified",
        ));
    }
    if actual_sha256 != expected_sha256 {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Changed,
            format!(
                "artifact SHA-256 mismatch: expected {expected_sha256}, actual {actual_sha256}"
            ),
        ));
    }
    Ok(())
}

/**
 * 从已持久化的工具输出制品中读取一页经过校验的内容。
 *
 * 必须同时提供配置的根目录与完整的制品引用。
 * 本函数会拒绝路径逃逸、符号链接逃逸、过期长度、摘要不匹配，
 * 以及为其它制品快照签发的游标。
 */
pub fn read_artifact_page(
    config: &ToolOutputArtifactConfig,
    artifact: &ToolOutputArtifactRef,
    cursor: Option<&str>,
    limit: ArtifactPageLimit,
) -> Result<ArtifactPage, ArtifactReadError> {
    read_artifact_page_inner(config, artifact, cursor, limit, None, None)
}

/**
 * 分页读取的内部实现。
 *
 * `initial_verified_identity` 表示调用方已完成首次摘要校验时的文件身份，可避免重复哈希；
 * `hash_passes` 用于测试中统计完整哈希次数。首页与末页会做完整摘要校验。
 */
fn read_artifact_page_inner(
    config: &ToolOutputArtifactConfig,
    artifact: &ToolOutputArtifactRef,
    cursor: Option<&str>,
    limit: ArtifactPageLimit,
    initial_verified_identity: Option<&ArtifactFileIdentity>,
    hash_passes: Option<&AtomicUsize>,
) -> Result<ArtifactPage, ArtifactReadError> {
    let path = resolve_artifact_path(config, &artifact.path)?;
    let expected_sha256 = validate_sha256(&artifact.sha256)?;
    if artifact.payload_bytes > artifact.artifact_bytes {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidReference,
            "artifact reference payload length exceeds its file length",
        ));
    }

    let mut file = open_artifact_readonly(&path)?;
    let metadata = file.metadata().map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot inspect artifact: {error}"),
        )
    })?;
    let identity = file_identity(&metadata);
    if identity.file_bytes != artifact.artifact_bytes {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Changed,
            format!(
                "artifact length changed: expected {}, actual {}",
                artifact.artifact_bytes, identity.file_bytes
            ),
        ));
    }
    let snapshot_id = artifact_snapshot_id(config, &path, &expected_sha256)?;

    let parsed_cursor = cursor
        .map(ArtifactCursor::parse)
        .transpose()
        .map_err(|message| ArtifactReadError::new(ArtifactReadErrorKind::InvalidCursor, message))?;
    if let Some(cursor) = parsed_cursor.as_ref()
        && (cursor.file_bytes != identity.file_bytes
            || cursor.modified_nanos != identity.modified_nanos
            || cursor.file_id != identity.file_id
            || cursor.snapshot_id != snapshot_id
            || cursor.expected_sha256 != expected_sha256)
    {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Changed,
            "artifact cursor does not identify the current artifact snapshot",
        ));
    }
    if parsed_cursor.is_none() {
        match initial_verified_identity {
            Some(verified) if verified == &identity => {}
            Some(_) => {
                return Err(ArtifactReadError::new(
                    ArtifactReadErrorKind::Changed,
                    "artifact changed after its initial digest was verified",
                ));
            }
            None => verify_artifact_digest(&mut file, &expected_sha256, &identity, hash_passes)?,
        }
    }
    let start = parsed_cursor
        .as_ref()
        .map(|value| value.offset)
        .unwrap_or(0);
    if start > identity.file_bytes {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidCursor,
            format!(
                "artifact cursor exceeds file size ({} bytes)",
                identity.file_bytes
            ),
        ));
    }

    let read_limit = read_window_bytes(limit)?;
    file.seek(std::io::SeekFrom::Start(start))
        .map_err(|error| {
            ArtifactReadError::new(
                ArtifactReadErrorKind::InvalidCursor,
                format!("cannot seek artifact: {error}"),
            )
        })?;
    let remaining = identity.file_bytes.saturating_sub(start);
    let read_limit = remaining.min(u64::try_from(read_limit).unwrap_or(u64::MAX));
    let mut bytes = Vec::new();
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            ArtifactReadError::new(
                ArtifactReadErrorKind::Io,
                format!("cannot read artifact: {error}"),
            )
        })?;
    let decoded = decode_utf8_prefix(bytes)
        .map_err(|message| ArtifactReadError::new(ArtifactReadErrorKind::InvalidUtf8, message))?;
    let content = match limit {
        ArtifactPageLimit::Bytes(max_bytes) => byte_bounded_prefix(&decoded, max_bytes),
        ArtifactPageLimit::Tokens(max_tokens) => token_bounded_prefix(&decoded, max_tokens),
    };
    let consumed = u64::try_from(content.len()).unwrap_or(u64::MAX);
    if start < identity.file_bytes && consumed == 0 {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::InvalidCursor,
            "artifact cursor is not on a UTF-8 character boundary",
        ));
    }

    let after_read = file.metadata().map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot re-inspect artifact after reading: {error}"),
        )
    })?;
    if file_identity(&after_read) != identity {
        return Err(ArtifactReadError::new(
            ArtifactReadErrorKind::Changed,
            "artifact changed while a page was being read",
        ));
    }

    let end = start.saturating_add(consumed).min(identity.file_bytes);
    let complete = end >= identity.file_bytes;
    if complete && parsed_cursor.is_some() {
        verify_artifact_digest(&mut file, &expected_sha256, &identity, hash_passes)?;
    }
    let next_cursor = (!complete).then(|| {
        ArtifactCursor {
            offset: end,
            file_bytes: identity.file_bytes,
            modified_nanos: identity.modified_nanos,
            file_id: identity.file_id.clone(),
            snapshot_id,
            expected_sha256: expected_sha256.clone(),
        }
        .encode()
    });
    Ok(ArtifactPage {
        content,
        next_cursor,
        complete,
        sha256: expected_sha256,
        start_byte: start,
        end_byte: end,
        total_bytes: identity.file_bytes,
    })
}

/**
 * 根据工具请求构造制品引用。
 *
 * 若提供了期望哈希则直接采用（后续读取时校验）；否则现场计算哈希，
 * 并返回已校验的文件身份以便首页跳过重复哈希。
 */
fn artifact_ref_for_tool_request(
    config: &ToolOutputArtifactConfig,
    requested: &Path,
    expected_sha256: Option<&str>,
) -> Result<(ToolOutputArtifactRef, Option<ArtifactFileIdentity>), ArtifactReadError> {
    let path = resolve_artifact_path(config, requested)?;
    let mut file = open_artifact_readonly(&path)?;
    let metadata = file.metadata().map_err(|error| {
        ArtifactReadError::new(
            ArtifactReadErrorKind::Io,
            format!("cannot inspect artifact: {error}"),
        )
    })?;
    let identity = file_identity(&metadata);
    let (sha256, initial_verified_identity) = match expected_sha256 {
        Some(value) => (validate_sha256(value)?, None),
        None => {
            let sha256 = sha256_reader(&mut file, None)?;
            let after_hash = file.metadata().map_err(|error| {
                ArtifactReadError::new(
                    ArtifactReadErrorKind::Io,
                    format!("cannot re-inspect artifact after hashing: {error}"),
                )
            })?;
            if file_identity(&after_hash) != identity {
                return Err(ArtifactReadError::new(
                    ArtifactReadErrorKind::Changed,
                    "artifact changed while its initial reference was derived",
                ));
            }
            (sha256, Some(identity.clone()))
        }
    };
    Ok((
        ToolOutputArtifactRef {
            path,
            artifact_bytes: identity.file_bytes,
            payload_bytes: identity.file_bytes,
            sha256,
            retention: config.retention.clone(),
        },
        initial_verified_identity,
    ))
}

/// 通过不透明字节游标分页读取不可变工具输出制品的工具。
pub struct ReadArtifactTool;

impl Tool for ReadArtifactTool {
    /// 工具名称。
    fn name(&self) -> &str {
        "read_artifact"
    }

    /// 面向模型的工具描述。
    fn description(&self) -> &str {
        "Read the complete spilled output of a previous tool call in bounded UTF-8 pages. Use the returned next_cursor until complete."
    }

    /// 所需权限：只读。
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Read]
    }

    /// 风险等级：只读。
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 参数的 JSON Schema。
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact artifact path returned by a previous tool result"
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque next_cursor returned by the previous read_artifact page"
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ARTIFACT_CONTENT_TOKENS,
                    "description": "Maximum content tokens for this page (default and hard maximum: 3500)"
                },
                "expected_sha256": {
                    "type": "string",
                    "description": "Optional full SHA-256 from the spill notice; fully verified on the first and final pages and bound into every cursor"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    /// 执行一次分页读取：校验参数后在阻塞线程中读取，并在结果元数据中附带分页信息。
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, echo_core::error::Result<ToolResult>> {
        Box::pin(async move {
            let allowed_parameters = ["path", "cursor", "max_tokens", "expected_sha256"];
            let mut unknown = parameters
                .keys()
                .filter(|key| !allowed_parameters.contains(&key.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            unknown.sort();
            if !unknown.is_empty() {
                return Ok(ToolResult::failure(
                    ToolFailureCategory::InvalidArguments,
                    format!("Unknown read_artifact parameter(s): {}", unknown.join(", ")),
                ));
            }

            let path_value = parameters
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::MissingParameter("path".to_string()))?;
            let max_tokens = match positive_token_limit(&parameters) {
                Ok(value) => value,
                Err(message) => {
                    return Ok(ToolResult::failure(
                        ToolFailureCategory::InvalidArguments,
                        message,
                    ));
                }
            };
            let config = match ctx.output_artifacts.clone() {
                Some(config) => config,
                None => {
                    return Ok(ToolResult::failure(
                        ToolFailureCategory::InvalidArguments,
                        "read_artifact is unavailable because this run has no artifact store",
                    ));
                }
            };
            let requested_path = PathBuf::from(path_value);
            let cursor = parameters
                .get("cursor")
                .and_then(Value::as_str)
                .map(str::to_string);
            let expected_sha256 = parameters
                .get("expected_sha256")
                .and_then(Value::as_str)
                .map(str::to_string);
            let read = tokio::task::spawn_blocking(move || {
                let parsed_cursor = cursor
                    .as_deref()
                    .map(ArtifactCursor::parse)
                    .transpose()
                    .map_err(|message| {
                        ArtifactReadError::new(ArtifactReadErrorKind::InvalidCursor, message)
                    })?;
                let effective_sha256 = expected_sha256.as_deref().or_else(|| {
                    parsed_cursor
                        .as_ref()
                        .map(|value| value.expected_sha256.as_str())
                });
                let (artifact, initial_verified_identity) =
                    artifact_ref_for_tool_request(&config, &requested_path, effective_sha256)?;
                let path = artifact.path.clone();
                let page = read_artifact_page_inner(
                    &config,
                    &artifact,
                    cursor.as_deref(),
                    ArtifactPageLimit::Tokens(max_tokens),
                    initial_verified_identity.as_ref(),
                    None,
                )?;
                Ok::<_, ArtifactReadError>((path, page))
            })
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "read_artifact".to_string(),
                message: format!("artifact read task failed: {error}"),
            })?;
            let (path, page) = match read {
                Ok(value) => value,
                Err(error) => {
                    let category = match error.kind() {
                        ArtifactReadErrorKind::Changed | ArtifactReadErrorKind::Io => {
                            ToolFailureCategory::Transient
                        }
                        ArtifactReadErrorKind::InvalidReference
                        | ArtifactReadErrorKind::InvalidCursor
                        | ArtifactReadErrorKind::InvalidUtf8 => {
                            ToolFailureCategory::InvalidArguments
                        }
                    };
                    return Ok(ToolResult::failure(category, error.to_string()));
                }
            };
            let notice = match page.next_cursor.as_deref() {
                Some(next) => format!(
                    "\n\n[Artifact page: bytes {}-{} of {}; continue with cursor={next}.]",
                    page.start_byte, page.end_byte, page.total_bytes
                ),
                None => format!(
                    "\n\n[Artifact complete: bytes {}-{} of {}.]",
                    page.start_byte, page.end_byte, page.total_bytes
                ),
            };
            let mut result = ToolResult::success(format!("{}{notice}", page.content))
                .with_truncated(!page.complete);
            result
                .metadata
                .insert("artifact_path".to_string(), path.display().to_string());
            result
                .metadata
                .insert("start_byte".to_string(), page.start_byte.to_string());
            result
                .metadata
                .insert("end_byte".to_string(), page.end_byte.to_string());
            result
                .metadata
                .insert("total_bytes".to_string(), page.total_bytes.to_string());
            result
                .metadata
                .insert("total_known".to_string(), "true".to_string());
            result.metadata.insert("sha256".to_string(), page.sha256);
            if let Some(next) = page.next_cursor {
                result.metadata.insert("next_cursor".to_string(), next);
            }
            Ok(result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::tools::artifact::{ToolOutputArtifactIdentity, persist_tool_output};

    /// 生成带进程号与时间戳的唯一临时根目录。
    fn test_root(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "echo-read-artifact-{label}-{}-{}",
            std::process::id(),
            nonce
        ))
    }

    /// 在指定根目录下持久化一段内容并返回制品引用。
    fn create_artifact(
        root: &Path,
        content: &str,
    ) -> echo_core::error::Result<echo_core::tools::artifact::ToolOutputArtifactRef> {
        let config = ToolOutputArtifactConfig::new(root, "test").threshold_bytes(1);
        persist_tool_output(
            config,
            ToolOutputArtifactIdentity {
                conversation_id: Some("conversation".to_string()),
                run_id: Some("run".to_string()),
                call_id: "call".to_string(),
                tool_name: "shell".to_string(),
            },
            content,
        )?
        .ok_or_else(|| echo_core::error::ReactError::Other("artifact was not created".to_string()))
    }

    #[test]
    fn q_flt_v02_unicode_pages_never_split_utf8_scalars() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = test_root("canonical-utf8");
        let artifact = create_artifact(&root, "你🙂好")?;
        let config = ToolOutputArtifactConfig::new(&root, "test");

        let first = read_artifact_page(&config, &artifact, None, ArtifactPageLimit::Bytes(1))?;
        assert_eq!(first.content, "你");
        assert!(!first.complete);
        let cursor = first
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first page did not return a cursor".to_string())?;
        assert!(cursor.starts_with("v4:"));
        let second = read_artifact_page(
            &config,
            &artifact,
            Some(cursor),
            ArtifactPageLimit::Bytes(4),
        )?;
        assert_eq!(second.content, "🙂");
        assert!(!second.complete);
        let cursor = second
            .next_cursor
            .as_deref()
            .ok_or_else(|| "second page did not return a cursor".to_string())?;
        let third = read_artifact_page(
            &config,
            &artifact,
            Some(cursor),
            ArtifactPageLimit::Bytes(4),
        )?;
        assert_eq!(third.content, "好");
        assert!(third.complete);
        assert_eq!(third.sha256, artifact.sha256);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn canonical_reader_rejects_mutated_artifact() -> Result<(), Box<dyn std::error::Error>> {
        let root = test_root("canonical-mutation");
        let artifact = create_artifact(&root, "original")?;
        let config = ToolOutputArtifactConfig::new(&root, "test");
        std::fs::write(&artifact.path, "mutated!")?;

        let error = read_artifact_page(&config, &artifact, None, ArtifactPageLimit::Bytes(16))
            .err()
            .ok_or_else(|| "mutated artifact was accepted".to_string())?;
        assert_eq!(error.kind(), ArtifactReadErrorKind::Changed);
        assert!(error.to_string().contains("SHA-256 mismatch"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn canonical_reader_rejects_same_size_mutation_on_final_page()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = test_root("canonical-final-mutation");
        let artifact = create_artifact(&root, "abcdefgh")?;
        let config = ToolOutputArtifactConfig::new(&root, "test");
        let first = read_artifact_page(&config, &artifact, None, ArtifactPageLimit::Bytes(4))?;
        let cursor = first
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first page did not return a cursor".to_string())?;
        std::fs::write(&artifact.path, "abcdWXYZ")?;

        let error = read_artifact_page(
            &config,
            &artifact,
            Some(cursor),
            ArtifactPageLimit::Bytes(4),
        )
        .err()
        .ok_or_else(|| "same-size final-page mutation was accepted".to_string())?;
        assert_eq!(error.kind(), ArtifactReadErrorKind::Changed);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn canonical_reader_hashes_only_first_and_final_pages() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = test_root("canonical-linear-hashing");
        let content = "中文🙂-linear-pages\n".repeat(200);
        let artifact = create_artifact(&root, &content)?;
        let config = ToolOutputArtifactConfig::new(&root, "test");
        let hash_passes = AtomicUsize::new(0);
        let mut cursor = None;
        let mut recovered = String::new();
        let mut page_count = 0_usize;

        loop {
            let page = read_artifact_page_inner(
                &config,
                &artifact,
                cursor.as_deref(),
                ArtifactPageLimit::Bytes(13),
                None,
                Some(&hash_passes),
            )?;
            recovered.push_str(&page.content);
            page_count = page_count.saturating_add(1);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }

        assert!(page_count > 100);
        assert_eq!(hash_passes.load(Ordering::Relaxed), 2);
        assert_eq!(recovered, content);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn canonical_reader_binds_cursor_to_artifact_identity() -> Result<(), Box<dyn std::error::Error>>
    {
        let root = test_root("canonical-cursor-identity");
        let first_artifact = create_artifact(&root, &"same content".repeat(100))?;
        let second_artifact = create_artifact(&root, &"same content".repeat(100))?;
        let config = ToolOutputArtifactConfig::new(&root, "test");
        let first_page =
            read_artifact_page(&config, &first_artifact, None, ArtifactPageLimit::Bytes(8))?;
        let cursor = first_page
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first artifact did not return a cursor".to_string())?;

        let error = read_artifact_page(
            &config,
            &second_artifact,
            Some(cursor),
            ArtifactPageLimit::Bytes(8),
        )
        .err()
        .ok_or_else(|| "cursor was accepted for a different artifact".to_string())?;
        assert_eq!(error.kind(), ArtifactReadErrorKind::Changed);
        assert!(error.to_string().contains("snapshot"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn canonical_reader_rejects_symlink_escape() -> Result<(), Box<dyn std::error::Error>> {
        let root = test_root("canonical-symlink-root");
        let outside = test_root("canonical-symlink-outside");
        let outside_artifact = create_artifact(&outside, "outside")?;
        std::fs::create_dir_all(&root)?;
        let link = root.join("artifact-link");
        std::os::unix::fs::symlink(&outside_artifact.path, &link)?;
        let mut linked_ref = outside_artifact;
        linked_ref.path = link;
        let config = ToolOutputArtifactConfig::new(&root, "test");

        let error = read_artifact_page(&config, &linked_ref, None, ArtifactPageLimit::Bytes(16))
            .err()
            .ok_or_else(|| "symlink escape was accepted".to_string())?;
        assert_eq!(error.kind(), ArtifactReadErrorKind::InvalidReference);
        assert!(error.to_string().contains("outside"));
        std::fs::remove_dir_all(root)?;
        std::fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[tokio::test]
    async fn q_flt_v10_paged_spilled_result_is_fully_recoverable() -> echo_core::error::Result<()> {
        let root = test_root("single-line");
        let content = format!(r#"{{"payload":"{}"}}"#, "数据🙂".repeat(110_000));
        assert!(content.len() >= 1024 * 1024);
        let artifact = create_artifact(&root, &content)?;
        let config = ToolOutputArtifactConfig::new(&root, "test").threshold_bytes(1);
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(config),
            ..Default::default()
        };
        let tool = ReadArtifactTool;
        let mut cursor = None;
        let mut recovered = String::new();
        loop {
            let mut parameters = ToolParameters::from([
                (
                    "path".to_string(),
                    Value::String(artifact.path.display().to_string()),
                ),
                ("max_tokens".to_string(), Value::from(128_u64)),
            ]);
            if let Some(value) = cursor.clone() {
                parameters.insert("cursor".to_string(), Value::String(value));
            }
            let result = tool.execute_with_context(parameters, &ctx).await?;
            assert!(result.success, "{}", result.error.unwrap_or_default());
            assert!(HeuristicTokenizer.count_tokens(&result.output) <= 4_000);
            assert_eq!(
                result.metadata.get("sha256").map(String::as_str),
                Some(artifact.sha256.as_str())
            );
            let (page, _) = result.output.split_once("\n\n[Artifact ").ok_or_else(|| {
                echo_core::error::ReactError::Other("missing page notice".to_string())
            })?;
            recovered.push_str(page);
            cursor = result.metadata.get("next_cursor").cloned();
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(recovered, content);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn rejects_cursor_after_artifact_changes() -> echo_core::error::Result<()> {
        let root = test_root("changed");
        let artifact = create_artifact(&root, &"中文🙂".repeat(1_000))?;
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(&root, "test")),
            ..Default::default()
        };
        let first = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([
                    (
                        "path".to_string(),
                        Value::String(artifact.path.display().to_string()),
                    ),
                    ("max_tokens".to_string(), Value::from(16_u64)),
                ]),
                &ctx,
            )
            .await?;
        let cursor = first.metadata.get("next_cursor").cloned().ok_or_else(|| {
            echo_core::error::ReactError::Other("first page had no cursor".to_string())
        })?;
        std::fs::write(&artifact.path, "changed")?;

        let second = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([
                    (
                        "path".to_string(),
                        Value::String(artifact.path.display().to_string()),
                    ),
                    ("cursor".to_string(), Value::String(cursor)),
                ]),
                &ctx,
            )
            .await?;
        assert!(!second.success);
        assert_eq!(
            second.failure.as_ref().map(|failure| failure.category),
            Some(ToolFailureCategory::Transient)
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[tokio::test]
    async fn reports_deleted_artifact() -> echo_core::error::Result<()> {
        let root = test_root("deleted");
        let artifact = create_artifact(&root, "temporary")?;
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(&root, "test")),
            ..Default::default()
        };
        std::fs::remove_file(&artifact.path)?;
        let result = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([(
                    "path".to_string(),
                    Value::String(artifact.path.display().to_string()),
                )]),
                &ctx,
            )
            .await?;
        assert!(!result.success);
        assert!(result.error.unwrap_or_default().contains("cannot resolve"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_symlink_escape_from_artifact_root() -> echo_core::error::Result<()> {
        let root = test_root("symlink-root");
        let outside = test_root("symlink-outside");
        let artifact = create_artifact(&outside, "outside")?;
        std::fs::create_dir_all(&root)?;
        let link = root.join("artifact-link");
        std::os::unix::fs::symlink(&artifact.path, &link)?;
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(&root, "test")),
            ..Default::default()
        };
        let result = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([(
                    "path".to_string(),
                    Value::String(link.display().to_string()),
                )]),
                &ctx,
            )
            .await?;
        assert!(!result.success);
        assert!(result.error.unwrap_or_default().contains("outside"));
        std::fs::remove_dir_all(root)?;
        std::fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[tokio::test]
    async fn rejects_artifacts_outside_configured_root() -> echo_core::error::Result<()> {
        let root = test_root("root");
        let outside = test_root("outside");
        let artifact = create_artifact(&outside, "secret")?;
        std::fs::create_dir_all(&root)?;
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(&root, "test")),
            ..Default::default()
        };
        let result = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([(
                    "path".to_string(),
                    Value::String(artifact.path.display().to_string()),
                )]),
                &ctx,
            )
            .await?;
        assert!(!result.success);
        std::fs::remove_dir_all(root)?;
        std::fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[tokio::test]
    async fn verifies_first_page_hash() -> echo_core::error::Result<()> {
        let root = test_root("hash");
        let artifact = create_artifact(&root, "complete result")?;
        let ctx = echo_core::tools::ToolContext {
            output_artifacts: Some(ToolOutputArtifactConfig::new(&root, "test")),
            ..Default::default()
        };
        let result = ReadArtifactTool
            .execute_with_context(
                ToolParameters::from([
                    (
                        "path".to_string(),
                        Value::String(artifact.path.display().to_string()),
                    ),
                    ("expected_sha256".to_string(), Value::String("0".repeat(64))),
                ]),
                &ctx,
            )
            .await?;
        assert!(!result.success);
        assert!(result.error.unwrap_or_default().contains("mismatch"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
