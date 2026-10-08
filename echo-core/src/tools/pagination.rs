//! 返回集合的工具所共享的游标分页契约。

use super::{ToolParameters, ToolResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::io::{self, Write};

/// 当前游标编码格式的版本号。
const CURSOR_VERSION: u8 = 1;
/// 游标字符串允许的最大字符数。
const MAX_CURSOR_CHARS: usize = 1_024;

/// 返回集合的工具所共享的分页输入。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRequest {
    /// 每页最多返回的条目数。
    pub limit: usize,
    /// 上一页返回的续页游标；为 `None` 时从第一页开始。
    pub cursor: Option<String>,
}

/// 返回集合的工具所共享的分页元数据。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageInfo {
    /// 获取下一页所用的游标；没有更多数据时为 `None`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// 本页之后是否还有未返回的数据。
    pub truncated: bool,
    /// 结果总数是否已知。
    pub total_known: bool,
    /// 结果总数（已知时）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    /// 本页实际返回的条目数。
    pub returned: usize,
}

/// 分页参数解析或游标校验失败时的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageError {
    /// `limit` 超出允许范围。
    InvalidLimit {
        /// 调用方请求的 limit。
        limit: usize,
        /// 允许的最大 limit。
        max: usize,
    },
    /// 分页参数无效，携带错误描述。
    InvalidParameter(String),
    /// 游标格式无效，携带错误描述。
    InvalidCursor(String),
    /// 游标与当前查询或结果快照不匹配。
    CursorQueryMismatch,
    /// 游标偏移超出了当前结果数量。
    CursorOutOfRange {
        /// 游标记录的偏移量。
        offset: usize,
        /// 当前结果总数。
        total: usize,
    },
}

impl fmt::Display for PageError {
    /// 输出面向模型的错误说明。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit { limit, max } => {
                write!(
                    formatter,
                    "limit must be between 1 and {max}, received {limit}"
                )
            }
            Self::InvalidParameter(message) | Self::InvalidCursor(message) => {
                formatter.write_str(message)
            }
            Self::CursorQueryMismatch => formatter.write_str(
                "cursor does not match the current query or result snapshot; restart without cursor",
            ),
            Self::CursorOutOfRange { offset, total } => write!(
                formatter,
                "cursor offset {offset} is beyond the current result count {total}; restart without cursor"
            ),
        }
    }
}

impl std::error::Error for PageError {}

/// 游标的内部结构，序列化为 JSON 后以十六进制编码。
#[derive(Debug, Serialize, Deserialize)]
struct CursorEnvelope {
    /// 游标格式版本。
    version: u8,
    /// 查询与结果快照的指纹，用于检测查询变更。
    query_fingerprint: String,
    /// 下一页的起始偏移。
    offset: usize,
}

/// 参与计算指纹的输入。
#[derive(Debug, Serialize)]
struct FingerprintInput<'a, Q, T> {
    /// 调用方的查询参数。
    query: &'a Q,
    /// 每页条目数。
    limit: usize,
    /// 完整的结果集快照。
    items: &'a [T],
}

impl PageRequest {
    /// 从工具参数中解析 `limit` 与 `cursor`，并校验 limit 是否在 `1..=max_limit` 范围内。
    pub fn from_parameters(
        parameters: &ToolParameters,
        default_limit: usize,
        max_limit: usize,
    ) -> Result<Self, PageError> {
        if default_limit == 0 || default_limit > max_limit {
            return Err(PageError::InvalidParameter(
                "pagination defaults are invalid".to_string(),
            ));
        }
        let limit = match parameters.get("limit") {
            Some(value) => value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    PageError::InvalidParameter("limit must be a positive integer".to_string())
                })?,
            None => default_limit,
        };
        if limit == 0 || limit > max_limit {
            return Err(PageError::InvalidLimit {
                limit,
                max: max_limit,
            });
        }
        let cursor = match parameters.get("cursor") {
            Some(value) => Some(
                value
                    .as_str()
                    .filter(|cursor| !cursor.is_empty())
                    .ok_or_else(|| {
                        PageError::InvalidParameter("cursor must be a non-empty string".to_string())
                    })?
                    .to_string(),
            ),
            None => None,
        };
        Ok(Self { limit, cursor })
    }

    /**
    对完整结果集进行分页，返回当前页的条目及分页元数据。

    若带有游标，会校验其版本以及查询/结果快照指纹是否与当前一致。
    */
    pub fn paginate<T, Q>(&self, items: Vec<T>, query: &Q) -> Result<(Vec<T>, PageInfo), PageError>
    where
        T: Serialize,
        Q: Serialize,
    {
        let fingerprint = query_fingerprint(query, self.limit, &items)?;
        let offset = match self.cursor.as_deref() {
            Some(cursor) => {
                let envelope = decode_cursor(cursor)?;
                if envelope.version != CURSOR_VERSION {
                    return Err(PageError::InvalidCursor(format!(
                        "unsupported cursor version {}",
                        envelope.version
                    )));
                }
                if envelope.query_fingerprint != fingerprint {
                    return Err(PageError::CursorQueryMismatch);
                }
                envelope.offset
            }
            None => 0,
        };

        let total = items.len();
        if offset > total {
            return Err(PageError::CursorOutOfRange { offset, total });
        }
        let end = offset.saturating_add(self.limit).min(total);
        let returned = end.saturating_sub(offset);
        let page = items.into_iter().skip(offset).take(returned).collect();
        let next_cursor = if end < total {
            Some(encode_cursor(&CursorEnvelope {
                version: CURSOR_VERSION,
                query_fingerprint: fingerprint,
                offset: end,
            })?)
        } else {
            None
        };
        Ok((
            page,
            PageInfo {
                truncated: next_cursor.is_some(),
                next_cursor,
                total_known: true,
                total: Some(total),
                returned,
            },
        ))
    }
}

impl PageInfo {
    /// 附加通用分页契约，并向模型暴露续页信息。
    pub fn apply_to(&self, result: &mut ToolResult) {
        result.truncated = result.truncated || self.truncated;
        result
            .metadata
            .insert("page.truncated".to_string(), self.truncated.to_string());
        result
            .metadata
            .insert("page.total_known".to_string(), self.total_known.to_string());
        result
            .metadata
            .insert("page.returned".to_string(), self.returned.to_string());
        if let Some(total) = self.total {
            result
                .metadata
                .insert("page.total".to_string(), total.to_string());
        }
        if let Some(cursor) = &self.next_cursor {
            result
                .metadata
                .insert("page.next_cursor".to_string(), cursor.clone());
            let continuation = serde_json::json!({
                "next_cursor": cursor,
                "returned": self.returned,
                "total": self.total,
                "total_known": self.total_known,
                "truncated": true,
            });
            result.output.push_str("\n[page]");
            result.output.push_str(&continuation.to_string());
        }
    }
}

/// 计算查询参数、limit 与结果快照的 SHA-256 指纹。
fn query_fingerprint<Q: Serialize, T: Serialize>(
    query: &Q,
    limit: usize,
    items: &[T],
) -> Result<String, PageError> {
    let mut writer = Sha256Writer::default();
    serde_json::to_writer(
        &mut writer,
        &FingerprintInput {
            query,
            limit,
            items,
        },
    )
    .map_err(|error| {
        PageError::InvalidParameter(format!("cannot fingerprint pagination snapshot: {error}"))
    })?;
    Ok(format!("{:x}", writer.0.finalize()))
}

/// 将写入的字节直接喂给 SHA-256 哈希器的 `Write` 适配器。
#[derive(Default)]
struct Sha256Writer(Sha256);

impl Write for Sha256Writer {
    /// 将字节追加到哈希状态中。
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    /// 无缓冲，无需刷新。
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// 将游标结构编码为十六进制字符串。
fn encode_cursor(envelope: &CursorEnvelope) -> Result<String, PageError> {
    let bytes = serde_json::to_vec(envelope)
        .map_err(|error| PageError::InvalidCursor(format!("cannot encode cursor: {error}")))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// 从十六进制字符串解码游标结构，并校验长度与格式。
fn decode_cursor(cursor: &str) -> Result<CursorEnvelope, PageError> {
    if cursor.chars().count() > MAX_CURSOR_CHARS || !cursor.len().is_multiple_of(2) {
        return Err(PageError::InvalidCursor(
            "cursor has an invalid length".to_string(),
        ));
    }
    let chars: Vec<char> = cursor.chars().collect();
    let mut bytes = Vec::with_capacity(chars.len() / 2);
    let (pairs, remainder) = chars.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(PageError::InvalidCursor(
            "cursor has an invalid length".to_string(),
        ));
    }
    for pair in pairs {
        let high = pair
            .first()
            .and_then(|value| value.to_digit(16))
            .ok_or_else(|| PageError::InvalidCursor("cursor is not valid hex".to_string()))?;
        let low = pair
            .get(1)
            .and_then(|value| value.to_digit(16))
            .ok_or_else(|| PageError::InvalidCursor("cursor is not valid hex".to_string()))?;
        let byte = high
            .checked_mul(16)
            .and_then(|value| value.checked_add(low))
            .and_then(|value| u8::try_from(value).ok())
            .ok_or_else(|| PageError::InvalidCursor("cursor byte is invalid".to_string()))?;
        bytes.push(byte);
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| PageError::InvalidCursor(format!("cursor is malformed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 构造带有 limit 和可选 cursor 的测试参数。
    fn params(limit: u64, cursor: Option<&str>) -> ToolParameters {
        let mut parameters = ToolParameters::new();
        parameters.insert("limit".to_string(), json!(limit));
        if let Some(cursor) = cursor {
            parameters.insert("cursor".to_string(), json!(cursor));
        }
        parameters
    }

    #[test]
    fn pages_without_duplicates_and_preserves_unicode() -> Result<(), String> {
        let items = vec!["中文", "emoji🙂", "third", "fourth"];
        let first = PageRequest::from_parameters(&params(2, None), 2, 10)
            .map_err(|error| error.to_string())?;
        let (first_items, first_info) = first
            .paginate(items.clone(), &json!({"query": "same"}))
            .map_err(|error| error.to_string())?;
        assert_eq!(first_items, vec!["中文", "emoji🙂"]);
        let cursor = first_info
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first page must have a cursor".to_string())?;
        let second = PageRequest::from_parameters(&params(2, Some(cursor)), 2, 10)
            .map_err(|error| error.to_string())?;
        let (second_items, second_info) = second
            .paginate(items, &json!({"query": "same"}))
            .map_err(|error| error.to_string())?;
        assert_eq!(second_items, vec!["third", "fourth"]);
        assert!(!second_info.truncated);
        assert_eq!(second_info.total, Some(4));
        Ok(())
    }

    #[test]
    fn rejects_cursor_after_query_or_limit_changes() -> Result<(), String> {
        let items = vec![1, 2, 3];
        let first = PageRequest::from_parameters(&params(1, None), 1, 10)
            .map_err(|error| error.to_string())?;
        let (_, info) = first
            .paginate(items.clone(), &json!({"query": "old"}))
            .map_err(|error| error.to_string())?;
        let cursor = info
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first page must have a cursor".to_string())?;

        let changed_query = PageRequest::from_parameters(&params(1, Some(cursor)), 1, 10)
            .map_err(|error| error.to_string())?;
        assert_eq!(
            changed_query.paginate(items.clone(), &json!({"query": "new"})),
            Err(PageError::CursorQueryMismatch)
        );
        let changed_limit = PageRequest::from_parameters(&params(2, Some(cursor)), 1, 10)
            .map_err(|error| error.to_string())?;
        assert_eq!(
            changed_limit.paginate(items, &json!({"query": "old"})),
            Err(PageError::CursorQueryMismatch)
        );
        Ok(())
    }

    #[test]
    fn rejects_cursor_after_result_snapshot_changes() -> Result<(), String> {
        let first = PageRequest::from_parameters(&params(1, None), 1, 10)
            .map_err(|error| error.to_string())?;
        let (_, info) = first
            .paginate(vec!["first", "second"], &json!({"query": "same"}))
            .map_err(|error| error.to_string())?;
        let cursor = info
            .next_cursor
            .as_deref()
            .ok_or_else(|| "first page must have a cursor".to_string())?;
        let next = PageRequest::from_parameters(&params(1, Some(cursor)), 1, 10)
            .map_err(|error| error.to_string())?;

        assert_eq!(
            next.paginate(
                vec!["inserted", "first", "second"],
                &json!({"query": "same"})
            ),
            Err(PageError::CursorQueryMismatch)
        );
        Ok(())
    }

    #[test]
    fn applying_page_info_preserves_existing_truncation() {
        let mut result = ToolResult::success("preview").with_truncated(true);
        PageInfo {
            next_cursor: None,
            truncated: false,
            total_known: true,
            total: Some(1),
            returned: 1,
        }
        .apply_to(&mut result);

        assert!(result.truncated);
        assert_eq!(
            result.metadata.get("page.truncated").map(String::as_str),
            Some("false")
        );
    }

    #[test]
    fn continuation_cursor_is_visible_in_result_output() {
        let mut result = ToolResult::success("first page");
        PageInfo {
            next_cursor: Some("opaque-cursor".to_string()),
            truncated: true,
            total_known: true,
            total: Some(3),
            returned: 1,
        }
        .apply_to(&mut result);

        assert!(result.output.contains("[page]{"));
        assert!(result.output.contains("\"next_cursor\":\"opaque-cursor\""));
    }

    #[test]
    fn cursor_hex_pairs_decode_even_length_and_reject_remainder() -> Result<(), String> {
        let envelope = CursorEnvelope {
            version: CURSOR_VERSION,
            query_fingerprint: "fingerprint".to_string(),
            offset: 7,
        };
        let cursor = encode_cursor(&envelope).map_err(|error| error.to_string())?;
        let decoded = decode_cursor(&cursor).map_err(|error| error.to_string())?;
        assert_eq!(decoded.version, CURSOR_VERSION);
        assert_eq!(decoded.query_fingerprint, "fingerprint");
        assert_eq!(decoded.offset, 7);

        let with_remainder = format!("{cursor}f");
        assert!(matches!(
            decode_cursor(&with_remainder),
            Err(PageError::InvalidCursor(message)) if message == "cursor has an invalid length"
        ));
        Ok(())
    }
}
