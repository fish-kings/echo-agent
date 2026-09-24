//! 针对读取边界可能切断多字节标量值的字节流做增量 UTF-8 解码。

/// 默认的流式分片字节数上限。
const DEFAULT_STREAM_CHUNK_BYTES: usize = 16 * 1024;

/// 有状态解码器，在多次读取之间保留不完整的 UTF-8 后缀。
///
/// 非法字节序列会被替换为 U+FFFD，而不完整的后缀会一直缓冲，直到下一次
/// `push` 或 `finish` 调用。返回的分片除非单个标量值本身超过限制
/// （对于正数限制下的合法 UTF-8 不会发生这种情况），否则不会超过配置的
/// 字节数上限。
#[derive(Debug)]
pub struct IncrementalUtf8Decoder {
    /// 尚未成功解码、缓冲等待下一次输入的字节。
    pending: Vec<u8>,
    /// 单次返回分片允许的最大字节数。
    max_chunk_bytes: usize,
}

impl Default for IncrementalUtf8Decoder {
    /// 使用默认的分片字节数上限创建解码器。
    fn default() -> Self {
        Self::new(DEFAULT_STREAM_CHUNK_BYTES)
    }
}

impl IncrementalUtf8Decoder {
    /// 创建一个新的解码器，`max_chunk_bytes` 至少为 1。
    pub fn new(max_chunk_bytes: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_chunk_bytes: max_chunk_bytes.max(1),
        }
    }

    /// 输入一段新的字节数据，返回可以解码出的完整字符串分片；
    /// 不完整的多字节序列会被缓冲，留待后续调用继续解码。
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    output.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid_len = error.valid_up_to();
                    if let Some(valid_bytes) = self.pending.get(..valid_len)
                        && let Ok(valid) = std::str::from_utf8(valid_bytes)
                    {
                        output.push_str(valid);
                    }
                    match error.error_len() {
                        Some(invalid_len) => {
                            output.push('\u{fffd}');
                            let consumed = valid_len.saturating_add(invalid_len);
                            self.pending.drain(..consumed.min(self.pending.len()));
                        }
                        None => {
                            self.pending.drain(..valid_len.min(self.pending.len()));
                            break;
                        }
                    }
                }
            }
        }
        split_utf8_chunks(output, self.max_chunk_bytes)
    }

    /// 结束解码，将缓冲区中剩余的字节以“有损”方式（无效序列替换为
    /// U+FFFD）转换为字符串并清空缓冲区；若缓冲区为空则返回 `None`。
    pub fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let output = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        Some(output)
    }
}

/// 将合法的 UTF-8 文本切分为若干字节数受限的分片，且不会从字符中间切开。
pub fn split_utf8_chunks(text: String, max_chunk_bytes: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let max_chunk_bytes = max_chunk_bytes.max(1);
    if text.len() <= max_chunk_bytes {
        return vec![text];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if current.len().saturating_add(character.len_utf8()) > max_chunk_bytes
            && !current.is_empty()
        {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(character);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证跨多次读取被切断的字符能够被正确保留并拼接
    #[test]
    fn preserves_a_scalar_split_across_reads() {
        let mut decoder = IncrementalUtf8Decoder::new(16);
        assert!(decoder.push(&[0xe4, 0xb8]).is_empty());
        assert_eq!(decoder.push(&[0xad]), vec!["中".to_string()]);
        assert!(decoder.finish().is_none());
    }

    // 验证分片只会发生在字符边界上
    #[test]
    fn chunks_only_at_character_boundaries() {
        let mut decoder = IncrementalUtf8Decoder::new(5);
        let chunks = decoder.push("中文ab".as_bytes());
        assert_eq!(chunks, vec!["中".to_string(), "文ab".to_string()]);
    }

    // 验证非法字节序列被替换为 U+FFFD，且不完整后缀会在 finish 时被冲出
    #[test]
    fn replaces_invalid_sequences_and_flushes_incomplete_suffix() {
        let mut decoder = IncrementalUtf8Decoder::new(16);
        assert_eq!(decoder.push(&[b'a', 0xff]), vec!["a\u{fffd}".to_string()]);
        assert!(decoder.push(&[0xf0, 0x9f]).is_empty());
        assert_eq!(decoder.finish(), Some("\u{fffd}".to_string()));
    }
}
