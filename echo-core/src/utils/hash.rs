//! 确定性哈希工具。

/// FNV-1a 64 位哈希 —— 跨进程重启保持确定性。
///
/// 用于基于内容的去重场景：相同输入必须始终产生相同输出，
/// 与进程生命周期无关。
pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证相同输入始终产生相同哈希值
    #[test]
    fn test_fnv1a_deterministic() {
        let a = fnv1a_64(b"hello world");
        let b = fnv1a_64(b"hello world");
        assert_eq!(a, b);
    }

    // 验证不同输入产生不同哈希值
    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }
}
