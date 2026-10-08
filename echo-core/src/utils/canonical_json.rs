//! 用于哈希与完整性契约的确定性 JSON 编码。

use serde::Serialize;

/**
序列化一个值，并在每一层递归地对对象的键排序。

显式排序可以保证序列化字节的稳定性，即便另一个消费者所编译的
`serde_json` 使用的是保留插入顺序的 map 实现也不受影响。
*/
pub fn canonical_json_bytes<T: Serialize + ?Sized>(
    value: &T,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    sort_value(&mut value);
    serde_json::to_vec(&value)
}

/**
递归地对 JSON 值中所有对象的键按字典序原地排序；数组元素保持原顺序，
其余标量类型（null/bool/数字/字符串）原样保留。
*/
fn sort_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                sort_value(item);
            }
        }
        serde_json::Value::Object(map) => {
            let mut entries = std::mem::take(map).into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, mut nested) in entries {
                sort_value(&mut nested);
                map.insert(key, nested);
            }
        }
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证不同插入顺序的嵌套对象经过规范化后产生相同的字节序列
    #[test]
    fn recursively_sorts_nested_object_keys() -> Result<(), serde_json::Error> {
        let forward: serde_json::Value =
            serde_json::from_str(r#"{"outer":{"alpha":1,"omega":2},"tail":3}"#)?;
        let reverse: serde_json::Value =
            serde_json::from_str(r#"{"tail":3,"outer":{"omega":2,"alpha":1}}"#)?;

        assert_eq!(
            canonical_json_bytes(&forward)?,
            canonical_json_bytes(&reverse)?
        );
        Ok(())
    }
}
