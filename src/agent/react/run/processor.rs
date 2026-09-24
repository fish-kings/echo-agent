/*!
SSE 数据块处理 —— 将原始 LLM 流式数据块转换为 AgentEvent 事件序列。

这些都是静态方法（没有 `&self`）—— 是转换数据块的纯函数。
*/

use crate::agent::AgentEvent;
use crate::llm::types::{ChatCompletionChunk, FunctionCall, ToolCall as LlmToolCall};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/**
处理流式响应数据块，收集内容并返回事件列表。

`in_reasoning` 用于跟踪当前是否正在输出 reasoning_content（Qwen3/DeepSeek 的思考过程）。
第一次遇到 reasoning_content 时会发出 ThinkStart 事件；
reasoning 结束后第一次遇到 content 或 tool_calls 时会发出 ThinkEnd 事件。
*/
#[allow(clippy::type_complexity)]
pub(crate) fn process_stream_chunk(
    chunk: &ChatCompletionChunk,
    content_buffer: &mut String,
    reasoning_buffer: &mut String,
    tool_call_map: &mut HashMap<u32, (String, String, String)>,
    in_reasoning: &mut bool,
    emit_content_tokens: bool,
) -> Vec<AgentEvent> {
    let mut events = Vec::new();

    if let Some(choice) = chunk.choices.first() {
        // 处理 reasoning_content（Qwen3/DeepSeek 的思考过程）
        if let Some(reasoning) = &choice.delta.reasoning_content
            && !reasoning.is_empty()
        {
            reasoning_buffer.push_str(reasoning);
            if !*in_reasoning {
                *in_reasoning = true;
                events.push(AgentEvent::ThinkStart);
            }
            events.push(AgentEvent::Token(reasoning.clone()));
        }

        // reasoning 结束后第一次遇到 content 时，结束思考块
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            if *in_reasoning {
                *in_reasoning = false;
                events.push(AgentEvent::ThinkEnd {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                });
            }
            content_buffer.push_str(content);
            if emit_content_tokens {
                events.push(AgentEvent::Token(content.clone()));
            }
        }

        if let Some(delta_calls) = &choice.delta.tool_calls {
            if *in_reasoning {
                *in_reasoning = false;
                events.push(AgentEvent::ThinkEnd {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                });
            }
            for dc in delta_calls {
                let entry = tool_call_map
                    .entry(dc.index)
                    .or_insert_with(|| (String::new(), String::new(), String::new()));
                if let Some(id) = &dc.id
                    && !id.is_empty()
                {
                    entry.0 = id.clone();
                }
                if let Some(f) = &dc.function {
                    if let Some(name) = &f.name
                        && !name.is_empty()
                    {
                        entry.1 = name.clone();
                    }
                    if let Some(args) = &f.arguments {
                        entry.2.push_str(args);
                    }
                }
            }
        }
    }

    events
}

/**
解析流式工具调用参数，能容忍部分供应商（尤其是 DeepSeek 的 "fake-stream"）
在重新拼接参数片段时引入的尾部字符损坏问题。参见 vLLM issue #42878：
流式片段拼接后不一定能还原成非流式路径会产生的那份 JSON —— 常见症状是
多出尾部的 `}` / `]` 或空白字符。

策略：
  1. 先按原样解析。
  2. 解析失败时，裁掉尾部的 `}`、`]`、`,` 和空白字符后重试。
  3. 只有修复后仍然失败时才放弃（返回 `Err`）。

解析成功时返回 `(parsed_value, args_to_echo)`：`args_to_echo` 是我们
回传给供应商、写入 assistant 消息中的规范化字符串 —— 它是修复/重新
序列化后的结果，保证下一轮看到的是合法 JSON。
*/
fn parse_tool_args(args_str: &str) -> Result<(Value, String), serde_json::Error> {
    match serde_json::from_str::<Value>(args_str) {
        Ok(v) => Ok((v.clone(), v.to_string())),
        Err(original_err) => {
            /* 修复尝试：每次去掉一个尾部的多余字符（有次数上限）后重试。
            这能覆盖 DeepSeek 多出一个 `}` 的情况。 */
            let trimmed = args_str.trim();
            let mut attempt = trimmed.to_string();
            for _ in 0..8 {
                let last = attempt.chars().last();
                if matches!(
                    last,
                    Some('}') | Some(']') | Some(',') | Some(' ' | '\n' | '\t' | '\r')
                ) {
                    attempt.pop();
                    if let Ok(v) = serde_json::from_str::<Value>(&attempt) {
                        tracing::warn!(
                            raw_args = args_str,
                            repaired_args = %v,
                            "Repaired streaming tool-call arguments (stripped trailing characters from a provider fake-stream artifact)"
                        );
                        return Ok((v.clone(), v.to_string()));
                    }
                } else {
                    break;
                }
            }

            Err(original_err)
        }
    }
}

/// 将收集到的 tool_call_map 转换成结构化的工具调用列表
pub(crate) type ParsedToolStep = (String, String, Value);

/// 将 `tool_call_map` 转换为发送给模型的 `LlmToolCall` 列表，以及内部使用的
/// `ParsedToolStep` 列表；若发现同一批次中存在重复的 tool_call id，则返回错误。
pub(crate) fn build_tool_calls_from_map(
    tool_call_map: &HashMap<u32, (String, String, String)>,
) -> Result<(Vec<LlmToolCall>, Vec<ParsedToolStep>), String> {
    let mut sorted_indices: Vec<u32> = tool_call_map.keys().cloned().collect();
    sorted_indices.sort();

    let mut msg_tool_calls: Vec<LlmToolCall> = Vec::new();
    let mut steps: Vec<ParsedToolStep> = Vec::new();
    let mut seen_ids = HashSet::new();

    for idx in &sorted_indices {
        let Some((id, name, args_str)) = tool_call_map.get(idx) else {
            continue;
        };
        let canonical_id = if id.trim().is_empty() {
            format!("call_{}", uuid::Uuid::new_v4())
        } else {
            id.clone()
        };
        if !seen_ids.insert(canonical_id.clone()) {
            return Err(format!(
                "duplicate tool call id '{}' in one model batch",
                canonical_id
            ));
        }
        match parse_tool_args(args_str) {
            Ok((args, canonical_args)) => {
                /* 把规范化（修复后）的参数回传给供应商，
                确保 assistant 消息中始终携带合法 JSON。 */
                msg_tool_calls.push(LlmToolCall {
                    id: canonical_id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: canonical_args,
                    },
                });
                steps.push((canonical_id, name.clone(), args));
            }
            Err(e) => {
                tracing::warn!(
                    tool_name = %name,
                    tool_call_id = %id,
                    raw_args = %args_str,
                    error = %e,
                    "Failed to parse streaming tool-call arguments even after repair; dropping this tool call"
                );
                /* 重要：直接丢弃这个工具调用 —— 不要把它回传给供应商
                （损坏的参数会导致 HTTP 400 "Invalid assistant message"），
                也不要发出一个没有匹配 tool_result 的半截 tool_call
                （那同样会破坏下一轮对话）。丢弃它可以让模型在下一轮
                干净地重新发起调用。 */
            }
        }
    }

    Ok((msg_tool_calls, steps))
}

#[cfg(test)]
mod tests {
    use super::{build_tool_calls_from_map, parse_tool_args, process_stream_chunk};
    use crate::llm::types::ChatCompletionChunk;
    use std::collections::HashMap;

    #[test]
    fn parse_tool_args_valid_json_passes_through() -> Result<(), serde_json::Error> {
        let raw = r#"{"task": {"agent_role": "explorer"}}"#;
        let (val, echo) = parse_tool_args(raw)?;
        assert_eq!(val["task"]["agent_role"], "explorer");
        // 回显的参数是规范化重新序列化后的形式。
        assert_eq!(echo, val.to_string());
        Ok(())
    }

    #[test]
    fn parse_tool_args_repairs_deepseek_extra_trailing_brace() -> Result<(), serde_json::Error> {
        /* DeepSeek "fake-stream" 产生的问题：尾部多出一个 `}`。
        参见 vLLM issue #42878。 */
        let raw = r#"{"task": {"agent_role": "explorer"}}}"#;
        let (val, _echo) = parse_tool_args(raw)?;
        assert_eq!(val["task"]["agent_role"], "explorer");
        Ok(())
    }

    #[test]
    fn parse_tool_args_repairs_trailing_whitespace_after_brace() -> Result<(), serde_json::Error> {
        // DeepSeek 的另一种情况：合法 JSON 后面跟着尾部空白字符。
        let raw = "{\"task\": {\"agent_role\": \"explorer\"}}   \n";
        let (val, _echo) = parse_tool_args(raw)?;
        assert_eq!(val["task"]["agent_role"], "explorer");
        Ok(())
    }

    #[test]
    fn parse_tool_args_repairs_multiple_extra_braces() -> Result<(), serde_json::Error> {
        // 最坏情况：尾部有多个多余的 `}`。
        let raw = r#"{"task": {"agent_role": "explorer"}}}}}"#;
        let (val, _echo) = parse_tool_args(raw)?;
        assert_eq!(val["task"]["agent_role"], "explorer");
        Ok(())
    }

    #[test]
    fn parse_tool_args_genuinely_broken_returns_err() {
        // 根本不是一个 JSON 对象 —— 修复逻辑也无能为力。
        let raw = "not json at all";
        assert!(parse_tool_args(raw).is_err());
    }

    #[test]
    fn parse_tool_args_empty_object() -> Result<(), serde_json::Error> {
        let (val, echo) = parse_tool_args("{}")?;
        assert!(val.is_object());
        assert_eq!(echo, "{}");
        Ok(())
    }

    #[test]
    fn tool_call_ids_are_non_empty_and_shared_across_projections() -> Result<(), String> {
        let calls = HashMap::from([(0, (String::new(), "Read".to_string(), "{}".to_string()))]);
        let (messages, steps) = build_tool_calls_from_map(&calls)?;
        let message_id = messages.first().map(|call| call.id.as_str());
        let step_id = steps.first().map(|(id, _, _)| id.as_str());
        assert!(message_id.is_some_and(|id| !id.is_empty()));
        assert_eq!(message_id, step_id);
        Ok(())
    }

    #[test]
    fn duplicate_tool_call_ids_reject_the_entire_batch() {
        let calls = HashMap::from([
            (
                0,
                (
                    "duplicate".to_string(),
                    "Read".to_string(),
                    "{}".to_string(),
                ),
            ),
            (
                1,
                (
                    "duplicate".to_string(),
                    "Read".to_string(),
                    "{}".to_string(),
                ),
            ),
        ]);
        assert!(build_tool_calls_from_map(&calls).is_err());
    }

    #[test]
    fn process_stream_chunk_accumulates_reasoning_for_protocol_replay()
    -> Result<(), serde_json::Error> {
        let chunk: ChatCompletionChunk = serde_json::from_value(serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "先分析，再调用工具。"}
            }]
        }))?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = HashMap::new();
        let mut in_reasoning = false;

        let _events = process_stream_chunk(
            &chunk,
            &mut content,
            &mut reasoning,
            &mut calls,
            &mut in_reasoning,
            false,
        );

        assert_eq!(reasoning, "先分析，再调用工具。");
        assert!(in_reasoning);
        Ok(())
    }
}
</content>
