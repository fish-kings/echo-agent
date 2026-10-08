//! 基于 LLM 的 Critic（评审者）实现。

use crate::error::{ReactError, Result};
use crate::llm::types::Message;
use crate::llm::{ChatRequest, LlmClient, ResponseFormat};
use echo_core::agent::Critic;
use echo_core::agent::{Critique, CritiqueOutput, critique_output_schema};
use echo_core::retry::{RetryPolicy, with_retry_if};
use futures::future::BoxFuture;
use std::sync::Arc;
use tracing::{debug, info, warn};

/**
基于 LLM 的评估器。

使用大语言模型评估 Agent 输出的质量，返回结构化的 `Critique`。
复用 `LlmPlanner` 的模式：LLM 调用 + 结构化 JSON 输出 + 自动修复。
*/
pub struct LlmCritic {
    /// 用于评估调用的模型名称。
    model: String,
    /// 已就绪的 LLM 客户端。
    client: Arc<dyn LlmClient>,
    /// 评估用的系统 prompt。
    system_prompt: String,
    /// 判定通过所需的最低分数。
    pass_threshold: f64,
    /// provider 端用户 ID，用于隔离/缓存等场景。
    cache_user_id: Option<String>,
    /// 瞬时性 provider 失败时使用的重试策略。
    retry_policy: RetryPolicy,
}

impl LlmCritic {
    /**
    创建一个 LLM 评估器。

    # 参数
    * `client` —— 由所属 Agent 提供的、已经准备好的客户端

    # 默认配置
    * 系统 prompt：多维度质量评估专家（准确性、完整性、清晰度、有用性）
    * 通过阈值：7.0（分数 >= 7.0 视为通过）
    */
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self {
            model: client.model_name().to_string(),
            client,
            system_prompt: Self::default_system_prompt().to_string(),
            pass_threshold: 7.0,
            cache_user_id: None,
            retry_policy: RetryPolicy::default(),
        }
    }

    /// 自定义系统 prompt。
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// 设置通过阈值（0.0 - 10.0）。
    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self
    }

    /**
    设置 provider 端用户 ID，用于在支持该特性的 provider（例如 DeepSeek）上
    实现 KVCache/内容安全/调度隔离。
    */
    pub fn with_cache_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.cache_user_id = Some(user_id.into());
        self
    }

    /// 设置评估过程中应对瞬时性 provider 失败所使用的重试策略。
    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// 返回默认的评估系统 prompt（英文原文，直接发送给模型）。
    fn default_system_prompt() -> &'static str {
        "You are a strict quality evaluation expert. You need to evaluate the quality of the given response.\n\n\
        Evaluation dimensions:\n\
        1. Accuracy: Are the facts correct\n\
        2. Completeness: Does it cover all key points\n\
        3. Clarity: Is the expression clear and easy to understand\n\
        4. Usefulness: Does it provide valuable information\n\n\
        Scoring standards:\n\
        - 9.0-10.0: Excellent, almost flawless\n\
        - 7.0-8.9: Good, basically correct but with minor flaws\n\
        - 5.0-6.9: Mediocre, with noticeable deficiencies\n\
        - 0.0-4.9: Poor, contains serious errors\n\n\
        Please strictly return structured data according to the JSON Schema."
    }

    /// 将 LLM 响应解析为 CritiqueOutput。
    fn parse_critique_output(content: &str) -> Result<CritiqueOutput> {
        // 1. 直接解析
        if let Ok(output) = serde_json::from_str::<CritiqueOutput>(content) {
            return Ok(output);
        }

        // 2. 从 markdown 代码块中提取
        let json_str = crate::utils::json_parse::extract_json_from_markdown(content);
        if let Ok(output) = serde_json::from_str::<CritiqueOutput>(&json_str) {
            return Ok(output);
        }

        // 3. 自动修复
        Self::try_auto_fix(&json_str)
    }

    /// 尝试清洗并修复格式不规范的 JSON，修复失败时兜底返回默认评估结果。
    fn try_auto_fix(json_str: &str) -> Result<CritiqueOutput> {
        let fixed = crate::utils::json_parse::clean_json(json_str);

        match serde_json::from_str::<CritiqueOutput>(&fixed) {
            Ok(output) => {
                info!("Auto-fix succeeded for LLM critique output");
                Ok(output)
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse critique output");
                // 兜底：构造一个默认的未通过评估结果
                Ok(CritiqueOutput {
                    score: 5.0,
                    passed: false,
                    feedback: json_str.trim().to_string(),
                    suggestions: vec![],
                })
            }
        }
    }
}

impl LlmCritic {
    /**
    使用给定的 `response_format` 发起一次评审 LLM 调用。

    之所以单独抽取出来，是为了让主 `critique` 流程能在 provider 拒绝
    结构化输出时，去掉 `response_format` 重试一次（部分 OpenAI 兼容端点
    会对 `response_format: json_schema` 返回 HTTP 400，例如
    "This response_format type is unavailable now"）。这里只返回原始文本，
    JSON 解析相关的处理仍然留在 `parse_critique_output` 中。
    */
    async fn call_llm_once(
        &self,
        messages: &[Message],
        response_format: Option<ResponseFormat>,
    ) -> Result<String> {
        let response = self
            .client
            .chat(ChatRequest {
                messages: messages.to_vec(),
                temperature: Some(0.3),
                max_tokens: Some(2048u32),
                response_format,
                user_id: self.cache_user_id.clone(),
                ..Default::default()
            })
            .await?;

        Ok(response.content().unwrap_or_default())
    }

    /**
    使用框架的重试策略调用 provider，以应对瞬时性 LLM 失败。
    永久性错误仍然只尝试一次，以便鉴权、请求格式等问题能立即暴露出来。
    */
    async fn call_llm(
        &self,
        messages: &[Message],
        response_format: Option<ResponseFormat>,
    ) -> Result<String> {
        with_retry_if(
            &self.retry_policy,
            || self.call_llm_once(messages, response_format.clone()),
            crate::agent::react::is_retryable_llm_error,
        )
        .await
    }

    /**
    判断一条错误信息是否表明 provider 拒绝了 `response_format` 参数
    （因此值得尝试文本兜底方案）。

    匹配的是不支持结构化输出的 OpenAI 兼容端点常见措辞
    （DeepSeek/Qwen/GLM 等会返回 "response_format type is unavailable"
    的各种变体）。判定较为保守：只有信号明确时才触发兜底，
    以免掩盖无关的 API 错误。
    */
    fn is_response_format_unsupported(err: &str) -> bool {
        let lower = err.to_ascii_lowercase();
        lower.contains("response_format")
            && (lower.contains("unavailable")
                || lower.contains("unsupported")
                || lower.contains("not supported")
                || lower.contains("invalid"))
    }

    /// 判断当前模型是否应跳过结构化 `response_format`（例如 DeepSeek 系列）。
    fn should_skip_structured_response_format(&self) -> bool {
        let model = self.model.to_ascii_lowercase();
        model.starts_with("deepseek-")
    }

    /// 构造不带 `response_format` 的兜底消息，显式要求模型仅输出 JSON。
    fn fallback_messages(&self, user_content: &str) -> Vec<Message> {
        vec![
            Message::system(self.system_prompt.clone()),
            Message::user(format!(
                "{user_content}\n\n\
                 IMPORTANT: respond with ONLY a JSON object matching this schema, \
                 no markdown fences, no prose. Fields: score (0-10 number), \
                 passed (boolean), feedback (string), suggestions (array of strings)."
            )),
        ]
    }
}

impl Critic for LlmCritic {
    /// 调用 LLM 对给定任务与回答进行评估，返回结构化的 `Critique`。
    fn critique<'a>(
        &'a self,
        task: &'a str,
        answer: &'a str,
        context: &'a str,
    ) -> BoxFuture<'a, Result<Critique>> {
        Box::pin(async move {
            info!(model = %self.model, "LlmCritic: evaluating answer");

            let user_content = if context.is_empty() {
                format!(
                    "Original task:\n{}\n\nResponse to evaluate:\n{}",
                    task, answer
                )
            } else {
                format!(
                    "Original task:\n{}\n\nResponse to evaluate:\n{}\n\nAdditional context:\n{}",
                    task, answer, context
                )
            };

            let messages = vec![
                Message::system(self.system_prompt.clone()),
                Message::user(user_content.clone()),
            ];
            let response_format = Some(ResponseFormat::json_schema(
                "critique_output",
                critique_output_schema(),
            ));

            let content = if self.should_skip_structured_response_format() {
                debug!(
                    model = %self.model,
                    "critique: skipping response_format for provider without json_schema support"
                );
                self.call_llm(&self.fallback_messages(&user_content), None)
                    .await
                    .map_err(|e| ReactError::Other(format!("LLM critique call failed: {e}")))?
            } else {
                /* 首次尝试：通过 json_schema 使用结构化输出（更优——在
                provider 支持时能保证输出符合 schema）。 */
                match self.call_llm(&messages, response_format).await {
                    Ok(text) => text,
                    Err(e) if Self::is_response_format_unsupported(&e.to_string()) => {
                        /* provider 拒绝了结构化输出——去掉 response_format 重试一次。
                        系统 prompt 里已经要求输出 JSON；这里再追加一条明确指令，
                        让模型输出 `parse_critique_output`（支持 markdown/自动修复）
                        仍能处理的可解析 JSON。这样即便 provider 不支持
                        json_schema，评审功能依然可用。 */
                        warn!(
                            error = %e,
                            "critique: structured output unsupported, retrying as plain text"
                        );
                        self.call_llm(&self.fallback_messages(&user_content), None)
                            .await
                            .map_err(|e| {
                                ReactError::Other(format!("LLM critique call failed: {e}"))
                            })?
                    }
                    Err(e) => {
                        return Err(ReactError::Other(format!("LLM critique call failed: {e}")));
                    }
                }
            };

            debug!(response = %content, "LlmCritic raw response");

            let output = Self::parse_critique_output(&content)?;
            let mut critique: Critique = output.into();

            // 用阈值覆盖 LLM 自身给出的通过判断
            critique.passed = critique.score >= self.pass_threshold;

            info!(
                score = critique.score,
                passed = critique.passed,
                "LlmCritic: evaluation complete"
            );

            Ok(critique)
        })
    }

    /// 返回该 Critic 的名称。
    fn name(&self) -> &str {
        "llm_critic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::testing::MockLlmClient;
    use std::time::Duration;

    fn critic_for_model(model: &str) -> LlmCritic {
        LlmCritic::new(Arc::new(
            MockLlmClient::new().with_model_name(model.to_string()),
        ))
    }

    #[test]
    fn test_parse_critique_output_json() {
        let json = r#"{"score": 8.5, "passed": true, "feedback": "Accurate response", "suggestions": ["Could be more detailed"]}"#;
        let output = LlmCritic::parse_critique_output(json).unwrap();
        assert_eq!(output.score, 8.5);
        assert!(output.passed);
    }

    #[test]
    fn test_parse_critique_output_markdown() {
        let response = r#"```json
{"score": 6.0, "passed": false, "feedback": "Not complete enough", "suggestions": ["Add examples"]}
```"#;
        let output = LlmCritic::parse_critique_output(response).unwrap();
        assert_eq!(output.score, 6.0);
        assert!(!output.passed);
    }

    #[test]
    fn test_parse_critique_auto_fix() {
        let json = r#"{"score": 7.0, "passed": true, "feedback": "Good",}"#;
        let output = LlmCritic::parse_critique_output(json).unwrap();
        assert_eq!(output.score, 7.0);
    }

    #[test]
    fn test_parse_critique_fallback() {
        let text = "Unparseable text";
        let output = LlmCritic::parse_critique_output(text).unwrap();
        assert!(!output.passed); // 兜底结果：未通过
        assert_eq!(output.score, 5.0);
    }

    #[test]
    fn test_is_response_format_unsupported_matches_provider_phrasings() {
        // 来自用户 bug 报告的真实措辞（国产 OpenAI 兼容端点）。
        let real = "API error (status 400): This response_format type is unavailable now";
        assert!(LlmCritic::is_response_format_unsupported(real));
        // 各 provider 之间常见的变体措辞。
        assert!(LlmCritic::is_response_format_unsupported(
            "response_format unsupported"
        ));
        assert!(LlmCritic::is_response_format_unsupported(
            "response_format is not supported by this model"
        ));
        assert!(LlmCritic::is_response_format_unsupported(
            "invalid response_format parameter"
        ));
    }

    #[test]
    fn test_is_response_format_unsupported_rejects_unrelated_errors() {
        /* 无关的 API 错误不应触发兜底逻辑（否则会掩盖鉴权/配额/网络等
        真实故障）。 */
        assert!(!LlmCritic::is_response_format_unsupported(
            "401 unauthorized"
        ));
        assert!(!LlmCritic::is_response_format_unsupported("rate limited"));
        assert!(!LlmCritic::is_response_format_unsupported(
            "network timeout"
        ));
        assert!(!LlmCritic::is_response_format_unsupported(
            "insufficient quota"
        ));
    }

    #[test]
    fn deepseek_skips_structured_response_format() {
        let critic = critic_for_model("deepseek-v4-flash");
        assert!(critic.should_skip_structured_response_format());

        let other = critic_for_model("gpt-5.1");
        assert!(!other.should_skip_structured_response_format());
    }

    #[test]
    fn fallback_messages_request_json_only() -> std::result::Result<(), &'static str> {
        let critic = critic_for_model("deepseek-v4-flash");
        let messages = critic.fallback_messages("Evaluate this");
        assert_eq!(messages.len(), 2);
        let user_message = messages
            .get(1)
            .and_then(|m| m.content.as_text())
            .ok_or("expected fallback user message text")?;
        assert!(user_message.contains("ONLY a JSON object"));
        assert!(user_message.contains("score (0-10 number)"));
        Ok(())
    }

    #[tokio::test]
    async fn critique_uses_the_prepared_client_transport() -> Result<()> {
        let client = Arc::new(
            MockLlmClient::new()
                .with_model_name("prepared-model")
                .with_response(r#"{"score":8.0,"passed":true,"feedback":"ok","suggestions":[]}"#),
        );
        let critic = LlmCritic::new(client.clone());

        let critique = critic.critique("task", "answer", "").await?;

        assert!(critique.passed);
        assert_eq!(client.call_count(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn critique_retries_transient_server_error() -> Result<()> {
        let client = Arc::new(
            MockLlmClient::new()
                .with_error(ReactError::Llm(Box::new(LlmError::ApiError {
                    status: 502,
                    message: "Upstream request failed".to_string(),
                })))
                .with_response(r#"{"score":8.0,"passed":true,"feedback":"ok","suggestions":[]}"#),
        );
        let critic =
            LlmCritic::new(client.clone()).with_retry_policy(RetryPolicy::new(1, Duration::ZERO));

        let critique = critic.critique("task", "answer", "").await?;

        assert!(critique.passed);
        assert_eq!(client.call_count(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn critique_does_not_retry_permanent_api_error() {
        let client = Arc::new(
            MockLlmClient::new()
                .with_error(ReactError::Llm(Box::new(LlmError::ApiError {
                    status: 401,
                    message: "Unauthorized".to_string(),
                })))
                .with_response(r#"{"score":8.0,"passed":true,"feedback":"ok","suggestions":[]}"#),
        );
        let critic =
            LlmCritic::new(client.clone()).with_retry_policy(RetryPolicy::new(3, Duration::ZERO));

        let result = critic.critique("task", "answer", "").await;

        assert!(result.is_err());
        assert_eq!(client.call_count(), 1);
    }
}
