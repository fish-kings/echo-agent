/*!
与具体供应商无关的 LLM 客户端运行时配置。

供应商目录、凭据和模型选择都归使用方应用所有。本模块只负责把显式的
供应商/模型约定转换为具体的线上协议客户端，并校验请求是否符合模型能力。
*/

use echo_core::error::{ConfigError, LlmError, Result};
use echo_core::llm::capabilities::resolve_thinking_profile;
use echo_core::llm::types::{ContentPart, Message, MessageContent};
use echo_core::llm::{LlmApiProtocol, LlmTimeouts, ModelInputModality, ThinkingProtocol};
use serde::{Deserialize, Serialize};

/**
针对某个显式协议，解析供应商的 API 根地址或完整端点。

若传入的是可识别的完整端点：协议一致时原样保留；
在同一供应商下选择了不同协议时，则替换为对应协议的端点路径。
*/
pub fn resolve_protocol_endpoint(base_url: &str, protocol: LlmApiProtocol) -> Result<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return Err(ConfigError::ConfigFileError(
            "provider base_url must not be empty".to_string(),
        )
        .into());
    }

    let mut url = url::Url::parse(trimmed).map_err(|error| {
        ConfigError::ConfigFileError(format!("invalid provider base_url '{trimmed}': {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(ConfigError::ConfigFileError(format!(
            "provider base_url must use http or https: '{trimmed}'"
        ))
        .into());
    }

    let existing_protocol = LlmApiProtocol::try_from_endpoint(url.as_str());
    if existing_protocol == Some(protocol) {
        return Ok(url.to_string());
    }

    let segments_to_remove = match existing_protocol {
        Some(LlmApiProtocol::ChatCompletions) => 2,
        Some(LlmApiProtocol::Responses | LlmApiProtocol::Anthropic) => 1,
        None => 0,
    };
    let mut segments = url.path_segments_mut().map_err(|()| {
        ConfigError::ConfigFileError(format!(
            "provider base_url cannot be used as an API root: '{trimmed}'"
        ))
    })?;
    segments.pop_if_empty();
    for _ in 0..segments_to_remove {
        segments.pop();
    }
    for segment in protocol.endpoint_path().split('/') {
        segments.push(segment);
    }
    drop(segments);
    Ok(url.to_string())
}

/// 由应用注入的、已完全解析的模型配置。
#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// 应用定义的稳定供应商 ID，用于模型能力策略。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_name: Option<String>,
    /// 该模型所选用的线上协议。
    pub api_protocol: LlmApiProtocol,
    /// [`Self::api_protocol`] 对应的完整端点 URL。
    pub base_url: String,
    /// 已解析的 API 凭据；本地供应商可以为空。
    pub api_key: String,
    /// 发送给供应商的模型名称。
    pub model: String,
    /// 该模型接受的输入类型，默认仅纯文本。
    #[serde(default = "ModelInputModality::text_only")]
    pub input_modalities: Vec<ModelInputModality>,
    /// 由运行时约定统一解析出的思考（thinking）线上方言。
    #[serde(default)]
    pub thinking_protocol: ThinkingProtocol,
    /// 所有供应商传输层共用的请求与流式超时策略。
    #[serde(default)]
    pub timeouts: LlmTimeouts,
}

impl std::fmt::Debug for LlmConfig {
    /// 输出配置信息，`api_key` 以占位符脱敏。
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlmConfig")
            .field("provider_name", &self.provider_name)
            .field("api_protocol", &self.api_protocol)
            .field("base_url", &self.base_url)
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("input_modalities", &self.input_modalities)
            .field("thinking_protocol", &self.thinking_protocol)
            .field("timeouts", &self.timeouts)
            .finish()
    }
}

impl LlmConfig {
    /// 根据显式的供应商约定构建运行时模型配置。
    pub fn for_provider(
        provider_name: impl Into<String>,
        base_url: impl AsRef<str>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        api_protocol: LlmApiProtocol,
    ) -> Result<Self> {
        let provider_name = provider_name.into();
        let model = model.into();
        if model.trim().is_empty() {
            return Err(ConfigError::MissingConfig(
                "model".to_string(),
                "model name must not be empty".to_string(),
            )
            .into());
        }
        let base_url = resolve_protocol_endpoint(base_url.as_ref(), api_protocol)?;
        let thinking_protocol =
            resolve_thinking_profile(&provider_name, &model, api_protocol, Some(&base_url))
                .protocol;
        Ok(Self {
            provider_name: (!provider_name.trim().is_empty()).then_some(provider_name),
            api_protocol,
            base_url,
            api_key: api_key.into(),
            model,
            input_modalities: ModelInputModality::text_only(),
            thinking_protocol,
            timeouts: LlmTimeouts::default(),
        })
    }

    /// 设置具体模型可接受的输入模态。
    pub fn with_input_modalities(mut self, input_modalities: Vec<ModelInputModality>) -> Self {
        self.input_modalities = normalize_input_modalities(input_modalities);
        self
    }

    /// 设置具体供应商客户端使用的超时策略。
    pub fn with_timeouts(mut self, timeouts: LlmTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// 构建 [`Self::api_protocol`] 所选协议对应的客户端。
    pub fn build_client(&self) -> Result<Box<dyn echo_core::llm::LlmClient>> {
        match self.api_protocol {
            LlmApiProtocol::Responses => Ok(Box::new(super::responses::ResponsesClient::new(
                self.clone(),
            )?)),
            LlmApiProtocol::ChatCompletions => {
                Ok(Box::new(super::openai::OpenAiClient::new(self.clone())?))
            }
            LlmApiProtocol::Anthropic => Ok(Box::new(
                super::anthropic::AnthropicClient::with_base_url(
                    &self.base_url,
                    &self.api_key,
                    &self.model,
                )
                .with_input_modalities(self.input_modalities.clone())
                .with_timeouts(self.timeouts),
            )),
        }
    }

    /// 校验消息中的输入模态是否都被当前模型支持。
    pub(crate) fn validate_input_modalities(&self, messages: &[Message]) -> Result<()> {
        validate_model_input_modalities(&self.model, &self.input_modalities, messages)
    }
}

/// 规范化输入模态列表：始终包含文本，并去除重复项。
fn normalize_input_modalities(
    input_modalities: Vec<ModelInputModality>,
) -> Vec<ModelInputModality> {
    let mut normalized = ModelInputModality::text_only();
    for modality in input_modalities {
        if !normalized.contains(&modality) {
            normalized.push(modality);
        }
    }
    normalized
}

/// 校验消息中的图片、音频、视频等附件是否都在模型允许的输入模态内。
pub(crate) fn validate_model_input_modalities(
    model: &str,
    input_modalities: &[ModelInputModality],
    messages: &[Message],
) -> Result<()> {
    for part in messages
        .iter()
        .filter_map(|message| match &message.content {
            MessageContent::Parts(parts) => Some(parts.as_slice()),
            MessageContent::Text(_) | MessageContent::Empty => None,
        })
        .flatten()
    {
        let modality = match part {
            ContentPart::ImageUrl { .. } => Some(ModelInputModality::Image),
            ContentPart::File { name, .. } => file_input_modality(name),
            ContentPart::ResourceLink { .. } => None,
            ContentPart::Text { .. } => None,
        };
        if let Some(modality) = modality.filter(|value| !input_modalities.contains(value)) {
            let label = match modality {
                ModelInputModality::Text => "text",
                ModelInputModality::Image => "image",
                ModelInputModality::Audio => "audio",
                ModelInputModality::Video => "video",
            };
            return Err(LlmError::InvalidResponse(format!(
                "model '{model}' is not configured for {label} input"
            ))
            .into());
        }
    }
    Ok(())
}

/// 根据文件扩展名推断附件的输入模态；无法识别时返回 `None`。
fn file_input_modality(name: &str) -> Option<ModelInputModality> {
    let extension = std::path::Path::new(name)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)?;
    match extension.as_str() {
        "apng" | "avif" | "bmp" | "gif" | "heic" | "heif" | "jpeg" | "jpg" | "png" | "tif"
        | "tiff" | "webp" => Some(ModelInputModality::Image),
        "aac" | "aiff" | "alac" | "flac" | "m4a" | "mp3" | "ogg" | "opus" | "wav" => {
            Some(ModelInputModality::Audio)
        }
        "avi" | "m4v" | "mkv" | "mov" | "mp4" | "mpeg" | "mpg" | "webm" => {
            Some(ModelInputModality::Video)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证协议端点解析支持单一根地址对应多种协议。
    #[test]
    fn protocol_endpoint_resolution_supports_one_root_with_multiple_protocols()
    -> std::result::Result<(), String> {
        let root = "https://gateway.example/api/v1?tenant=eko";
        assert_eq!(
            resolve_protocol_endpoint(root, LlmApiProtocol::Responses)
                .map_err(|error| error.to_string())?,
            "https://gateway.example/api/v1/responses?tenant=eko"
        );
        assert_eq!(
            resolve_protocol_endpoint(root, LlmApiProtocol::ChatCompletions)
                .map_err(|error| error.to_string())?,
            "https://gateway.example/api/v1/chat/completions?tenant=eko"
        );
        assert_eq!(
            resolve_protocol_endpoint(root, LlmApiProtocol::Anthropic)
                .map_err(|error| error.to_string())?,
            "https://gateway.example/api/v1/messages?tenant=eko"
        );
        Ok(())
    }

    /// 验证提供商配置默认仅文本能力，并保留显式声明的能力。
    #[test]
    fn provider_config_defaults_to_text_and_preserves_explicit_capabilities()
    -> std::result::Result<(), String> {
        let text = LlmConfig::for_provider(
            "custom",
            "https://gateway.example/v1",
            "",
            "text-model",
            LlmApiProtocol::ChatCompletions,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(text.input_modalities, ModelInputModality::text_only());
        assert_eq!(text.timeouts, LlmTimeouts::default());

        let timeouts =
            LlmTimeouts::default().with_first_chunk_timeout(std::time::Duration::from_secs(15));
        let multimodal = text
            .with_input_modalities(vec![
                ModelInputModality::Image,
                ModelInputModality::Audio,
                ModelInputModality::Video,
            ])
            .with_timeouts(timeouts);
        assert_eq!(
            multimodal.input_modalities,
            ModelInputModality::all_supported()
        );
        assert_eq!(multimodal.timeouts, timeouts);
        Ok(())
    }

    /// 验证提供商配置无需用户字段即可解析 thinking 设置。
    #[test]
    fn provider_config_resolves_thinking_without_user_fields() -> std::result::Result<(), String> {
        let config = LlmConfig::for_provider(
            "openai",
            "https://api.openai.com/v1",
            "test-key",
            "gpt-5.6-sol",
            LlmApiProtocol::Responses,
        )
        .map_err(|error| error.to_string())?;
        assert_eq!(
            config.thinking_protocol,
            ThinkingProtocol::OpenaiReasoningEffort
        );
        Ok(())
    }

    /// 验证纯文本模型会拒绝图片、音频和视频输入。
    #[test]
    fn text_only_model_rejects_image_audio_and_video() {
        let mut image = Message::user(String::new());
        image.content = MessageContent::Parts(vec![ContentPart::ImageUrl {
            image_url: echo_core::llm::types::ImageUrl {
                url: "data:image/png;base64,AA==".to_string(),
                detail: None,
            },
        }]);
        assert!(
            validate_model_input_modalities(
                "text-model",
                &ModelInputModality::text_only(),
                &[image],
            )
            .is_err()
        );

        for (name, modality) in [("meeting.mp3", "audio"), ("demo.mp4", "video")] {
            let mut message = Message::user(String::new());
            message.content = MessageContent::Parts(vec![ContentPart::File {
                name: name.to_string(),
                content: String::new(),
            }]);
            let error = validate_model_input_modalities(
                "text-model",
                &ModelInputModality::text_only(),
                &[message],
            )
            .err()
            .map(|value| value.to_string())
            .unwrap_or_default();
            assert!(error.contains(modality));
        }
    }
}
