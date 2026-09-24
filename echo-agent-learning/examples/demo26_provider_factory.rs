/*! demo26_provider_factory.rs - 显式的 provider/model 契约示例。

一个应用自定义的 provider 连接可以承载多个模型。每个模型可以
选择自己的通信协议与输入能力。
*/

use echo_agent::prelude::*;

/// 演示为同一个 provider 连接下的多个模型分别配置协议与输入模态。
fn main() -> echo_agent::error::Result<()> {
    let provider_id = "company-gateway";
    let api_root = "https://gateway.example/v1";
    let api_key = "example-key";

    let text_model = LlmConfig::for_provider(
        provider_id,
        api_root,
        api_key,
        "reasoning-model",
        LlmApiProtocol::Responses,
    )?;
    let vision_model = LlmConfig::for_provider(
        provider_id,
        api_root,
        api_key,
        "vision-model",
        LlmApiProtocol::ChatCompletions,
    )?
    .with_input_modalities(vec![ModelInputModality::Text, ModelInputModality::Image]);
    let audio_video_model = LlmConfig::for_provider(
        provider_id,
        api_root,
        api_key,
        "media-model",
        LlmApiProtocol::Anthropic,
    )?
    .with_input_modalities(vec![
        ModelInputModality::Text,
        ModelInputModality::Audio,
        ModelInputModality::Video,
    ]);

    for config in [text_model, vision_model, audio_video_model] {
        let client = config.build_client()?;
        println!(
            "provider={} model={} protocol={:?} capabilities={:?}",
            config.provider_name.as_deref().unwrap_or("custom"),
            client.model_name(),
            config.api_protocol,
            config.input_modalities,
        );
    }

    Ok(())
}
