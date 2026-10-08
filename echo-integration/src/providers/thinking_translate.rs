/*!
 * 共享的 thinking 配置 → OpenAI 兼容请求字段转换。
 *
 * 供 OpenAI、Azure、Gemini（OpenAI 兼容端点）、Default 与 DeepSeek
 * 等 LLM 客户端使用。Anthropic 有自己基于内容块的协议
 * （`thinking.budget_tokens` / `effort` / adaptive），在 `anthropic.rs`
 * 中转换；Ollama 使用其原生的 `think` 字段。
 *
 * 所有协议均已对照各厂商官方 API 文档核实（2026 年年中）：
 * - OpenAI / DeepSeek / Azure / Gemini 兼容 → `reasoning_effort`
 * - Qwen3 → `enable_thinking` + `thinking_budget`
 * - Kimi 2.6 → `thinking:{type:"enabled"|"disabled"}`
 * - GLM-5.2+ → `reasoning_effort`（+ `thinking.type`）
 */

use echo_core::llm::types::ThinkingTypeBlock;
use echo_core::llm::{LlmApiProtocol, ThinkingConfig, ThinkingLevel, ThinkingProtocol};
use tracing::warn;

/// 由 [`ThinkingConfig`] 推导出的 OpenAI 兼容请求字段。
#[derive(Default, Debug)]
pub struct OpenAiCompatThinking {
    /// `reasoning_effort`（OpenAI / DeepSeek / Azure / Gemini 兼容 / GLM-5.x）。
    pub reasoning_effort: Option<String>,
    /// `enable_thinking`（Qwen3）。
    pub enable_thinking: Option<bool>,
    /// `thinking_budget`（Qwen3）。
    pub thinking_budget: Option<u32>,
    /// `thinking:{type}`（Kimi 2.6 的开关，以及 GLM-5.2+ 的配套字段）。
    pub thinking_type: Option<ThinkingTypeBlock>,
    /// Ollama 的 `think` 字段：布尔值或强度字符串。
    pub ollama_think: Option<serde_json::Value>,
    /// 是否去掉 `temperature`（o 系列 / GPT-5 推理模型会拒绝该参数）。
    pub drop_temperature: bool,
}

/**
 * 根据模型的 [`ThinkingProtocol`]，将
 * [`ChatRequest::thinking`](echo_core::llm::ChatRequest::thinking)
 * 转换为 OpenAI 兼容的请求字段。
 *
 * 当 `thinking` 为 `None` 或模型不支持任何 thinking 协议时，
 * 所有字段均为 `None`，并保留 `temperature`。
 */
pub fn translate_thinking_openai_compat(
    model_name: &str,
    api_protocol: LlmApiProtocol,
    thinking_protocol: ThinkingProtocol,
    thinking: &Option<ThinkingConfig>,
) -> OpenAiCompatThinking {
    match (thinking, thinking_protocol) {
        (None, _)
        | (Some(_), ThinkingProtocol::None)
        | (Some(_), ThinkingProtocol::ModelManaged) => OpenAiCompatThinking::default(),
        (Some(_), ThinkingProtocol::AnthropicAdaptive) => {
            warn!(
                model = model_name,
                "thinking config ignored: model uses adaptive thinking (no request field)"
            );
            OpenAiCompatThinking::default()
        }
        (Some(cfg), ThinkingProtocol::DeepseekReasoningEffort) => {
            /*
             * DeepSeek V4 接受 `low`/`high`/`max`；超出该集合的语义级别
             * 会在序列化前被钳制到最近的取值。
             *
             * DeepSeek 同时支持 `thinking:{type:"enabled"|"disabled"}` 作为开关。
             * 当用户显式关闭 thinking 时，必须只发送 `thinking:{type:"disabled"}`
             * 而不带 reasoning_effort——因为即便是 `reasoning_effort:"low"`
             * 也会触发思考循环（服务端会把 low 映射为 high）。启用模式下，
             * 则同时发送 `reasoning_effort` 与 `thinking:{type:"enabled"}`，
             * 以符合官方 API 规范。
             */
            if api_protocol == LlmApiProtocol::Responses {
                return OpenAiCompatThinking {
                    reasoning_effort: Some(match cfg {
                        ThinkingConfig::Disabled | ThinkingConfig::Level(ThinkingLevel::None) => {
                            "none".to_string()
                        }
                        _ => match cfg.to_reasoning_effort().unwrap_or("high") {
                            "max" | "xhigh" => "max".to_string(),
                            "low" => "low".to_string(),
                            _ => "high".to_string(),
                        },
                    }),
                    drop_temperature: true,
                    ..Default::default()
                };
            }
            match cfg {
                ThinkingConfig::Disabled | ThinkingConfig::Level(ThinkingLevel::None) => {
                    OpenAiCompatThinking {
                        thinking_type: Some(ThinkingTypeBlock {
                            block_type: "disabled".to_string(),
                        }),
                        drop_temperature: false,
                        ..Default::default()
                    }
                }
                _ => {
                    let raw = cfg.to_reasoning_effort().unwrap_or("high");
                    let clamped = match raw {
                        "max" | "xhigh" => "max",
                        "low" => "low",
                        _ => "high",
                    };
                    OpenAiCompatThinking {
                        reasoning_effort: Some(clamped.to_string()),
                        thinking_type: Some(ThinkingTypeBlock {
                            block_type: "enabled".to_string(),
                        }),
                        drop_temperature: true,
                        ..Default::default()
                    }
                }
            }
        }
        (Some(cfg), ThinkingProtocol::OpenaiReasoningEffort) => OpenAiCompatThinking {
            reasoning_effort: cfg.to_reasoning_effort().map(str::to_string),
            drop_temperature: true,
            ..Default::default()
        },
        (Some(cfg), ThinkingProtocol::GlmReasoningEffort) => {
            // GLM-5.x：reasoning_effort + thinking.type 开关（enabled）。
            OpenAiCompatThinking {
                reasoning_effort: cfg.to_glm_reasoning_effort().map(str::to_string),
                thinking_type: Some(ThinkingTypeBlock {
                    block_type: cfg.to_glm_thinking_type().to_string(),
                }),
                ..Default::default()
            }
        }
        (Some(cfg), ThinkingProtocol::EnableThinkingFlag) => {
            // Qwen3：enable_thinking + 可选的 thinking_budget。
            let budget = if let ThinkingConfig::BudgetTokens(n) = cfg {
                Some(*n)
            } else {
                None
            };
            OpenAiCompatThinking {
                enable_thinking: Some(cfg.to_enable_thinking()),
                thinking_budget: budget,
                ..Default::default()
            }
        }
        (Some(cfg), ThinkingProtocol::ThinkingType) => OpenAiCompatThinking {
            thinking_type: Some(ThinkingTypeBlock {
                block_type: cfg.to_glm_thinking_type().to_string(),
            }),
            ..Default::default()
        },
        (Some(cfg), ThinkingProtocol::OllamaThink) => {
            let think = match cfg {
                ThinkingConfig::Disabled | ThinkingConfig::Level(ThinkingLevel::None) => {
                    serde_json::Value::Bool(false)
                }
                ThinkingConfig::Level(level) => serde_json::Value::String(
                    match level {
                        ThinkingLevel::None => "none",
                        ThinkingLevel::Minimal => "minimal",
                        ThinkingLevel::Low => "low",
                        ThinkingLevel::Medium => "medium",
                        ThinkingLevel::High => "high",
                        ThinkingLevel::Xhigh => "xhigh",
                        ThinkingLevel::Max => "max",
                    }
                    .to_string(),
                ),
                ThinkingConfig::BudgetTokens(_) => serde_json::Value::Bool(true),
            };
            OpenAiCompatThinking {
                ollama_think: Some(think),
                ..Default::default()
            }
        }
        (Some(_), ThinkingProtocol::AnthropicEffort)
        | (Some(_), ThinkingProtocol::AnthropicThinkingBudget) => {
            // 非 OpenAI 兼容协议；优雅降级。
            OpenAiCompatThinking::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::llm::ThinkingLevel;
    use echo_core::llm::capabilities::{ModelProfile, ProviderCapabilities};

    /// 测试辅助：按模型与提供商推导 thinking 协议后执行转换。
    fn translate(
        model: &str,
        provider: &str,
        thinking: Option<ThinkingConfig>,
    ) -> OpenAiCompatThinking {
        let protocol =
            ModelProfile::new(model, provider, ProviderCapabilities::openai_compatible())
                .thinking_protocol;
        translate_thinking_openai_compat(
            model,
            LlmApiProtocol::ChatCompletions,
            protocol,
            &thinking,
        )
    }

    /// GPT-5 的 High 级别映射为 reasoning_effort 并去掉 temperature。
    #[test]
    fn gpt5_high_maps_to_reasoning_effort_and_drops_temperature() {
        let r = translate(
            "gpt-5.6-sol",
            "openai",
            Some(ThinkingConfig::Level(ThinkingLevel::High)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("high"));
        assert!(r.enable_thinking.is_none());
        assert!(r.thinking_budget.is_none());
        assert!(r.thinking_type.is_none());
        assert!(
            r.drop_temperature,
            "GPT-5 reasoning models must drop temperature (would 400)"
        );
    }

    /// Gemini 的 Minimal 级别映射为 reasoning_effort。
    #[test]
    fn gemini_minimal_maps_to_reasoning_effort() {
        let r = translate(
            "gemini-3-pro",
            "google",
            Some(ThinkingConfig::Level(ThinkingLevel::Minimal)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("minimal"));
        assert!(r.drop_temperature);
    }

    /// OpenAI 的 Xhigh 级别原样映射。
    #[test]
    fn openai_xhigh_effort() {
        let r = translate(
            "gpt-5.6-sol",
            "openai",
            Some(ThinkingConfig::Level(ThinkingLevel::Xhigh)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("xhigh"));
    }

    /// OpenAI 的 Max 级别不会被折叠为 xhigh。
    #[test]
    fn openai_max_is_not_collapsed_to_xhigh() {
        let r = translate(
            "gpt-5.6-sol",
            "openai",
            Some(ThinkingConfig::Level(ThinkingLevel::Max)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("max"));
    }

    /// DeepSeek 同时发送 reasoning_effort 与 thinking.type=enabled。
    #[test]
    fn deepseek_uses_reasoning_effort_and_thinking_type_enabled() {
        /*
         * 按官方 API 规范，DeepSeek-V3.2+ 同时使用 reasoning_effort 与
         * thinking:{type:"enabled"}。thinking.type 是开关，
         * reasoning_effort 控制强度（仅 high/max）。
         */
        let r = translate(
            "deepseek-v4",
            "deepseek",
            Some(ThinkingConfig::Level(ThinkingLevel::High)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("high"));
        assert!(
            r.thinking_type.is_some(),
            "DeepSeek must send thinking.type alongside reasoning_effort"
        );
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled"),
            "thinking.type must be 'enabled' for high effort"
        );
        assert!(r.enable_thinking.is_none());
        assert!(r.drop_temperature);
    }

    /// DeepSeek 关闭时仅发送 thinking.type=disabled。
    #[test]
    fn deepseek_disabled_sends_thinking_type_disabled_no_reasoning_effort() {
        /*
         * 关闭 thinking 时，DeepSeek 必须只使用 thinking:{type:"disabled"}，
         * 不带 reasoning_effort。发送任何 reasoning_effort 值（即便是 "low"）
         * 都会触发思考循环，因为服务端会把 low 映射为 high。
         */
        let r = translate(
            "deepseek-v4-pro",
            "deepseek",
            Some(ThinkingConfig::Disabled),
        );
        assert!(
            r.reasoning_effort.is_none(),
            "DeepSeek disabled: must NOT send reasoning_effort (any value engages thinking)"
        );
        assert!(
            r.thinking_type.is_some(),
            "DeepSeek disabled: must send thinking.type"
        );
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("disabled"),
            "thinking.type must be 'disabled'"
        );
        assert!(
            !r.drop_temperature,
            "non-thinking mode must keep temperature"
        );
    }

    /// DeepSeek 在 Responses 协议下关闭时使用 reasoning_effort=none。
    #[test]
    fn deepseek_responses_disabled_uses_reasoning_none() {
        let r = translate_thinking_openai_compat(
            "deepseek-v4",
            LlmApiProtocol::Responses,
            ThinkingProtocol::DeepseekReasoningEffort,
            &Some(ThinkingConfig::Level(ThinkingLevel::None)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("none"));
        assert!(r.thinking_type.is_none());
    }

    /// DeepSeek 的 Xhigh 被钳制为 max。
    #[test]
    fn deepseek_xhigh_clamps_to_max_with_thinking_enabled() {
        let r = translate(
            "deepseek-v4-pro",
            "deepseek",
            Some(ThinkingConfig::Level(ThinkingLevel::Xhigh)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("max"));
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
        assert!(r.drop_temperature);
    }

    /// DeepSeek 的 Low 保持为 low。
    #[test]
    fn deepseek_low_remains_low() {
        let r = translate(
            "deepseek-v4-pro",
            "deepseek",
            Some(ThinkingConfig::Level(ThinkingLevel::Low)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
    }

    /// DeepSeek 的 Medium 被钳制为 high。
    #[test]
    fn deepseek_medium_clamps_to_high() {
        let r = translate(
            "deepseek-v4-pro",
            "deepseek",
            Some(ThinkingConfig::Level(ThinkingLevel::Medium)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
    }

    /// DeepSeek 较小的预算 token 映射为 low。
    #[test]
    fn deepseek_budget_tokens_low_uses_low() {
        let r = translate(
            "deepseek-v4-pro",
            "deepseek",
            Some(ThinkingConfig::BudgetTokens(2000)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
    }

    /// 经 DashScope 访问的 DeepSeek 使用 enable_thinking。
    #[test]
    fn deepseek_via_dashscope_uses_enable_thinking() {
        /*
         * 关键：同一个 deepseek-v4-pro 模型托管在阿里云百炼（Bailian / DashScope）
         * 上时，使用的是 `enable_thinking`——而不是 reasoning_effort。
         * 提供商优先于模型名称。
         * 已核实：https://help.aliyun.com/zh/model-studio/deep-thinking
         */
        let r = translate(
            "deepseek-v4-pro",
            "dashscope",
            Some(ThinkingConfig::Level(ThinkingLevel::High)),
        );
        assert_eq!(r.enable_thinking, Some(true));
        assert!(
            r.reasoning_effort.is_none(),
            "DeepSeek via DashScope must use enable_thinking, not reasoning_effort"
        );
        assert!(!r.drop_temperature);
    }

    /// 阿里云提供商别名下的 Qwen 使用 enable_thinking。
    #[test]
    fn qwen_via_aliyun_provider_alias_uses_enable_thinking() {
        // 百炼提供商的任何别名都必须解析为 enable_thinking。
        let r = translate(
            "qwen3-max",
            "aliyun",
            Some(ThinkingConfig::BudgetTokens(4096)),
        );
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.thinking_budget, Some(4096));
        assert!(r.reasoning_effort.is_none());
    }

    /// Qwen3 使用 enable_thinking 开关。
    #[test]
    fn qwen3_enable_thinking_flag() {
        let r = translate(
            "qwen3-235b-a22b",
            "dashscope",
            Some(ThinkingConfig::Level(ThinkingLevel::Medium)),
        );
        assert!(r.reasoning_effort.is_none());
        assert_eq!(r.enable_thinking, Some(true));
        assert!(r.thinking_budget.is_none());
        assert!(!r.drop_temperature);
    }

    /// Qwen3 的预算 token 输出为 thinking_budget。
    #[test]
    fn qwen3_budget_tokens_emits_thinking_budget() {
        let r = translate(
            "qwen3-max",
            "dashscope",
            Some(ThinkingConfig::BudgetTokens(8000)),
        );
        assert_eq!(r.thinking_budget, Some(8000));
        assert_eq!(r.enable_thinking, Some(true));
    }

    /// Kimi 2.6 仅使用 thinking.type 块。
    #[test]
    fn kimi_26_uses_thinking_type_block_only() {
        let r = translate(
            "kimi-k2.6",
            "moonshot",
            Some(ThinkingConfig::Level(ThinkingLevel::High)),
        );
        assert!(r.thinking_type.is_some());
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
        assert!(
            r.reasoning_effort.is_none(),
            "Kimi K2.6 exposes a toggle, not an effort level"
        );
        assert!(r.enable_thinking.is_none());
        assert!(!r.drop_temperature);
    }

    /// Kimi 2.6 关闭时输出 disabled 类型。
    #[test]
    fn kimi_26_disabled_emits_disabled_type() {
        let r = translate("kimi-k2.6", "moonshot", Some(ThinkingConfig::Disabled));
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("disabled")
        );
    }

    /// GLM-5.2 同时使用 reasoning_effort 与 thinking.type。
    #[test]
    fn glm_52_uses_reasoning_effort_and_thinking_type() {
        // GLM-5.2：reasoning_effort（max/xhigh/high/...）+ thinking.type。
        let r = translate(
            "glm-5.2",
            "zhipu",
            Some(ThinkingConfig::Level(ThinkingLevel::Xhigh)),
        );
        assert_eq!(r.reasoning_effort.as_deref(), Some("xhigh"));
        assert!(r.thinking_type.is_some());
        assert_eq!(
            r.thinking_type
                .as_ref()
                .map(|block| block.block_type.as_str()),
            Some("enabled")
        );
    }

    /// 非推理模型静默忽略 thinking 配置。
    #[test]
    fn non_reasoning_model_drops_thinking_silently() {
        let r = translate("gpt-4o", "openai", Some(ThinkingConfig::medium()));
        assert!(r.reasoning_effort.is_none());
        assert!(r.enable_thinking.is_none());
        assert!(r.thinking_budget.is_none());
        assert!(r.thinking_type.is_none());
        assert!(!r.drop_temperature);
    }

    /// Ollama 使用原生 think 级别。
    #[test]
    fn ollama_uses_native_think_level() {
        let r = translate(
            "gpt-oss:20b",
            "ollama",
            Some(ThinkingConfig::Level(ThinkingLevel::Medium)),
        );
        assert_eq!(
            r.ollama_think,
            Some(serde_json::Value::String("medium".to_string()))
        );
    }

    /// 未配置 thinking 时不输出任何字段。
    #[test]
    fn none_thinking_emits_nothing() {
        let r = translate("gpt-5.6-sol", "openai", None);
        assert!(r.reasoning_effort.is_none());
        assert!(
            !r.drop_temperature,
            "no thinking config must keep temperature"
        );
    }
}
