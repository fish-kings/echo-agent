/*!
聊天请求的思考 / 推理深度控制。

主流模型通过多种互不兼容的线上协议来暴露“模型在回答前应该思考多深”。
以下内容已对照各厂商官方 API 文档核实（2026 年中）：

| 系列 | 模型 | 线上字段 | 取值 |
|--------|--------|-----------|--------|
| OpenAI reasoning | GPT-5.6 | `reasoning_effort` | `none`/`low`/`medium`/`high`/`xhigh`/`max` |
| Anthropic effort | Claude 4.6 (Opus/Sonnet) | `effort` + `thinking:{type:"adaptive"}` | `low`/`medium`/`high`/`xhigh`/`max` |
| Anthropic 仅自适应 | Claude 4.7+ | 无请求字段 | 由模型决定 |
| Qwen3 | Qwen3-* | `enable_thinking:bool` + `thinking_budget:int` | bool / int |
| GLM effort | GLM-5.2+ | `reasoning_effort`（+ `thinking.type`） | 实际生效 `none`/`high`/`max` |
| Kimi | K3 / K2.7 / K2.6 | effort / 模型托管 / `thinking.type` | 因模型而异 |

我们不让各厂商的方言渗透进框架，而是只暴露**一个**语义旋钮
（[`ThinkingConfig`]），并在 LLM 客户端实现中按 provider 进行转换。
这样调用点可以在不同模型之间保持可移植。

[`ModelProfile`]: crate::llm::capabilities::ModelProfile
*/

use serde::{Deserialize, Serialize};

/**
统一的推理深度旋钮，由各个 `LlmClient` 实现转换为对应 provider 的线上格式。

[`crate::llm::ChatRequest`] 上为 `None` 表示“使用模型的默认行为”
—— 不发送任何 thinking 字段。
*/
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingConfig {
    /**
    在模型支持开关的情况下完全关闭思考
    （Qwen3 `enable_thinking:false`、GLM `thinking.type:"disabled"`、
    GPT-5.1+ 上的 OpenAI `reasoning_effort:"none"`）。对于没有关闭开关的
    模型，按最低可用档位处理。
    */
    Disabled,
    /**
    离散的推理档位。映射到 OpenAI `reasoning_effort` 与 Anthropic `effort`；
    对于基于预算的协议，会换算为 `max_tokens` 的一定比例。
    */
    Level(ThinkingLevel),
    /**
    内部推理的精确 token 预算。面向 Anthropic `budget_tokens`
    （Claude 3.7–4.5）与 Qwen3 `thinking_budget`；其他系列会近似到最接近的档位。
    */
    BudgetTokens(#[serde(rename = "budget_tokens")] u32),
}

impl ThinkingConfig {
    /**
    便捷构造：medium 档位（当用户只想“多思考一些”而不指定细节时，
    这是一个合理的默认值）。
    */
    pub fn medium() -> Self {
        Self::Level(ThinkingLevel::Medium)
    }

    /**
    将面向用户/配置的灵活描述解析为 `ThinkingConfig`。

    接受：
    - `"auto"` / `""` → 返回 `None`（使用模型默认值）
    - `"disabled"` / `"off"` / `"none"` → [`ThinkingConfig::Disabled`]
    - `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` / `"max"` → [`ThinkingConfig::Level`]
    - 纯整数字符串（`"4000"`）→ [`ThinkingConfig::BudgetTokens`]

    对 `"auto"`/空串返回 `Ok(None)`，对无法识别的字符串返回 `Err`
    （让配置拼写错误显式暴露，而不是悄悄回退到默认值）。
    */
    pub fn parse_spec(s: &str) -> std::result::Result<Option<Self>, String> {
        let trimmed = s.trim();
        match trimmed.to_ascii_lowercase().as_str() {
            "" | "auto" | "default" => Ok(None),
            "disabled" | "off" | "false" => Ok(Some(Self::Disabled)),
            other => {
                if let Ok(n) = other.parse::<u32>() {
                    return Ok(Some(Self::BudgetTokens(n)));
                }
                ThinkingLevel::parse(other)
                    .map(|lvl| Some(Self::Level(lvl)))
                    .ok_or_else(|| format!("unrecognized thinking spec: '{s}'"))
            }
        }
    }

    /**
    该配置映射到的 OpenAI `reasoning_effort` 字符串，或 `None`。

    供 OpenAI 兼容的 provider 使用（OpenAI、Azure、DeepSeek-V3.2+、
    Gemini 的 OpenAI 兼容接口）。`Disabled` 映射为 `"none"`（GPT-5.1+），
    较旧的模型会在 API 层将其视为 `"minimal"`；provider 可能会向下降档。
    */
    pub fn to_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            Self::Disabled => Some("minimal"),
            Self::Level(ThinkingLevel::None) => Some("none"),
            Self::Level(ThinkingLevel::Minimal) => Some("minimal"),
            Self::Level(ThinkingLevel::Low) => Some("low"),
            Self::Level(ThinkingLevel::Medium) => Some("medium"),
            Self::Level(ThinkingLevel::High) => Some("high"),
            Self::Level(ThinkingLevel::Xhigh) => Some("xhigh"),
            Self::Level(ThinkingLevel::Max) => Some("max"),
            /* 原始预算映射到最接近的 effort 档位：<4k ≈ low，
            <12k ≈ medium，<24k ≈ high，<48k ≈ xhigh，其余为 max。 */
            Self::BudgetTokens(n) => Some(if *n < 4_000 {
                "low"
            } else if *n < 12_000 {
                "medium"
            } else if *n < 24_000 {
                "high"
            } else if *n < 48_000 {
                "xhigh"
            } else {
                "max"
            }),
        }
    }

    /**
    该配置映射到的 Anthropic `effort` 字符串（Claude 4.6+）；
    关闭思考时返回 `None`。

    Claude 4.6 接受：`low`/`medium`/`high`/`xhigh`/`max`。

    `Minimal` → `None` (关闭思考,不是 `low`),与"最低档 = 极速响应"定位一致。
    `Disabled` → `None` (不发 effort 字段 + 不发 thinking block = 完全关闭)。
    */
    pub fn to_anthropic_effort(&self) -> Option<&'static str> {
        match self {
            Self::Disabled => None,
            Self::Level(ThinkingLevel::None) => None,
            Self::Level(ThinkingLevel::Minimal) => None,
            Self::Level(ThinkingLevel::Low) => Some("low"),
            Self::Level(ThinkingLevel::Medium) => Some("medium"),
            Self::Level(ThinkingLevel::High) => Some("high"),
            Self::Level(ThinkingLevel::Xhigh) => Some("xhigh"),
            Self::Level(ThinkingLevel::Max) => Some("max"),
            Self::BudgetTokens(n) => Some(if *n < 4_000 {
                "low"
            } else if *n < 12_000 {
                "medium"
            } else if *n < 24_000 {
                "high"
            } else if *n < 48_000 {
                "xhigh"
            } else {
                "max"
            }),
        }
    }

    /**
    根据请求的 `max_tokens` 计算 Anthropic `budget_tokens`（Claude 3.7–4.5）
    （预算必须严格小于 `max_tokens`）。

    `Disabled` / `None` / `Minimal` → `None` (关闭思考,极速响应)。
    当 `max_tokens` 过小（<= 1）无法容纳预算时返回 `None`；否则预算会被截断到 `max_tokens - 1`。
    */
    pub fn to_anthropic_budget(&self, max_tokens: u32) -> Option<u32> {
        let budget = match self {
            Self::Disabled => return None,
            Self::Level(ThinkingLevel::None) => return None,
            Self::Level(ThinkingLevel::Minimal) => return None,
            Self::BudgetTokens(n) => *n,
            Self::Level(level) => {
                let frac = match level {
                    ThinkingLevel::Low => 0.25,
                    ThinkingLevel::Medium => 0.5,
                    ThinkingLevel::High => 0.8,
                    ThinkingLevel::Xhigh => 0.95,
                    ThinkingLevel::Max => 0.98,
                    // None / Minimal 不可达（已在上方提前返回）
                    ThinkingLevel::None | ThinkingLevel::Minimal => return None,
                };
                ((max_tokens as f64) * frac).round() as u32
            }
        };
        if max_tokens <= 1 {
            return None;
        }
        Some(budget.min(max_tokens - 1))
    }

    /**
    该配置映射到的 Qwen3/GLM `enable_thinking` 布尔值。

    `Minimal` 返回 `false` because "最低档" = 关闭思考(极速响应,最省
    成本),与 `Disabled` 一致。只有 Low/Medium/High/Xhigh 才开启思考。
    */
    pub fn to_enable_thinking(&self) -> bool {
        !matches!(
            self,
            Self::Disabled | Self::Level(ThinkingLevel::None) | Self::Level(ThinkingLevel::Minimal)
        )
    }

    /**
    GLM 的 `thinking.type` 取值（`"enabled"` 或 `"disabled"`）。

    `Minimal` → `"disabled"` (最低档 = 关闭思考,极速响应)。
    */
    pub fn to_glm_thinking_type(&self) -> &'static str {
        if matches!(
            self,
            Self::Disabled | Self::Level(ThinkingLevel::None) | Self::Level(ThinkingLevel::Minimal)
        ) {
            "disabled"
        } else {
            "enabled"
        }
    }

    /**
    该配置映射到的 GLM-5.x `reasoning_effort` 字符串，或 `None`。

    GLM-5.2 接受 `max`/`xhigh`/`high`/`medium`/`low`/`minimal`/`none`。
    （服务端会把 `low`/`medium` 映射为 `high`，但我们仍发送字面值，
    以便在请求日志中保留用户的原始意图。）
    */
    pub fn to_glm_reasoning_effort(&self) -> Option<&'static str> {
        match self {
            Self::Disabled => Some("none"),
            Self::Level(ThinkingLevel::None) => Some("none"),
            Self::Level(ThinkingLevel::Minimal) => Some("minimal"),
            Self::Level(ThinkingLevel::Low) => Some("low"),
            Self::Level(ThinkingLevel::Medium) => Some("medium"),
            Self::Level(ThinkingLevel::High) => Some("high"),
            Self::Level(ThinkingLevel::Xhigh) => Some("xhigh"),
            Self::Level(ThinkingLevel::Max) => Some("max"),
            Self::BudgetTokens(n) => Some(if *n < 4_000 {
                "low"
            } else if *n < 12_000 {
                "medium"
            } else if *n < 24_000 {
                "high"
            } else {
                "max"
            }),
        }
    }
}

/// 离散的推理档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    /**
    不思考（GPT-5.1+ `reasoning_effort:"none"`）。在没有真正关闭开关的
    模型上会向下取整为 Minimal。
    */
    None,
    /**
    最少思考（最快、最便宜）。只有接受 `reasoning_effort: "minimal"` 的
    模型（GPT-5 系列）会精确遵循该档位。
    */
    Minimal,
    /// 低推理强度。
    Low,
    /// 中等推理强度。
    Medium,
    /// 高推理强度。
    High,
    /// 超高推理强度。
    Xhigh,
    /**
    最大推理强度。对于 GPT-5.6 和 Claude 4.6 等模型，它与 `xhigh` 是不同的档位。
    */
    Max,
}

impl ThinkingLevel {
    /**
    从面向用户的字符串解析（不区分大小写）。对无法识别的输入返回 `None`，
    而不是 panic。
    */
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Some(Self::None),
            "minimal" | "min" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" | "med" | "normal" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::Xhigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/**
模型所使用的 thinking 线上协议（如果有的话）。

用于决定：(a) 是否要发送 thinking 字段；(b) 应用哪种转换。它由 provider
端点、API 协议和模型 id 集中解析得出，而不是持久化在用户配置中。
*/
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingProtocol {
    /**
    模型不支持思考控制。请求上的任何 `ThinkingConfig` 都会被静默丢弃
    （并记录一条 `warn!`）。
    */
    #[default]
    None,
    /// 模型自动进行推理，不暴露请求侧的控制项。
    ModelManaged,
    /**
    OpenAI `reasoning_effort`（GPT-5 系列、o 系列、**DeepSeek-V3.2+**）。
    o 系列禁止使用 `temperature`/`top_p`。
    */
    OpenaiReasoningEffort,
    /**
    DeepSeek `reasoning_effort` 加 `thinking:{type}`。尽管两者都使用
    OpenAI 兼容的传输方式，但它接受的 effort 取值和关闭语义与 OpenAI 不同。
    */
    DeepseekReasoningEffort,
    /**
    Anthropic `effort` + `thinking:{type:"adaptive"}`（Claude 4.6 Sonnet/Opus）。
    在 Opus 4.7+ 上会完全去掉 `thinking` 块（仅自适应）。
    */
    AnthropicEffort,
    /**
    Anthropic `thinking:{type:"enabled", budget_tokens}`（Claude 3.7 – 4.5）。
    从 4.5 起这些模型也接受 `effort` 参数。
    */
    AnthropicThinkingBudget,
    /**
    Claude Opus 4.7+ —— 仅支持自适应思考。发送任何 thinking 字段都会
    返回 400，因此我们会丢弃该配置并记录一条 `warn!`。
    */
    AnthropicAdaptive,
    /// Qwen3 `enable_thinking` 布尔值 + 可选的 `thinking_budget` 整数。
    EnableThinkingFlag,
    /**
    通用的 `thinking:{type:"enabled"|"disabled"}` 开关，
    被 Kimi 2.6 及兼容的聊天端点使用。
    */
    ThinkingType,
    /**
    GLM-5.2+ `reasoning_effort`。
    接受 `max`/`xhigh`/`high`/`medium`/`low`/`minimal`/`none`；`low`/
    `medium` 会被服务端映射为 `high`，但我们仍如实发送。
    */
    GlmReasoningEffort,
    /// Ollama `think` 布尔值或档位字符串。
    OllamaThink,
}

impl ThinkingProtocol {
    /**
    当且仅当该协议确实需要发送非 `None` 的 thinking 字段时返回 true。
    `None` 与 `AnthropicAdaptive` 都表示“什么都不发送”。
    */
    pub fn emits_field(&self) -> bool {
        matches!(
            self,
            Self::OpenaiReasoningEffort
                | Self::DeepseekReasoningEffort
                | Self::AnthropicEffort
                | Self::AnthropicThinkingBudget
                | Self::EnableThinkingFlag
                | Self::ThinkingType
                | Self::GlmReasoningEffort
                | Self::OllamaThink
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_parse() {
        assert_eq!(ThinkingLevel::parse("high"), Some(ThinkingLevel::High));
        assert_eq!(ThinkingLevel::parse("XHIGH"), Some(ThinkingLevel::Xhigh));
        assert_eq!(ThinkingLevel::parse("max"), Some(ThinkingLevel::Max));
        assert_eq!(ThinkingLevel::parse("ultra"), None);
        assert_eq!(ThinkingLevel::parse("none"), Some(ThinkingLevel::None));
        assert_eq!(ThinkingLevel::parse("MIN"), Some(ThinkingLevel::Minimal));
        assert_eq!(ThinkingLevel::parse("medium"), Some(ThinkingLevel::Medium));
        assert_eq!(ThinkingLevel::parse("bogus"), None);
        assert_eq!(ThinkingLevel::parse(""), None);
    }

    #[test]
    fn test_to_reasoning_effort() {
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::High).to_reasoning_effort(),
            Some("high")
        );
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Xhigh).to_reasoning_effort(),
            Some("xhigh")
        );
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Max).to_reasoning_effort(),
            Some("max")
        );
        assert_eq!(
            ThinkingConfig::Disabled.to_reasoning_effort(),
            Some("minimal")
        );
        assert_eq!(
            ThinkingConfig::BudgetTokens(2000).to_reasoning_effort(),
            Some("low")
        );
        assert_eq!(
            ThinkingConfig::BudgetTokens(50_000).to_reasoning_effort(),
            Some("max")
        );
    }

    #[test]
    fn test_to_anthropic_effort() {
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Medium).to_anthropic_effort(),
            Some("medium")
        );
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Xhigh).to_anthropic_effort(),
            Some("xhigh")
        );
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Max).to_anthropic_effort(),
            Some("max")
        );
        // Disabled → None（4.6+ 模型将缺省视为由模型自行决定）。
        assert_eq!(ThinkingConfig::Disabled.to_anthropic_effort(), None);
    }

    #[test]
    fn test_anthropic_budget_levels() {
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Medium).to_anthropic_budget(8000),
            Some(4000)
        );
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::High).to_anthropic_budget(1000),
            Some(800)
        );
        assert_eq!(ThinkingConfig::Disabled.to_anthropic_budget(8000), None);
    }

    #[test]
    fn test_anthropic_budget_must_be_below_max() {
        assert_eq!(
            ThinkingConfig::BudgetTokens(10_000).to_anthropic_budget(4000),
            Some(3999)
        );
        assert_eq!(ThinkingConfig::medium().to_anthropic_budget(1), None);
    }

    #[test]
    fn test_enable_thinking_flag() {
        assert!(!ThinkingConfig::Disabled.to_enable_thinking());
        assert!(ThinkingConfig::medium().to_enable_thinking());
    }

    #[test]
    fn test_glm_thinking_type() {
        assert_eq!(ThinkingConfig::Disabled.to_glm_thinking_type(), "disabled");
        assert_eq!(ThinkingConfig::medium().to_glm_thinking_type(), "enabled");
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::High).to_glm_thinking_type(),
            "enabled"
        );
    }

    #[test]
    fn test_emits_field() {
        assert!(!ThinkingProtocol::None.emits_field());
        assert!(!ThinkingProtocol::ModelManaged.emits_field());
        assert!(!ThinkingProtocol::AnthropicAdaptive.emits_field());
        assert!(ThinkingProtocol::OpenaiReasoningEffort.emits_field());
        assert!(ThinkingProtocol::DeepseekReasoningEffort.emits_field());
        assert!(ThinkingProtocol::AnthropicEffort.emits_field());
        assert!(ThinkingProtocol::AnthropicThinkingBudget.emits_field());
        assert!(ThinkingProtocol::EnableThinkingFlag.emits_field());
        assert!(ThinkingProtocol::ThinkingType.emits_field());
        assert!(ThinkingProtocol::GlmReasoningEffort.emits_field());
        assert!(ThinkingProtocol::OllamaThink.emits_field());
    }

    #[test]
    fn test_to_glm_reasoning_effort() {
        assert_eq!(
            ThinkingConfig::Level(ThinkingLevel::Xhigh).to_glm_reasoning_effort(),
            Some("xhigh")
        );
        assert_eq!(
            ThinkingConfig::BudgetTokens(50_000).to_glm_reasoning_effort(),
            Some("max")
        );
        assert_eq!(
            ThinkingConfig::Disabled.to_glm_reasoning_effort(),
            Some("none")
        );
    }

    #[test]
    fn test_parse_spec() {
        assert_eq!(ThinkingConfig::parse_spec("auto"), Ok(None));
        assert_eq!(ThinkingConfig::parse_spec(""), Ok(None));
        assert_eq!(
            ThinkingConfig::parse_spec("disabled"),
            Ok(Some(ThinkingConfig::Disabled))
        );
        assert_eq!(
            ThinkingConfig::parse_spec("xhigh"),
            Ok(Some(ThinkingConfig::Level(ThinkingLevel::Xhigh)))
        );
        assert_eq!(
            ThinkingConfig::parse_spec("max"),
            Ok(Some(ThinkingConfig::Level(ThinkingLevel::Max)))
        );
        assert_eq!(
            ThinkingConfig::parse_spec("4000"),
            Ok(Some(ThinkingConfig::BudgetTokens(4000)))
        );
        assert!(ThinkingConfig::parse_spec("bogus").is_err());
    }
}
