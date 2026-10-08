/*!
与具体供应商无关的框架配置。

文件发现、产品提示词、模型目录、渠道、UI 设置以及服务器设置都属于
宿主应用的职责范围。框架本身只接受一个带类型的配置值和显式路径。
*/

use crate::agent::AgentConfig;
use echo_core::budget::TokenBudgetConfig;
use echo_core::llm::LlmApiProtocol;
use echo_core::llm::capabilities::infer_context_window;
use serde::{Deserialize, Serialize};

/// 默认的 Agent system prompt
pub const DEFAULT_AGENT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// 可复用的单个 Agent 运行时配置。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct FrameworkConfig {
    /// 模型相关配置
    pub model: ModelConfig,
    /// Agent 相关配置
    pub agent: AgentSettings,
}

/**
 * 解析出实际使用的上下文窗口大小：优先使用显式配置，其次按 provider/model 推断，
 * 最后回退到默认 token 限制
 */
fn resolve_context_window(explicit: Option<u32>, provider: &str, model_name: &str) -> usize {
    explicit
        .or_else(|| infer_context_window(provider, model_name))
        .unwrap_or(u32::try_from(crate::agent::config::DEFAULT_TOKEN_LIMIT).unwrap_or(128_000))
        .clamp(1, 10_000_000) as usize
}

/// 将 `FrameworkConfig` 转换为底层的 `AgentConfig`
impl From<FrameworkConfig> for AgentConfig {
    /// 执行从框架配置到 Agent 配置的转换。
    fn from(value: FrameworkConfig) -> Self {
        let FrameworkConfig { model, agent } = value;
        let context_window =
            resolve_context_window(model.context_window, &model.provider, &model.name);
        let token_limit = if agent.token_limit > 0 {
            agent.token_limit
        } else if model.context_window.is_some() {
            context_window
        } else {
            usize::MAX
        };
        let token_budget_config = if model.context_window.is_some() || agent.token_limit > 0 {
            TokenBudgetConfig {
                total_window: Some(context_window),
                ..Default::default()
            }
        } else {
            TokenBudgetConfig::default()
        };

        let mut config = AgentConfig::standard(&model.name, &agent.name, &agent.system_prompt)
            .enable_tool(agent.enable_tools)
            .enable_memory(agent.enable_memory)
            .enable_human_in_loop(agent.enable_human_in_loop)
            .enable_subagent(agent.enable_subagent)
            .register_agent_dispatch_tool(agent.register_agent_dispatch_tool)
            .max_iterations(agent.max_iterations)
            .subagent_timeout_secs(agent.subagent_timeout_secs)
            .memory_path(&agent.memory_path)
            .temperature(model.temperature)
            .max_tokens(model.max_tokens)
            .token_limit(token_limit)
            .token_budget(token_budget_config)
            .tool_execution(crate::tools::ToolExecutionConfig {
                timeout_ms: agent.tool_timeout_ms,
                ..Default::default()
            });
        if agent.max_tool_output_tokens > 0 {
            config = config.max_tool_output_tokens(agent.max_tool_output_tokens);
        }
        config
    }
}

impl FrameworkConfig {
    /// 判断当前配置是否需要启用上下文压缩器
    pub fn has_compressor(&self) -> bool {
        self.agent.token_limit > 0
            || self.model.context_window.is_some()
            || !self.agent.compress_strategy.is_empty()
    }

    /// 根据配置的压缩策略，为该 agent 装配对应的上下文压缩器
    pub async fn apply_compressor(&self, agent: &crate::agent::ReactAgent) {
        use crate::compression::compressor::SlidingWindowCompressor;

        if !self.has_compressor() {
            return;
        }
        let context_window = resolve_context_window(
            self.model.context_window,
            &self.model.provider,
            &self.model.name,
        );
        let window = self.agent.compress_window.max(2);
        match self.agent.compress_strategy.as_str() {
            "summary" => {
                use crate::compression::compressor::SummaryCompressor;
                match agent.llm_client().cloned() {
                    Some(llm) => {
                        agent
                            .set_compressor(SummaryCompressor::new(llm, window))
                            .await
                    }
                    None => {
                        tracing::warn!(
                            "summary compression requires an LLM client; using sliding window"
                        );
                        agent
                            .set_compressor(SlidingWindowCompressor::new(window))
                            .await;
                    }
                }
            }
            "hybrid" => {
                use crate::compression::compressor::HybridCompressor;
                match agent.llm_client().cloned() {
                    Some(llm) => {
                        agent
                            .set_compressor(HybridCompressor::summary_buffer(llm, window))
                            .await;
                    }
                    None => {
                        tracing::warn!(
                            "hybrid compression requires an LLM client; using sliding window"
                        );
                        agent
                            .set_compressor(SlidingWindowCompressor::new(window))
                            .await;
                    }
                }
            }
            "adaptive" => {
                use crate::compression::levels::{
                    AdaptiveCompressionConfig, AdaptiveCompressor, tune_for_model,
                };
                let mut config = AdaptiveCompressionConfig::default();
                tune_for_model(&mut config, context_window);
                agent.set_compressor(AdaptiveCompressor::new(config)).await;
            }
            "sliding" | "" => {
                agent
                    .set_compressor(SlidingWindowCompressor::new(window))
                    .await;
            }
            other => {
                tracing::warn!(
                    strategy = other,
                    "unknown compression strategy; using sliding"
                );
                agent
                    .set_compressor(SlidingWindowCompressor::new(window))
                    .await;
            }
        }
    }
}

/// 宿主应用直接选择的模型设置。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    /// 供应商标识
    pub provider: String,
    /// 模型名称
    pub name: String,
    /// 鉴权 token
    pub auth_token: Option<String>,
    /// API 基础 URL
    pub base_url: Option<String>,
    /// API 协议类型
    pub api_protocol: Option<LlmApiProtocol>,
    /// 单次生成的最大 token 数
    pub max_tokens: Option<u32>,
    /// 采样温度
    pub temperature: Option<f32>,
    /// 显式指定的上下文窗口大小
    pub context_window: Option<u32>,
}

impl ModelConfig {
    /// 获取非空的鉴权 token
    pub fn get_auth_token(&self) -> Option<String> {
        self.auth_token.clone().filter(|value| !value.is_empty())
    }

    /// 获取非空的基础 URL
    pub fn get_base_url(&self) -> Option<String> {
        self.base_url.clone().filter(|value| !value.is_empty())
    }

    /// 获取模型名称
    pub fn get_model_name(&self) -> String {
        self.name.clone()
    }
}

/**
可序列化的、与供应商无关的 Agent 设置。

该值与具体格式无关；应用可以从 YAML、JSON、环境变量或其他配置源加载它。
*/
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct AgentSettings {
    /// Agent 名称
    pub name: String,
    /// 系统提示词
    pub system_prompt: String,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 是否启用工具
    pub enable_tools: bool,
    /// 是否启用记忆
    pub enable_memory: bool,
    /// 是否启用人工介入（human-in-the-loop）
    pub enable_human_in_loop: bool,
    /// 启用框架层面的 Subagent 注册与调度支持。
    pub enable_subagent: bool,
    /// 暴露可被模型调用的 `agent_tool` 调度器。
    pub register_agent_dispatch_tool: bool,
    /// 记忆存储路径
    pub memory_path: String,
    /// 工具执行超时时间（毫秒）
    pub tool_timeout_ms: u64,
    /// 工具输出的最大 token 数
    pub max_tool_output_tokens: usize,
    /// token 限制
    pub token_limit: usize,
    /// 上下文压缩策略
    pub compress_strategy: String,
    /// 压缩窗口大小
    pub compress_window: usize,
    /// subagent 执行超时时间（秒）
    pub subagent_timeout_secs: u64,
}

impl Default for AgentSettings {
    /// 创建默认实例。
    fn default() -> Self {
        Self {
            name: "assistant".to_string(),
            system_prompt: DEFAULT_AGENT_SYSTEM_PROMPT.to_string(),
            max_iterations: 10,
            enable_tools: false,
            enable_memory: false,
            enable_human_in_loop: false,
            enable_subagent: false,
            register_agent_dispatch_tool: false,
            memory_path: String::new(),
            tool_timeout_ms: 120_000,
            max_tool_output_tokens: 0,
            token_limit: 0,
            compress_strategy: String::new(),
            compress_window: 20,
            subagent_timeout_secs: 600,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_do_not_choose_product_or_persistence_policy() {
        let config = FrameworkConfig::default();
        assert_eq!(config.agent.system_prompt, DEFAULT_AGENT_SYSTEM_PROMPT);
        assert!(!config.agent.enable_memory);
        assert!(config.agent.memory_path.is_empty());
        assert!(config.model.name.is_empty());
    }
}
