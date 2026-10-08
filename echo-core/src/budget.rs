/*!
Token 预算管理 —— 对 LLM 上下文窗口进行细粒度分配。

将模型的上下文窗口划分为四类：
- 系统提示词
- 工具定义
- 对话历史
- 输出生成

并预留一部分安全余量。当对话历史超出其分配额度时，预算会发出需要压缩的信号。
*/

use serde::Serialize;

use crate::error::{ReactError, Result};

/**
细粒度的 token 预算，将模型的上下文窗口按固定百分比进行分配。
*/
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// 上下文窗口总大小（取决于模型，例如 GPT-4o 为 128000）。
    total_window: usize,
    /// 为系统提示词预留的比例（默认 0.10）。
    system_prompt_pct: f64,
    /// 为工具定义预留的比例（默认 0.05）。
    tool_defs_pct: f64,
    /// 为输出生成预留的比例（默认 0.10）。
    output_pct: f64,
    /// 作为安全余量保留的比例（默认 0.10）。
    safety_pct: f64,
}

impl TokenBudget {
    /// 该预算所管理的上下文窗口总大小。
    pub fn total_window(&self) -> usize {
        self.total_window
    }
    /**
    根据上下文窗口总大小创建默认预算。

    使用一种安全的分配方式：10% 系统提示词、5% 工具、10% 输出、10% 安全余量，
    剩余 65% 留给对话历史。窗口大小为 0 时返回错误。
    */
    pub fn new(total_window: usize) -> Result<Self> {
        if total_window == 0 {
            return Err(ReactError::Other(
                "token budget total window must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            total_window,
            system_prompt_pct: 0.10,
            tool_defs_pct: 0.05,
            output_pct: 0.10,
            safety_pct: 0.10,
        })
    }

    /**
    自定义各项分配比例。所有值均为 `total_window` 的比例（例如 `0.10` = 10%）。

    每个值必须是 [0, 1] 内的有限值，且总和不得超过 1.0，否则返回错误。
    */
    pub fn with_allocations(
        mut self,
        system_pct: f64,
        tool_pct: f64,
        output_pct: f64,
        safety_pct: f64,
    ) -> Result<Self> {
        let allocations = [system_pct, tool_pct, output_pct, safety_pct];
        if allocations
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(ReactError::Other(
                "token budget allocations must be finite values in [0, 1]".to_string(),
            ));
        }
        let total = allocations.iter().sum::<f64>();
        if total > 1.0 {
            return Err(ReactError::Other(format!(
                "token budget allocations total {total}, exceeding 1.0"
            )));
        }
        self.system_prompt_pct = system_pct;
        self.tool_defs_pct = tool_pct;
        self.output_pct = output_pct;
        self.safety_pct = safety_pct;
        Ok(self)
    }

    /// 系统提示词可用的最大 token 数。
    pub fn system_prompt_budget(&self) -> usize {
        (self.total_window as f64 * self.system_prompt_pct) as usize
    }

    /// 工具定义可用的最大 token 数。
    pub fn tool_definitions_budget(&self) -> usize {
        (self.total_window as f64 * self.tool_defs_pct) as usize
    }

    /// 为 LLM 输出预留的最大 token 数。
    pub fn output_budget(&self) -> usize {
        (self.total_window as f64 * self.output_pct) as usize
    }

    /// 为应对 provider 计数偏差而保留的安全余量 token 数。
    pub fn safety_budget(&self) -> usize {
        (self.total_window as f64 * self.safety_pct) as usize
    }

    /**
    对话历史可用的最大 token 数。
    即扣除所有分配项与安全余量之后剩余的部分。
    */
    pub fn conversation_budget(&self) -> usize {
        let allocated =
            self.system_prompt_pct + self.tool_defs_pct + self.output_pct + self.safety_pct;
        let remaining = (1.0 - allocated).max(0.0);
        (self.total_window as f64 * remaining) as usize
    }

    /**
    根据估算的 token 数，返回一个详细的 [`TokenAllocation`]，
    说明哪些部分放得下、哪些需要裁剪。
    */
    pub fn allocate(
        &self,
        system_size: usize,
        tool_defs_size: usize,
        conversation_size: usize,
    ) -> TokenAllocation {
        let sys_ok = system_size <= self.system_prompt_budget();
        let tool_ok = tool_defs_size <= self.tool_definitions_budget();
        /* 各类别的百分比只是报告用的目标值，而非硬性分区。
        未用完的容量仍可供其他输入类别使用。 */
        let effective_conversation_budget = self
            .total_window
            .saturating_sub(self.output_budget())
            .saturating_sub(self.safety_budget())
            .saturating_sub(system_size)
            .saturating_sub(tool_defs_size);
        let conv_ok = conversation_size <= effective_conversation_budget;
        let conversation_excess = conversation_size.saturating_sub(effective_conversation_budget);

        let total_used = system_size
            .saturating_add(tool_defs_size)
            .saturating_add(conversation_size);
        let usage_pct = (total_used as f64 / self.total_window as f64) * 100.0;

        let output_fits = self.output_budget() > 0;

        TokenAllocation {
            system_fits: sys_ok,
            tool_defs_fit: tool_ok,
            conversation_fits: conv_ok,
            output_fits,
            conversation_excess,
            usage_pct,
        }
    }

    /// 生成供 CLI 展示的可读预算报告。
    pub fn report(
        &self,
        system_size: usize,
        tool_defs_size: usize,
        conversation_size: usize,
        estimated_output: usize,
    ) -> BudgetReport {
        let allocation = self.allocate(system_size, tool_defs_size, conversation_size);
        BudgetReport {
            total_window: self.total_window,
            system_prompt: system_size,
            system_prompt_budget: self.system_prompt_budget(),
            tool_definitions: tool_defs_size,
            tool_definitions_budget: self.tool_definitions_budget(),
            conversation: conversation_size,
            conversation_budget: self.conversation_budget(),
            estimated_output,
            output_budget: self.output_budget(),
            usage_pct: allocation.usage_pct,
            needs_compression: allocation.needs_compression(),
        }
    }
}

impl Default for TokenBudget {
    /// 创建一个使用默认分配比例、保守的 128K 预算。
    fn default() -> Self {
        Self {
            total_window: 128_000,
            system_prompt_pct: 0.10,
            tool_defs_pct: 0.05,
            output_pct: 0.10,
            safety_pct: 0.10,
        }
    }
}

/// 将 token 预算与实际用量比对后的结果。
#[derive(Debug, Clone)]
pub struct TokenAllocation {
    /// 系统提示词是否在其预算之内。
    pub system_fits: bool,
    /// 工具定义是否在其预算之内。
    pub tool_defs_fit: bool,
    /// 对话历史是否在其预算之内。
    pub conversation_fits: bool,
    /// 是否还有空间容纳输出预算。
    pub output_fits: bool,
    /// 需要释放的对话 token 数（在预算内时为 0）。
    pub conversation_excess: usize,
    /// 总用量占整个窗口的百分比。
    pub usage_pct: f64,
}

impl TokenAllocation {
    /// 当所有类别都在各自预算内时返回 true。
    pub fn ok(&self) -> bool {
        self.system_fits && self.tool_defs_fit && self.conversation_fits && self.output_fits
    }

    /// 当应触发压缩（对话超出预算）时返回 true。
    pub fn needs_compression(&self) -> bool {
        self.conversation_excess > 0
    }
}

/// 供 CLI 展示的预算状态报告。
#[derive(Debug, Clone, Serialize)]
pub struct BudgetReport {
    /// 上下文窗口总大小。
    pub total_window: usize,
    /// 系统提示词实际占用的 token 数。
    pub system_prompt: usize,
    /// 系统提示词的预算。
    pub system_prompt_budget: usize,
    /// 工具定义实际占用的 token 数。
    pub tool_definitions: usize,
    /// 工具定义的预算。
    pub tool_definitions_budget: usize,
    /// 对话历史实际占用的 token 数。
    pub conversation: usize,
    /// 对话历史的预算。
    pub conversation_budget: usize,
    /// 预估的输出 token 数。
    pub estimated_output: usize,
    /// 输出的预算。
    pub output_budget: usize,
    /// 总用量占窗口的百分比。
    pub usage_pct: f64,
    /// 是否需要触发压缩。
    pub needs_compression: bool,
}

/// 可嵌入 AgentConfig 的 token 预算配置。
#[derive(Debug, Clone)]
pub struct TokenBudgetConfig {
    /// 上下文窗口总大小。为 `None` 时根据模型自动检测。
    pub total_window: Option<usize>,
    /// 自定义分配比例：系统提示词所占比例。
    pub system_pct: f64,
    /// 工具定义所占比例。
    pub tool_pct: f64,
    /// 输出生成所占比例。
    pub output_pct: f64,
    /// 安全余量所占比例。
    pub safety_pct: f64,
    /// 是否启用预算检查。
    pub enabled: bool,
}

impl Default for TokenBudgetConfig {
    /// 默认启用、自动检测窗口，并使用默认分配比例。
    fn default() -> Self {
        Self {
            total_window: None,
            system_pct: 0.10,
            tool_pct: 0.05,
            output_pct: 0.10,
            safety_pct: 0.10,
            enabled: true,
        }
    }
}

impl TokenBudgetConfig {
    /// 创建一个启用状态、自动检测窗口大小的配置。
    pub fn enabled() -> Self {
        Self::default()
    }

    /// 创建一个禁用状态的配置（跳过预算检查）。
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// 显式设置上下文窗口大小，而不是自动检测。
    pub fn with_total_window(mut self, window: usize) -> Self {
        self.total_window = Some(window);
        self
    }

    /// 根据此配置构建 TokenBudget；未设置窗口时使用 `fallback_window`。
    pub fn build(&self, fallback_window: usize) -> Result<TokenBudget> {
        let window = self.total_window.unwrap_or(fallback_window);
        TokenBudget::new(window)?.with_allocations(
            self.system_pct,
            self.tool_pct,
            self.output_pct,
            self.safety_pct,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget() {
        let budget = TokenBudget::new(100_000).unwrap_or_default();
        assert_eq!(budget.system_prompt_budget(), 10_000); // 10%
        assert_eq!(budget.tool_definitions_budget(), 5_000); // 5%
        assert_eq!(budget.output_budget(), 10_000); // 10%
        // 100% - 10% - 5% - 10% - 10% = 65%
        assert_eq!(budget.conversation_budget(), 65_000);
    }

    #[test]
    fn default_budget_uses_conservative_context_window() {
        let budget = TokenBudget::default();
        assert_eq!(budget.total_window(), 128_000);
    }

    #[test]
    fn test_allocation_ok() {
        let budget = TokenBudget::new(100_000).unwrap_or_default();
        let all = budget.allocate(5_000, 2_000, 30_000);
        assert!(all.ok());
        assert!(!all.needs_compression());
    }

    #[test]
    fn test_allocation_needs_compression() {
        let budget = TokenBudget::new(100_000).unwrap_or_default();
        let all = budget.allocate(5_000, 2_000, 75_000); // 超出 80K 的总输入容量
        assert!(!all.ok());
        assert!(all.needs_compression());
        assert_eq!(all.conversation_excess, 2_000);
    }

    #[test]
    fn system_and_tool_overflow_reduce_conversation_capacity() -> Result<()> {
        let budget = TokenBudget::new(100_000)?;
        let allocation = budget.allocate(15_000, 10_000, 50_000);

        /* 为输出与安全余量预留 20K 后，扣除实际的系统提示词与工具用量，
        对话历史还剩 55K 可用。 */
        assert_eq!(allocation.conversation_excess, 0);
        let over = budget.allocate(15_000, 10_000, 60_000);
        assert_eq!(over.conversation_excess, 5_000);
        assert!(over.needs_compression());
        Ok(())
    }

    #[test]
    fn test_custom_allocations() {
        let budget = TokenBudget::new(100_000)
            .and_then(|budget| budget.with_allocations(0.05, 0.05, 0.05, 0.05))
            .unwrap_or_default();
        assert_eq!(budget.system_prompt_budget(), 5_000);
        assert_eq!(budget.conversation_budget(), 80_000); // 100% - 20% = 80%
    }

    #[test]
    fn invalid_allocations_are_rejected() {
        for allocations in [
            [-0.1, 0.1, 0.1, 0.1],
            [f64::NAN, 0.1, 0.1, 0.1],
            [f64::INFINITY, 0.1, 0.1, 0.1],
            [0.4, 0.4, 0.4, 0.0],
        ] {
            let result = TokenBudget::new(100).and_then(|budget| {
                budget.with_allocations(
                    allocations[0],
                    allocations[1],
                    allocations[2],
                    allocations[3],
                )
            });
            assert!(result.is_err(), "accepted allocations {allocations:?}");
        }
        assert!(TokenBudget::new(0).is_err());
    }

    #[test]
    fn extreme_usage_counts_do_not_overflow() -> Result<()> {
        let budget = TokenBudget::new(100)?;
        let allocation = budget.allocate(usize::MAX, 1, 1);
        assert!(allocation.usage_pct.is_finite());
        assert!(!allocation.ok());
        Ok(())
    }

    #[test]
    fn test_disabled_config() {
        let config = TokenBudgetConfig::disabled();
        assert!(!config.enabled);
    }
}
