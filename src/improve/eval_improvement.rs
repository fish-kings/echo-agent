/*!
基于评估驱动的改进引擎 —— 评估改进循环的统一入口。

一个开关即可启用全部能力：带 LLM 评分器的评估、A/B 对比、
提示词再生成、HTML 报告生成，以及迭代式改进。

原名 `SelfEvolution`，为避免与新的 `evolution` 模块
（记忆/技能/规则生命周期）产生命名冲突而重命名为 `EvalDrivenImprovement`。

# 用法

```rust,ignore
let result = EvalDrivenImprovement::new()
    .with_eval_cases(cases)
    .with_run_store(store)
    .enable()
    .run(agent_factory)
    .await;
```
*/

use crate::eval::{EvalCase, generate_html};
use crate::improve::{ImprovementLoop, LoopResult};
use crate::trace::RunStore;
use std::sync::Arc;

/**
基于评估驱动的改进引擎。调用一次 `.enable()` 即可启用全部能力。

这是一个以评估为核心的改进系统，使用评估用例来迭代式地
改进 Agent 的提示词。它与负责记忆/技能/规则生命周期管理的
`evolution` 模块是相互独立的两套体系。
*/
pub struct EvalDrivenImprovement {
    /// 用于驱动改进循环的评估用例集合。
    cases: Vec<EvalCase>,
    /// 可选的运行记录存储，用于持久化每次运行结果。
    run_store: Option<Arc<dyn RunStore>>,
    /// 改进循环允许执行的最大迭代次数。
    max_iterations: usize,
    /// 可选的 HTML 报告输出目录。
    report_dir: Option<String>,
    /// 引擎是否已启用。
    enabled: bool,
}

impl Default for EvalDrivenImprovement {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalDrivenImprovement {
    /// 创建一个默认配置、尚未启用的改进引擎。
    pub fn new() -> Self {
        Self {
            cases: Vec::new(),
            run_store: None,
            max_iterations: 5,
            report_dir: None,
            enabled: false,
        }
    }

    /// 设置用于驱动改进循环的评估用例。
    pub fn with_eval_cases(mut self, cases: Vec<EvalCase>) -> Self {
        self.cases = cases;
        self
    }
    /// 设置用于持久化运行记录的存储实现。
    pub fn with_run_store(mut self, store: Arc<dyn RunStore>) -> Self {
        self.run_store = Some(store);
        self
    }
    /// 设置改进循环允许执行的最大迭代次数。
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
    /// 设置 HTML 报告的输出目录。
    pub fn with_report_dir(mut self, dir: &str) -> Self {
        self.report_dir = Some(dir.to_string());
        self
    }

    /// 启用基于评估驱动的改进引擎。
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }
    /// 返回该引擎当前是否已启用。
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// 运行完整的改进流水线。
    pub async fn run<F>(&self, agent_factory: F) -> Option<LoopResult>
    where
        F: Fn() -> Box<dyn crate::agent::Agent>,
    {
        if !self.enabled || self.cases.is_empty() {
            return None;
        }

        let runner = ImprovementLoop {
            max_iterations: self.max_iterations,
            ..ImprovementLoop::default()
        };
        let result = runner
            .run(&self.cases, agent_factory, &self.run_store)
            .await;

        if let Some(ref dir) = self.report_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!("Failed to create report directory {dir}: {e}");
            }
            for (i, iter) in result.iterations.iter().enumerate() {
                let html = generate_html(&iter.eval_report, &format!("Iteration {i}"));
                let path = format!("{dir}/iter_{i}.html");
                if let Err(e) = std::fs::write(&path, html) {
                    tracing::warn!("Failed to write iteration report to {path}: {e}");
                }
            }
            if let Some(last) = result.iterations.last() {
                let html = generate_html(&last.eval_report, "Final");
                let path = format!("{dir}/final.html");
                if let Err(e) = std::fs::write(&path, html) {
                    tracing::warn!("Failed to write final report to {path}: {e}");
                }
            }
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{EvalConstraints, SuccessCriteria};
    use crate::testing::MockAgent;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn failing_case() -> EvalCase {
        EvalCase {
            id: "iteration-config".to_string(),
            name: "Iteration config".to_string(),
            description: String::new(),
            domain: None,
            task: "Return the configured marker".to_string(),
            project_fixture: None,
            success_criteria: SuccessCriteria::OutputContains {
                substring: "expected-marker".to_string(),
            },
            constraints: EvalConstraints::default(),
        }
    }

    #[tokio::test]
    async fn configured_max_iterations_controls_real_run() -> std::result::Result<(), String> {
        let engine = EvalDrivenImprovement::new()
            .with_eval_cases(vec![failing_case()])
            .max_iterations(2)
            .enable();

        let result = engine
            .run(|| Box::new(MockAgent::new("iteration-config").with_response("not-matched")))
            .await
            .ok_or_else(|| "enabled improvement run returned no result".to_string())?;

        assert_eq!(result.iterations.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn zero_iterations_does_not_construct_an_agent() -> std::result::Result<(), String> {
        let calls = Arc::new(AtomicUsize::new(0));
        let factory_calls = Arc::clone(&calls);
        let engine = EvalDrivenImprovement::new()
            .with_eval_cases(vec![failing_case()])
            .max_iterations(0)
            .enable();

        let result = engine
            .run(move || {
                factory_calls.fetch_add(1, Ordering::SeqCst);
                Box::new(MockAgent::new("zero-iterations"))
            })
            .await
            .ok_or_else(|| "zero-iteration run returned no result".to_string())?;

        assert!(result.iterations.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn disabled_or_empty_runs_still_skip_agent_construction() {
        let calls = Arc::new(AtomicUsize::new(0));
        let disabled_calls = Arc::clone(&calls);
        let disabled = EvalDrivenImprovement::new()
            .with_eval_cases(vec![failing_case()])
            .max_iterations(2);
        let disabled_result = disabled
            .run(move || {
                disabled_calls.fetch_add(1, Ordering::SeqCst);
                Box::new(MockAgent::new("disabled"))
            })
            .await;

        let empty_calls = Arc::clone(&calls);
        let empty_result = EvalDrivenImprovement::new()
            .max_iterations(2)
            .enable()
            .run(move || {
                empty_calls.fetch_add(1, Ordering::SeqCst);
                Box::new(MockAgent::new("empty"))
            })
            .await;

        assert!(disabled_result.is_none());
        assert!(empty_result.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
