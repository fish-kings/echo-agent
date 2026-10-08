/*!
 * ProgressBridge —— 将 Agent 回调连接到 TaskEventBus。
 *
 * 把 `AgentCallback` 事件（on_iteration、on_tool_start 等）转换为
 * `TaskEvent::Progress` 事件发出，使后台任务跟踪能够实时观察 Agent 的执行过程。
 */

use crate::agent::{AgentCallback, StepType};
use crate::error::ReactError;
use crate::llm::types::Message;
use crate::tasks::{TaskEvent, TaskEventBus, TaskProgress};
use chrono::Utc;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/**
 * 将 Agent 执行回调桥接为 [`TaskEventBus`] 上的进度事件。
 *
 * 在开始执行前将其注册为回调；执行完成后先调用 [`disable()`](Self::disable)，
 * 再通过 [`remove_callbacks_by_type_name`](crate::agent::ReactAgent::remove_callbacks_by_type_name) 移除。
 *
 * # 进度估算
 *
 * 当 `max_iterations` 已知时，按线性方式计算进度。
 * 当未知（为 0）时，使用渐近逼近 95% 的递减曲线——确保在 `on_final_answer`
 * 触发之前任务永远不会报告“已完成”。
 */
pub struct ProgressBridge {
    /// 要上报进度的任务 ID。
    task_id: String,
    /// 用于发出进度事件的事件总线。
    event_bus: Arc<TaskEventBus>,
    /// 预期的最大迭代次数（0 表示未知）。
    max_iterations: usize,
    /// 任务开始时间，用于估算剩余时间。
    task_start: Instant,
    /// 当前迭代序号。
    current_iteration: AtomicUsize,
    /// 是否仍在发出进度事件。
    enabled: AtomicBool,
}

impl ProgressBridge {
    /**
     * 创建新的进度桥接器。
     *
     * # 参数
     * - `task_id`：要为其发出进度事件的任务 ID
     * - `event_bus`：用于发出进度事件的事件总线
     * - `max_iterations`：预期的最大迭代次数（0 表示未知，使用递减曲线）
     */
    pub fn new(task_id: String, event_bus: Arc<TaskEventBus>, max_iterations: usize) -> Self {
        Self {
            task_id,
            event_bus,
            max_iterations,
            task_start: Instant::now(),
            current_iteration: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /**
     * 停止发出进度事件，但不移除回调。
     *
     * 适用于执行已完成而回调尚未移除的情况——防止发出过期的进度事件。
     */
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// 返回此桥接器上报进度所对应的任务 ID。
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// 判断当前是否仍启用进度上报。
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// 构造 `TaskProgress` 并在启用时发送到事件总线。
    fn emit_progress(&self, pct: f64, phase: &str, message: Option<String>) {
        if !self.is_enabled() {
            return;
        }
        let eta = self.estimate_eta(pct);
        let progress = TaskProgress {
            task_id: self.task_id.clone(),
            percentage: pct.clamp(0.0, 100.0),
            current_phase: phase.to_string(),
            phase_index: self.current_iteration.load(Ordering::Relaxed),
            total_phases: self.max_iterations,
            message,
            eta_secs: eta,
            updated_at: Utc::now(),
        };
        self.event_bus.emit(TaskEvent::Progress {
            task_id: self.task_id.clone(),
            progress,
        });
    }

    /// 根据已耗时与当前百分比线性估算剩余秒数。
    fn estimate_eta(&self, pct: f64) -> Option<u64> {
        if pct <= 0.0 || pct >= 100.0 {
            return None;
        }
        let elapsed = self.task_start.elapsed().as_secs_f64();
        let total_estimated = elapsed / (pct / 100.0);
        let remaining = total_estimated - elapsed;
        Some(remaining.max(0.0) as u64)
    }
}

impl AgentCallback for ProgressBridge {
    /// 返回回调类型名，用于按类型移除回调。
    fn callback_kind(&self) -> Option<&'static str> {
        Some("ProgressBridge")
    }

    /// 以任务 ID 作为回调标识。
    fn callback_id(&self) -> Option<&str> {
        Some(&self.task_id)
    }

    /// 记录当前迭代并上报“thinking”阶段进度。
    fn on_iteration<'a>(
        &'a self,
        _agent: &'a str,
        iteration: usize,
    ) -> futures::future::BoxFuture<'a, ()> {
        self.current_iteration.store(iteration, Ordering::Relaxed);
        let pct = if self.max_iterations > 0 {
            (iteration as f64 / self.max_iterations as f64) * 100.0
        } else {
            let base = 1.0 - (0.9_f64).powi(iteration as i32);
            (base * 95.0).min(95.0)
        };
        let msg = format!("Iteration {}", iteration + 1);
        self.emit_progress(pct, "thinking", Some(msg));
        Box::pin(async {})
    }

    /// 思考开始时不上报进度。
    fn on_think_start<'a>(
        &'a self,
        _agent: &'a str,
        _messages: &'a [Message],
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 思考结束时若计划调用工具，则上报规划进度。
    fn on_think_end<'a>(
        &'a self,
        _agent: &'a str,
        steps: &'a [StepType],
        _prompt_tokens: usize,
        _completion_tokens: usize,
    ) -> futures::future::BoxFuture<'a, ()> {
        let tool_names: Vec<&str> = steps
            .iter()
            .filter_map(|s| match s {
                StepType::Call { function_name, .. } => Some(function_name.as_str()),
                _ => None,
            })
            .collect();
        if !tool_names.is_empty() {
            let iteration = self.current_iteration.load(Ordering::Relaxed);
            let pct = if self.max_iterations > 0 {
                ((iteration as f64 + 0.5) / self.max_iterations as f64) * 100.0
            } else {
                let base = 1.0 - (0.9_f64).powi(iteration as i32);
                (base * 95.0).min(95.0)
            };
            let msg = format!("Planning: {}", tool_names.join(", "));
            self.emit_progress(pct, "thinking", Some(msg));
        }
        Box::pin(async {})
    }

    /// 工具开始执行时上报以工具名为阶段的进度。
    fn on_tool_start<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        _args: &'a Value,
    ) -> futures::future::BoxFuture<'a, ()> {
        let iteration = self.current_iteration.load(Ordering::Relaxed);
        let pct = if self.max_iterations > 0 {
            ((iteration as f64 + 0.7) / self.max_iterations as f64) * 100.0
        } else {
            let base = 1.0 - (0.9_f64).powi(iteration as i32);
            (base * 95.0).min(95.0)
        };
        let msg = format!("Using: {}", tool);
        self.emit_progress(pct, tool, Some(msg));
        Box::pin(async {})
    }

    /// 工具执行结束时不上报进度。
    fn on_tool_end<'a>(
        &'a self,
        _agent: &'a str,
        _tool: &'a str,
        _result: &'a str,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    /// 工具出错时上报“error”阶段进度。
    fn on_tool_error<'a>(
        &'a self,
        _agent: &'a str,
        tool: &'a str,
        _err: &'a ReactError,
    ) -> futures::future::BoxFuture<'a, ()> {
        let iteration = self.current_iteration.load(Ordering::Relaxed);
        let pct = if self.max_iterations > 0 {
            ((iteration as f64 + 0.7) / self.max_iterations as f64) * 100.0
        } else {
            let base = 1.0 - (0.9_f64).powi(iteration as i32);
            (base * 95.0).min(95.0)
        };
        let msg = format!("Tool error: {}", tool);
        self.emit_progress(pct, "error", Some(msg));
        Box::pin(async {})
    }

    /// 产生最终答案时上报 100% 完成。
    fn on_final_answer<'a>(
        &'a self,
        _agent: &'a str,
        _answer: &'a str,
    ) -> futures::future::BoxFuture<'a, ()> {
        self.emit_progress(100.0, "completed", Some("Task completed".to_string()));
        Box::pin(async {})
    }
}
