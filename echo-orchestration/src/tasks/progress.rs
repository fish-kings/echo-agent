//! 长时间运行任务的进度追踪。
//!
//! 提供基于加权阶段的进度计算，以及通过 `tokio::sync::watch`
//! channel 实现的实时进度广播。
//!
//! # 示例
//!
//! ```rust,ignore
//! use echo_orchestration::tasks::progress::{Phase, PhasePlan, ProgressReporter};
//!
//! let plan = PhasePlan::new(vec![
//!     Phase::new("search", "Search", 2.0),
//!     Phase::new("analyze", "Analyze", 3.0),
//!     Phase::new("report", "Report", 1.0),
//! ]);
//!
//! let mut reporter = ProgressReporter::new("task-1".into(), plan);
//! reporter.enter_phase(0, Some("Starting search".into()));
//! let sub = reporter.subscribe();
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::watch;

// ── Phase ──────────────────────────────────────────────────────────

/// 长时间运行流水线中的单个阶段。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    /// 阶段的唯一标识符。
    pub id: String,
    /// 人类可读的阶段名称。
    pub name: String,
    /// 用于进度计算的相对权重（越大表示按比例耗时越长）。
    pub weight: f64,
    /// 该阶段的最大重试次数（0 表示不重试）。
    pub max_retries: u32,
    /// 该阶段的超时时间，单位为秒（0 表示不设超时）。
    pub timeout_secs: u64,
    /// 该阶段在继续之前是否需要人工审批。
    pub human_checkpoint: bool,
}

impl Phase {
    /// 创建一个仅包含 id、名称和权重的简单阶段。
    pub fn new(id: impl Into<String>, name: impl Into<String>, weight: f64) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            weight,
            max_retries: 0,
            timeout_secs: 0,
            human_checkpoint: false,
        }
    }

    /// 设置该阶段的最大重试次数。
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// 设置该阶段的超时时间。
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// 将该阶段标记为需要人工检查点。
    pub fn with_human_checkpoint(mut self, checkpoint: bool) -> Self {
        self.human_checkpoint = checkpoint;
        self
    }
}

// ── PhasePlan ──────────────────────────────────────────────────────

/// 带累计加权进度计算能力的有序阶段列表。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePlan {
    /// 按执行顺序排列的各个阶段。
    pub phases: Vec<Phase>,
    /// 预先计算好的总权重（所有阶段权重之和）。
    total_weight: f64,
}

impl PhasePlan {
    /// 根据有序的阶段列表创建一个新的阶段计划。
    pub fn new(phases: Vec<Phase>) -> Self {
        let total_weight = phases.iter().map(|p| p.weight).sum();
        Self {
            phases,
            total_weight,
        }
    }

    /// 根据当前阶段索引及阶段内部进度（0.0–1.0），计算总体进度百分比（0.0–100.0）。
    ///
    /// `current_phase_idx` 之前的所有阶段都视为已完成 100%。
    /// 当前阶段贡献 `phase_internal_pct * current_weight`。
    pub fn progress_pct(&self, current_phase_idx: usize, phase_internal_pct: f64) -> f64 {
        if self.total_weight <= 0.0 || self.phases.is_empty() {
            return 0.0;
        }
        let completed_weight: f64 = self.phases[..current_phase_idx.min(self.phases.len())]
            .iter()
            .map(|p| p.weight)
            .sum();
        let current_weight = self
            .phases
            .get(current_phase_idx)
            .map(|p| p.weight)
            .unwrap_or(0.0);
        let partial = current_weight * phase_internal_pct.clamp(0.0, 1.0);
        ((completed_weight + partial) / self.total_weight * 100.0).clamp(0.0, 100.0)
    }

    /// 获取给定索引处阶段的名称，越界则返回 `""`。
    pub fn phase_name(&self, idx: usize) -> &str {
        self.phases.get(idx).map(|p| p.name.as_str()).unwrap_or("")
    }

    /// 阶段数量。
    pub fn len(&self) -> usize {
        self.phases.len()
    }

    /// 该计划是否不包含任何阶段。
    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }
}

// ── TaskProgress ───────────────────────────────────────────────────

/// 长时间运行任务的实时进度快照。
///
/// 可序列化，便于通过 SSE、WebSocket 或其他 channel 传输。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    /// 该进度所属的任务。
    pub task_id: String,
    /// 总体进度百分比（0.0–100.0）。
    pub percentage: f64,
    /// 当前正在执行的阶段名称。
    pub current_phase: String,
    /// 当前阶段的从零开始的索引。
    pub phase_index: usize,
    /// 计划中的总阶段数。
    pub total_phases: usize,
    /// 可选的阶段内部消息（例如 "正在搜索论文：已找到 12/20 篇"）。
    pub message: Option<String>,
    /// 根据已耗时和进度估算的剩余秒数。
    pub eta_secs: Option<u64>,
    /// 本次进度更新的时间戳。
    #[serde(with = "echo_core::utils::time::local_rfc3339")]
    pub updated_at: DateTime<Utc>,
}

// ── ProgressReporter ───────────────────────────────────────────────

/// 通过 `watch` channel 向订阅者广播进度更新。
///
/// 采用“最新值”语义：订阅者总能拿到最新的快照。
/// 这与使用广播（全事件）语义的 `TaskEventBus` 形成互补。
pub struct ProgressReporter {
    /// 进度更新的发送端
    sender: watch::Sender<TaskProgress>,
    /// 进度更新的接收端（用于派生新的订阅者）
    receiver: watch::Receiver<TaskProgress>,
    /// 阶段计划
    plan: PhasePlan,
    /// 整个任务的起始时刻
    task_start: Instant,
    /// 当前阶段的起始时刻
    phase_start: Instant,
}

impl ProgressReporter {
    /// 为给定任务及阶段计划创建一个新的进度报告器。
    pub fn new(task_id: String, plan: PhasePlan) -> Self {
        let initial = TaskProgress {
            task_id,
            percentage: 0.0,
            current_phase: plan.phase_name(0).to_string(),
            phase_index: 0,
            total_phases: plan.len(),
            message: None,
            eta_secs: None,
            updated_at: Utc::now(),
        };
        let (sender, receiver) = watch::channel(initial);
        let now = Instant::now();
        Self {
            sender,
            receiver,
            plan,
            task_start: now,
            phase_start: now,
        }
    }

    /// 在进入新阶段时调用。
    pub fn enter_phase(&mut self, phase_idx: usize, message: Option<String>) {
        self.phase_start = Instant::now();
        let pct = self.plan.progress_pct(phase_idx, 0.0);
        let task_id = self.sender.borrow().task_id.clone();
        let _ = self.sender.send(TaskProgress {
            task_id,
            percentage: pct,
            current_phase: self.plan.phase_name(phase_idx).to_string(),
            phase_index: phase_idx,
            total_phases: self.plan.len(),
            message,
            eta_secs: self.calculate_eta(pct),
            updated_at: Utc::now(),
        });
    }

    /// 用于阶段内部进度更新。
    pub fn update_phase_progress(&self, phase_pct: f64, message: Option<String>) {
        let current = self.sender.borrow();
        let pct = self.plan.progress_pct(current.phase_index, phase_pct);
        let task_id = current.task_id.clone();
        let current_phase = current.current_phase.clone();
        let phase_index = current.phase_index;
        let total_phases = current.total_phases;
        drop(current);
        let _ = self.sender.send(TaskProgress {
            task_id,
            percentage: pct,
            current_phase,
            phase_index,
            total_phases,
            message,
            eta_secs: self.calculate_eta(pct),
            updated_at: Utc::now(),
        });
    }

    /// 订阅进度更新（用于 SSE/WebSocket 流式推送）。
    ///
    /// 返回一个始终持有最新快照的 `watch::Receiver`。
    pub fn subscribe(&self) -> watch::Receiver<TaskProgress> {
        self.receiver.clone()
    }

    /// 获取当前的进度快照。
    pub fn current(&self) -> TaskProgress {
        self.sender.borrow().clone()
    }

    /// 根据已耗时和进度百分比估算剩余时间。
    fn calculate_eta(&self, pct: f64) -> Option<u64> {
        if pct <= 0.0 {
            return None;
        }
        let elapsed = self.task_start.elapsed().as_secs();
        let total_estimated = (elapsed as f64 / pct * 100.0) as u64;
        Some(total_estimated.saturating_sub(elapsed))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个用于测试的示例阶段计划
    fn sample_plan() -> PhasePlan {
        PhasePlan::new(vec![
            Phase::new("search", "Search", 2.0),
            Phase::new("analyze", "Analyze", 3.0),
            Phase::new("report", "Report", 1.0),
        ])
    }

    #[test]
    fn test_phase_plan_progress_at_start() {
        let plan = sample_plan();
        let pct = plan.progress_pct(0, 0.0);
        assert!((pct - 0.0).abs() < 0.01, "Expected 0%, got {pct}%");
    }

    #[test]
    fn test_phase_plan_progress_mid_first_phase() {
        let plan = sample_plan();
        let pct = plan.progress_pct(0, 0.5);
        // 阶段权重 2.0 * 0.5 / 6.0 = 16.67%
        assert!((pct - 16.67).abs() < 0.5, "Expected ~16.67%, got {pct}%");
    }

    #[test]
    fn test_phase_plan_progress_after_first_phase() {
        let plan = sample_plan();
        // 进入阶段 1 意味着阶段 0 已经 100% 完成
        let pct = plan.progress_pct(1, 0.0);
        // 2.0 / 6.0 = 33.33%
        assert!((pct - 33.33).abs() < 0.5, "Expected ~33.33%, got {pct}%");
    }

    #[test]
    fn test_phase_plan_progress_complete() {
        let plan = sample_plan();
        // 所有阶段完成 = 索引超出末尾
        let pct = plan.progress_pct(3, 0.0);
        assert!((pct - 100.0).abs() < 0.01, "Expected 100%, got {pct}%");
    }

    #[test]
    fn test_phase_plan_empty() {
        let plan = PhasePlan::new(vec![]);
        assert_eq!(plan.progress_pct(0, 0.5), 0.0);
        assert!(plan.is_empty());
    }

    #[tokio::test]
    async fn test_progress_reporter_subscribe() {
        let plan = sample_plan();
        let mut reporter = ProgressReporter::new("task-1".into(), plan);
        let sub = reporter.subscribe();

        reporter.enter_phase(1, Some("Starting analysis".into()));

        // 订阅者应当能看到这次更新
        assert!(sub.has_changed().unwrap_or(true));
        let progress = sub.borrow().clone();
        assert_eq!(progress.task_id, "task-1");
        assert_eq!(progress.phase_index, 1);
        assert_eq!(progress.current_phase, "Analyze");
        assert!(progress.percentage > 30.0);
    }
}
