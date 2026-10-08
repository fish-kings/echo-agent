/*!
 * demo67_progress.rs — 进度跟踪 + 任务元数据（P0 + P1）
 *
 * 演示内容：
 *   1. 带权重阶段的 `PhasePlan`
 *   2. 通过 `watch` 通道广播进度的 `ProgressReporter`
 *   3. 存放在 `TaskSpec::extension` 中的产品自有数据
 *   4. 经由 `TaskEventBus` 流转的 `TaskEvent::Progress`
 *
 * 契约测试：`contract_demo67_progress`。
 */

use echo_agent::tasks::{Phase, PhasePlan, ProgressReporter, TaskEvent, TaskEventBus, TaskSpec};
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};

// ── 强类型元数据结构 ───────────────────────────────────────────────

/// 调研任务参数，作为产品自有数据序列化到 `TaskSpec::extension`。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResearchParams {
    /// 调研主题。
    topic: String,
    /// 最多检索的论文数量。
    max_papers: usize,
    /// 是否要求经过同行评审。
    require_peer_review: bool,
}

/// demo67 契约测试入口：演示阶段进度广播与任务扩展数据。
#[tokio::test]
async fn contract_demo67_progress() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 === demo67: Progress Tracking + Task Extension ===\n");

    // ── 1. 构建包含 3 个带权阶段的 PhasePlan ────────────────
    let plan = PhasePlan::new(vec![
        Phase::new("search", "🔍 Search", 2.0),
        Phase::new("analyze", "📊 Analyze", 3.0),
        Phase::new("report", "📝 Report", 1.0),
    ]);
    println!(
        "📋 PhasePlan created: {} phases, total weight = {:.1}",
        plan.len(),
        plan.phases.iter().map(|p| p.weight).sum::<f64>()
    );

    // ── 2. ProgressReporter + 订阅者 ───────────────────────────
    let mut reporter = ProgressReporter::new("task-42".into(), plan);
    let mut sub = reporter.subscribe();

    // ── 3. TaskEventBus + 事件订阅者 ─────────────────────────
    let bus = TaskEventBus::new();
    let mut bus_rx = bus.subscribe();

    // 启动一个推进进度的子 agent
    let bus_clone = bus.clone();
    let subagent = tokio::spawn(async move {
        // 阶段 0：搜索
        reporter.enter_phase(0, Some("Querying arxiv…".into()));
        emit_progress(&reporter, &bus_clone);
        for step in 1..=4 {
            sleep(Duration::from_millis(50)).await;
            reporter.update_phase_progress(
                step as f64 / 4.0,
                Some(format!("Found {} papers", step * 5)),
            );
            emit_progress(&reporter, &bus_clone);
        }

        // 阶段 1：分析
        reporter.enter_phase(1, Some("Running citation analysis…".into()));
        emit_progress(&reporter, &bus_clone);
        for step in 1..=6 {
            sleep(Duration::from_millis(40)).await;
            reporter.update_phase_progress(
                step as f64 / 6.0,
                Some(format!("Analyzed {}/30 papers", step * 5)),
            );
            emit_progress(&reporter, &bus_clone);
        }

        // 阶段 2：报告
        reporter.enter_phase(2, Some("Generating summary…".into()));
        emit_progress(&reporter, &bus_clone);
        sleep(Duration::from_millis(60)).await;
        reporter.update_phase_progress(1.0, Some("Report complete".into()));
        emit_progress(&reporter, &bus_clone);
    });

    // ── 4. 从 watch 订阅者打印进度 ───────────────
    let printer = tokio::spawn(async move {
        while let Ok(()) = sub.changed().await {
            {
                let p = sub.borrow();
                let eta = p.eta_secs.map(|s| format!(" ETA {s}s")).unwrap_or_default();
                println!(
                    "  📡 [{}/{}] {} — {:.1}%{}",
                    p.phase_index + 1,
                    p.total_phases,
                    p.current_phase,
                    p.percentage,
                    eta,
                );
                if let Some(msg) = &p.message {
                    println!("      └─ {msg}");
                }
                if p.percentage >= 100.0 {
                    break;
                }
            }
        }
    });

    // 并行消费若干总线事件
    let bus_printer = tokio::spawn(async move {
        let mut count = 0u32;
        while let Ok(event) = bus_rx.recv().await {
            let TaskEvent::Progress { task_id, progress } = event.as_ref();
            if count.is_multiple_of(3) {
                println!(
                    "  🔔 bus: {} @ {:.1}% ({})",
                    task_id, progress.percentage, progress.current_phase
                );
            }
            count += 1;
        }
    });

    let _ = tokio::join!(subagent, printer);
    // 给总线打印任务一点时间消费完，然后丢弃总线使其退出
    sleep(Duration::from_millis(50)).await;
    drop(bus);
    let _ = bus_printer.await;

    // ── 5. 产品自有的强类型扩展数据 ─────────────────────────────────
    println!("\n--- Task Extension ---");
    let extension = serde_json::to_value(ResearchParams {
        topic: "Retrieval-Augmented Generation".into(),
        max_papers: 20,
        require_peer_review: true,
    })?;
    let task = TaskSpec {
        id: "research-01".to_string(),
        title: "RAG literature review".to_string(),
        description: "Literature review on RAG systems".to_string(),
        depends_on: Vec::new(),
        max_retries: 2,
        extension,
    };

    println!("🏷️  Task id: {}", task.id);
    println!("📦 extension: {}", task.extension);

    // ── 6. 在应用边界解码产品数据 ───────────────
    let params: ResearchParams = serde_json::from_value(task.extension.clone())?;
    println!(
        "✅ extension → topic={:?}, max_papers={}, peer_review={}",
        params.topic, params.max_papers, params.require_peer_review
    );

    println!("\n🎉 === demo67 complete ===");
    Ok(())
}

/// 辅助函数：获取 reporter 当前进度快照，并以 `TaskEvent::Progress` 形式发出。
fn emit_progress(reporter: &ProgressReporter, bus: &TaskEventBus) {
    let progress = reporter.current();
    bus.emit(TaskEvent::Progress {
        task_id: progress.task_id.clone(),
        progress,
    });
}
