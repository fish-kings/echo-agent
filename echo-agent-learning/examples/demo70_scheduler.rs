/*!
 * demo70_scheduler.rs — Cron 任务调度器演示
 *
 * 演示 scheduler 模块：CronTask 定义、CronTaskStore 持久化，
 * 以及 SchedulerRunner 管理 API。
 *
 * 运行：cargo run -p echo-agent-learning --example demo70_scheduler
 */

use echo_agent::scheduler::{
    CronTask, CronTaskStatus, CronTaskStore, OccurrenceFireFn, SchedulerRunner,
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// 示例入口：演示 cron 任务的存储、校验、手动触发与管理。
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⏰ === demo70: Scheduler Module ===\n");

    // ── 1. 创建基于文件的 CronTaskStore（临时路径） ──────────────
    let tmp = tempfile::tempdir()?;
    let store_path = tmp.path().join("tasks.json");
    let store = CronTaskStore::new().with_path(store_path.clone());
    println!("📁 Store path: {}", store_path.display());

    // ── 2. 添加两个调度周期不同的 CronTask ─────────────────
    let daily = CronTask::new("daily-backup", "0 2 * * *", "Run nightly database backup");
    let weekly = CronTask::new(
        "weekly-report",
        "0 9 * * 1",
        "Generate weekly analytics report",
    );
    store.add(daily.clone()).await?;
    store.add(weekly.clone()).await?;
    println!(
        "✅ Added 2 cron tasks: '{}' and '{}'\n",
        daily.name, weekly.name
    );

    // ── 3. 校验 cron 表达式 ──────────────────────────────────
    println!("🔍 Cron validation:");
    for t in [&daily, &weekly] {
        println!(
            "   '{}' ({}) → valid: {}",
            t.name,
            t.cron_expr,
            t.validate_cron()
        );
    }
    let bad = CronTask::new("bad-task", "not a cron", "test");
    println!(
        "   '{}' ({}) → valid: {}\n",
        bad.name,
        bad.cron_expr,
        bad.validate_cron()
    );

    // ── 4. 展示 next_run() 计算结果 ────────────────────────────────
    println!("📅 Next run times:");
    for t in [&daily, &weekly] {
        if let Some(next) = t.next_run() {
            println!("   '{}' → {}", t.name, next.format("%Y-%m-%d %H:%M %Z"));
        }
    }
    println!("   '{}' → {:?}\n", bad.name, bad.next_run());

    // ── 5. 使用简单的触发函数创建 SchedulerRunner ───────
    let cancel = CancellationToken::new();
    let fire_fn: OccurrenceFireFn = Arc::new(|invocation| {
        let name = invocation.task.name.clone();
        let occurrence_id = invocation.occurrence_id;
        Box::pin(async move {
            println!("   🔥 Firing task: {} ({})", name, occurrence_id);
            Ok(format!("Executed: {}", name))
        })
    });
    let runner =
        Arc::new(SchedulerRunner::new_with_occurrence_context(store, cancel, fire_fn).await?);
    println!("🤖 SchedulerRunner created (fire fn prints task name)\n");

    // ── 6. 通过 run_once() 手动触发任务 ─────────────────────
    println!("▶️  Manual trigger (run_once):");
    let result = runner.run_once(&daily.id).await?;
    println!("   Result: {}\n", result);

    // ── 7. list_tasks()、set_status()、remove_task() ──────────────────
    println!("📋 All tasks:");
    for t in runner.list_tasks().await {
        println!("   [{:?}] {} — {}", t.status, t.name, t.cron_expr);
    }

    println!("\n🚫 Disabling '{}'...", daily.name);
    runner
        .set_status(&daily.id, CronTaskStatus::Disabled)
        .await?;

    println!("📋 Tasks after disable:");
    for t in runner.list_tasks().await {
        println!("   [{:?}] {} — {}", t.status, t.name, t.cron_expr);
    }

    println!("\n🗑️  Removing '{}'...", weekly.name);
    let removed = runner.remove_task(&weekly.id).await?;
    println!("   Removed: {}", removed);

    println!("📋 Remaining tasks:");
    for t in runner.list_tasks().await {
        println!("   [{:?}] {} — {}", t.status, t.name, t.cron_expr);
    }

    println!("\n✅ === demo70 complete ===");
    Ok(())
}
