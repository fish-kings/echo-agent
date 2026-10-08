/*!
 * demo68_human_selection — HumanLoopProvider 选择（任务检查点）演示
 *
 * 演示统一的人在回路 Selection 类型，它取代了旧版 HumanGate。
 * 运行中的任务可以暂停，并通过与工具审批相同的 HumanLoopProvider
 * 等待人工输入。
 *
 * ```bash
 * cargo run -p echo-agent-learning --example demo68_human_gate --features subagent,human-loop
 * ```
 */

use echo_agent::human_loop::{
    ApprovalDecision, HumanLoopEvent, HumanLoopManager, HumanLoopProvider, HumanLoopRequest,
    HumanLoopResponse,
};
use std::sync::Arc;
use std::time::Duration;

/// 示例入口：依次演示选择请求、选择超时与审批请求三个场景。
#[tokio::main]
async fn main() {
    println!("=== demo68: HumanLoopProvider Selection ===\n");

    // ── 场景 1：通过 HumanLoopManager 事件循环进行选择 ─────────
    println!("--- Scenario 1: Selection via Manager events ---");
    let manager = Arc::new(HumanLoopManager::new());

    // 启动事件处理任务（模拟前端）
    let mgr = manager.clone();
    let handler = tokio::spawn(async move {
        while let Some(event) = mgr.recv_event().await {
            if let HumanLoopEvent::SelectionRequest {
                task_id,
                prompt,
                options,
                phase,
                responder,
                ..
            } = event
            {
                println!("  👀 [Frontend]  task={task_id}, phase={phase}");
                println!("  📋 [Frontend]  \"{prompt}\"");
                for (i, opt) in options.iter().enumerate() {
                    println!("  [{i}] {opt}");
                }
                // 自动选择 "Approve"
                responder.respond("Approve".to_string(), None);
                println!("  📨 [Frontend]  responded: Approve");
            }
        }
    });

    // 任务发起选择请求
    let req = HumanLoopRequest::selection(
        "task-1",
        "Review the draft and choose an action",
        vec!["Approve".into(), "Revise".into(), "Cancel".into()],
    )
    .with_context(serde_json::json!({ "draft": "Hello, world!" }))
    .with_phase("review");

    println!("  🚀 [Task]      requesting selection...");
    match manager.request(req).await {
        Ok(HumanLoopResponse::Selection {
            selection,
            instructions,
        }) => {
            println!("  ✅ [Task]      selection={selection}, instructions={instructions:?}");
        }
        Ok(other) => println!("  ⚠️  [Task]      unexpected response: {other:?}"),
        Err(e) => println!("  ❌ [Task]      error: {e}"),
    }

    drop(manager);
    let _ = handler.await;

    // ── 场景 2：超时 ──────────────────────────────────────────
    println!("\n--- Scenario 2: Selection timeout ---");
    let manager2 = Arc::new(HumanLoopManager::new());

    // 启动一个从不响应的处理任务（模拟没有前端）
    let mgr2 = manager2.clone();
    let handler2 = tokio::spawn(async move {
        while let Some(_event) = mgr2.recv_event().await {
            // 故意不响应以触发超时
        }
    });

    let req2 = HumanLoopRequest::selection(
        "task-2",
        "Approve deployment",
        vec!["Deploy".into(), "Abort".into()],
    );
    // 沿用 approval_with_timeout 模式，改用于选择请求
    let mut req2 = req2;
    req2.timeout = Some(Duration::from_millis(200));

    println!("  🚀 [Task-2]    requesting with 200ms timeout...");
    match manager2.request(req2).await {
        Ok(HumanLoopResponse::Timeout) => println!("  ⏰ [Task-2]    timed out (as expected)"),
        Ok(other) => println!("  ⚠️  [Task-2]    unexpected: {other:?}"),
        Err(e) => println!("  ❌ [Task-2]    error: {e}"),
    }

    drop(manager2);
    handler2.abort();

    // ── 场景 3：审批类型仍然可用 ────────────────────────
    println!("\n--- Scenario 3: Approval kind (backward compat) ---");
    let manager3 = Arc::new(HumanLoopManager::new());

    let mgr3 = manager3.clone();
    let handler3 = tokio::spawn(async move {
        while let Some(event) = mgr3.recv_event().await {
            if let HumanLoopEvent::ApprovalRequest {
                tool_name,
                responder,
                ..
            } = event
            {
                println!("  👀 [Frontend]  approval for tool: {tool_name}");
                responder.respond(ApprovalDecision::Approved);
            }
        }
    });

    let req3 = HumanLoopRequest::approval("write_file", serde_json::json!({"path": "test.rs"}));
    println!("  🚀 [Task-3]    requesting approval...");
    match manager3.request(req3).await {
        Ok(HumanLoopResponse::Approved) => println!("  ✅ [Task-3]    approved"),
        Ok(other) => println!("  ⚠️  [Task-3]    unexpected: {other:?}"),
        Err(e) => println!("  ❌ [Task-3]    error: {e}"),
    }

    drop(manager3);
    let _ = handler3.await;

    println!("\n🎉 === demo68 complete ===");
}
