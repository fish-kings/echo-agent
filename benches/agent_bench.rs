/*! echo-agent 核心操作的 Criterion 基准测试。

运行方式：`cargo bench -p echo_agent`
*/

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use echo_agent::prelude::*;
use std::sync::Arc;

/// 基准测试：通过构建器（builder）构造 Agent。
fn bench_agent_build(c: &mut Criterion) {
    c.bench_function("agent_build", |b| {
        b.iter(|| {
            ReactAgentBuilder::new()
                .model("bench-model")
                .name("bench-agent")
                .system_prompt("You are a benchmark agent")
                .max_iterations(1)
                .build()
                .unwrap()
        })
    });
}

/// 基准测试：启用工具（tools）后构造 Agent。
fn bench_agent_build_with_tools(c: &mut Criterion) {
    c.bench_function("agent_build_with_tools", |b| {
        b.iter(|| {
            ReactAgentBuilder::new()
                .model("bench-model")
                .name("bench-agent")
                .system_prompt("You are a benchmark agent")
                .max_iterations(1)
                .enable_tools()
                .build()
                .unwrap()
        })
    });
}

/// 基准测试：创建 SharedAgent（移除 Mutex 之后的无锁实现）。
fn bench_shared_agent_create(c: &mut Criterion) {
    c.bench_function("shared_agent_create", |b| {
        b.iter(|| {
            let agent = ReactAgentBuilder::new()
                .model("bench")
                .name("bench")
                .system_prompt("p")
                .max_iterations(1)
                .build()
                .unwrap();
            let shared: Arc<dyn echo_agent::agent::Agent> = Arc::new(agent);
            black_box(shared)
        })
    });
}

/// 基准测试：构建一个简单的 Graph 工作流。
fn bench_workflow_build_simple(c: &mut Criterion) {
    c.bench_function("workflow_build_simple", |b| {
        b.iter(|| {
            use echo_orchestration::workflow::GraphBuilder;
            let _graph = GraphBuilder::new("bench_graph")
                .add_function_node("start", |state| {
                    Box::pin(async move {
                        let _ = state.set("key", "value");
                        Ok(())
                    })
                })
                .set_entry("start")
                .set_finish("start")
                .build()
                .unwrap();
            black_box(())
        })
    });
}

/// 基准测试：Token 预算分配计算（纯 CPU 密集型的微基准测试）。
fn bench_token_budget_allocate(c: &mut Criterion) {
    c.bench_function("token_budget_allocate", |b| {
        let budget = echo_core::budget::TokenBudget::new(128_000).unwrap_or_default();
        b.iter(|| {
            let alloc = budget.allocate(5000, 3000, 40000);
            black_box(alloc.ok())
        })
    });
}

criterion_group!(
    benches,
    bench_agent_build,
    bench_agent_build_with_tools,
    bench_shared_agent_create,
    bench_workflow_build_simple,
    bench_token_budget_allocate,
);
criterion_main!(benches);
