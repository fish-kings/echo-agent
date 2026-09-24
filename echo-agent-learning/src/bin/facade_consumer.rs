use echo_agent::agent::AgentInvocationContext;
use echo_agent::config::FrameworkConfig;
use echo_agent::paths::DataRoot;
use echo_agent::tools::{InvocationResourceGuard, StandardToolPack, ToolPack};

/// 用于标识某个资源租约的复合身份信息。
#[derive(PartialEq, Eq)]
struct LeaseIdentity {
    /// 租约所属的作用域名称。
    scope: &'static str,
    /// 租约的代次编号，用于区分同一作用域下的不同租约。
    generation: u64,
}

/// 演示如何在 `AgentInvocationContext` 中携带一个不对工具暴露的资源守卫（resource guard）。
fn main() {
    let root = DataRoot::new("./agent-data");
    let config = FrameworkConfig::default();
    let pack = StandardToolPack::new();
    let invocation = AgentInvocationContext {
        // 任何 Send + Sync 的所有者都可以被保留，且不会暴露给工具。
        resource_guards: vec![InvocationResourceGuard::new_identified(
            "example-lease".to_string(),
            LeaseIdentity {
                scope: "workspace",
                generation: 1,
            },
        )],
        ..AgentInvocationContext::default()
    };

    println!(
        "facade example: root={}, model={}, tool_pack={}, guards={}",
        root.as_path().display(),
        config.model.name,
        pack.name(),
        invocation.resource_guards.len()
    );
    assert!(
        invocation
            .resource_guards
            .first()
            .is_some_and(InvocationResourceGuard::retains::<String>)
    );
    assert!(invocation.resource_guards.first().is_some_and(|guard| {
        guard.matches_identity(&LeaseIdentity {
            scope: "workspace",
            generation: 1,
        })
    }));
}
