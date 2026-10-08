/*!
Tasks facade（任务门面模块）

本模块是对 `echo_orchestration::tasks` 的公共 SDK 重新导出。
*/

/// 直接从 `echo_orchestration::tasks` 重新导出。
pub mod orchestration {
    pub use echo_orchestration::tasks::*;
}

pub use echo_orchestration::tasks::*;

/**
将 Agent 的任务关联工具替换为由 `service` 支撑的工具。
工具注册是按名称进行的，因此这一操作能够原子化地
切换所使用的 store/policy 服务，而不会暴露出第二套任务 API。
*/
pub fn register_task_tools(
    agent: &mut crate::agent::ReactAgent,
    service: std::sync::Arc<TaskRevisionService>,
) {
    /*
    保留下来的权威来源必须与工具保持一致：宿主侧的接口
    会读取 `ReactAgent::task_revision_service()`，并且必须
    观察到与已注册工具实际使用的完全相同的那个服务。
    */
    agent.set_task_revision_service(service);
    for tool in build_task_tools(agent.task_revision_service().clone()) {
        agent.replace_tool(tool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证注册任务工具会替换工具管理器使用的默认任务服务。
    #[tokio::test]
    async fn register_task_tools_replaces_default_service_used_by_tool_manager()
    -> Result<(), String> {
        let mut agent = crate::agent::ReactAgent::new(crate::agent::AgentConfig::minimal(
            "test-model",
            "task-tool-replacement",
        ));
        if agent.tool_manager().get_tool("task_create").is_none() {
            return Err("ReactAgent default task_create tool is missing".to_string());
        }

        let replacement_service = std::sync::Arc::new(TaskRevisionService::new(
            std::sync::Arc::new(InMemoryRevisionedTaskStore::new()),
            std::sync::Arc::new(DefaultTaskToolPolicy::new("replacement-scope")),
        ));
        register_task_tools(&mut agent, replacement_service.clone());

        let parameters = serde_json::from_value(serde_json::json!({
            "tasks": [{
                "id": "replacement-task",
                "title": "Use the replacement task store",
                "description": "Prove ToolManager dispatches to the supplied service",
                "extension": {
                    "kind": "investigation",
                    "acceptance_criteria": ["replacement graph is persisted"]
                }
            }]
        }))
        .map_err(|error| error.to_string())?;
        let result = agent
            .tool_manager()
            .execute_tool("task_create", parameters)
            .await
            .map_err(|error| error.to_string())?;
        if !result.success {
            return Err(format!("replacement task_create failed: {}", result.output));
        }

        let graph = replacement_service
            .load("replacement-scope")
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "replacement service did not persist the task graph".to_string())?;
        let task = graph
            .snapshot
            .tasks
            .first()
            .ok_or_else(|| "replacement graph contains no tasks".to_string())?;
        assert_eq!(task.spec.id, "replacement-task");
        assert_eq!(
            task.spec
                .extension
                .get("kind")
                .and_then(serde_json::Value::as_str),
            Some("investigation")
        );
        Ok(())
    }
}
