/*!
Manager-Subagent 规划，编译为规范的、带版本控制的 Team 图。

本模块不持有任何就绪前沿（ready-frontier）循环或 checkpoint 格式。
manager 的计划会作为一次图版本（graph revision）被提交，之后
普通的运行时任务会通过
[`TaskRevisionService`](echo_orchestration::tasks::TaskRevisionService)
添加进去。
*/

use echo_core::error::{ReactError, Result};
use echo_orchestration::planning::PlanValidator;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

use super::{CompiledTeamGraph, team_task};

/// manager 规划任务的固定任务 ID。
const PLAN_TASK_ID: &str = "team-plan";
/// manager 汇总任务的固定任务 ID。
const SYNTHESIS_TASK_ID: &str = "team-synthesis";

/// manager 从 LLM 输出中解析出的原始计划。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeamManagerPlan {
    /// 计划中包含的任务列表。
    tasks: Vec<TeamManagerTask>,
}

/// manager 计划中单个任务的原始（尚未编译为运行时 id）表示。
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TeamManagerTask {
    /// manager 分配的稳定逻辑 id。
    id: String,
    /// 负责执行该任务的已注册子 Agent 名称。
    subagent: String,
    /// 任务的具体描述。
    description: String,
    /// 该任务依赖的其他逻辑 id 列表。
    #[serde(default)]
    depends_on: Vec<String>,
}

/// 构建仅包含 manager 规划任务的初始 Team 图。
pub(super) fn initial_graph(spec: &super::TeamSpec, objective: &str) -> CompiledTeamGraph {
    let member_names = spec.subagents.join(", ");
    CompiledTeamGraph {
        tasks: vec![team_task(
            PLAN_TASK_ID,
            &spec.manager,
            format!(
                "You are the Team manager. Decompose the objective into independently executable \
                 tasks for these Subagents: {member_names}.\n\nObjective:\n{objective}\n\n\
                 Write the Team plan as exactly one fenced JSON object before any framework-owned \
                 final `## Result` section, using this exact shape:\n\
                 {{\"tasks\":[{{\"id\":\"stable-logical-id\",\"subagent\":\"registered-name\",\
                 \"description\":\"concrete task\",\"depends_on\":[\"earlier-logical-id\"]}}]}}\n\
                 Every subagent must be one of the registered names above. IDs must be unique, \
                 dependencies must reference IDs in the same response, and the tasks array must not be empty."
            ),
            Vec::new(),
        )],
        terminal_task_id: PLAN_TASK_ID.to_string(),
    }
}

/// 解析 manager 生成的计划文本，并将其展开为完整的运行时 Team 图。
pub(super) fn expand_graph(
    spec: &super::TeamSpec,
    objective: &str,
    manager_plan: &str,
) -> Result<CompiledTeamGraph> {
    let plan_body = manager_plan
        .split_once("\n## Result")
        .map(|(body, _)| body)
        .unwrap_or(manager_plan);
    let plan_json = echo_core::utils::json_parse::extract_json_from_markdown(plan_body);
    let planned = serde_json::from_str::<TeamManagerPlan>(&plan_json).map_err(|error| {
        ReactError::Other(format!(
            "Team manager plan must be valid typed JSON: {error}"
        ))
    })?;
    if planned.tasks.is_empty() {
        return Err(ReactError::Other(
            "Team manager plan must contain at least one task".to_string(),
        ));
    }
    let max_planned_tasks = PlanValidator::default().max_tasks.saturating_sub(2);
    if planned.tasks.len() > max_planned_tasks {
        return Err(ReactError::Other(format!(
            "Team manager produced {} tasks; maximum is {max_planned_tasks}",
            planned.tasks.len()
        )));
    }
    if spec.subagents.is_empty() {
        return Err(ReactError::Other(
            "Manager-Subagent Team requires at least one executable Subagent".to_string(),
        ));
    }

    let registered = spec
        .subagents
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut logical_ids = HashMap::with_capacity(planned.tasks.len());
    for (index, task) in planned.tasks.iter().enumerate() {
        let logical_id = task.id.trim();
        if logical_id.is_empty() {
            return Err(ReactError::Other(
                "Team manager task IDs cannot be empty".to_string(),
            ));
        }
        if task.description.trim().is_empty() {
            return Err(ReactError::Other(format!(
                "Team manager task '{logical_id}' has an empty description"
            )));
        }
        if !registered.contains(task.subagent.as_str()) {
            return Err(ReactError::Other(format!(
                "Team manager assigned task '{logical_id}' to unknown Subagent '{}'",
                task.subagent
            )));
        }
        let runtime_id = format!("team-member-{index}");
        if logical_ids
            .insert(logical_id.to_string(), runtime_id)
            .is_some()
        {
            return Err(ReactError::Other(format!(
                "Team manager task ID '{logical_id}' is duplicated"
            )));
        }
    }

    let mut tasks = Vec::with_capacity(planned.tasks.len().saturating_add(1));
    let mut member_ids = Vec::with_capacity(planned.tasks.len());
    for task in planned.tasks {
        let runtime_id = logical_ids.get(task.id.trim()).cloned().ok_or_else(|| {
            ReactError::Other(format!("Team manager task '{}' lost its identity", task.id))
        })?;
        let mut dependencies = vec![PLAN_TASK_ID.to_string()];
        let mut seen_dependencies = HashSet::new();
        for dependency in task.depends_on {
            let dependency = dependency.trim();
            if dependency == task.id.trim() {
                return Err(ReactError::Other(format!(
                    "Team manager task '{}' cannot depend on itself",
                    task.id
                )));
            }
            if !seen_dependencies.insert(dependency.to_string()) {
                return Err(ReactError::Other(format!(
                    "Team manager task '{}' repeats dependency '{dependency}'",
                    task.id
                )));
            }
            let dependency_id = logical_ids.get(dependency).cloned().ok_or_else(|| {
                ReactError::Other(format!(
                    "Team manager task '{}' references unknown dependency '{dependency}'",
                    task.id
                ))
            })?;
            dependencies.push(dependency_id);
        }
        tasks.push(team_task(
            &runtime_id,
            &task.subagent,
            task.description,
            dependencies,
        ));
        member_ids.push(runtime_id);
    }
    tasks.push(team_task(
        SYNTHESIS_TASK_ID,
        &spec.manager,
        format!(
            "Synthesize the completed Team work into one coherent answer.\n\nOriginal objective:\n{objective}"
        ),
        member_ids,
    ));
    Ok(CompiledTeamGraph {
        tasks,
        terminal_task_id: SYNTHESIS_TASK_ID.to_string(),
    })
}

/// 返回汇总（synthesis）任务的 id。
pub(super) const fn synthesis_task_id() -> &'static str {
    SYNTHESIS_TASK_ID
}
