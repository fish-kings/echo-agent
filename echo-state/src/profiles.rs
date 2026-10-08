/*!
Agent 与用户画像 —— 动态的能力与偏好模型。

提供 `AgentProfile`（跟踪技能熟练度和工具使用统计）
与 `UserProfile`（跟踪用户偏好、擅长领域和常见任务）。
两者都通过框架的 `Store` trait 持久化。
*/

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use echo_core::memory::Store;

use crate::skill_telemetry::SkillTelemetry;

/// Agent 能力画像 —— 随时间跟踪技能熟练度和工具使用情况。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// 按技能划分的能力评分。
    pub capabilities: HashMap<String, CapabilityScore>,
    /// 按工具划分的使用统计。
    pub tool_usage: HashMap<String, ToolUsageStats>,
    /// 最后更新时间戳（epoch 毫秒）。
    pub last_updated: u64,
}

impl AgentProfile {
    /// 创建一个新的空画像。
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
            tool_usage: HashMap::new(),
            last_updated: epoch_millis(),
        }
    }

    /// 根据一批技能遥测数据更新能力评分。
    pub fn update_from_telemetry(&mut self, telemetry: &[SkillTelemetry]) {
        for t in telemetry {
            let score = self
                .capabilities
                .entry(t.skill_name.clone())
                .or_insert_with(|| CapabilityScore {
                    skill_name: t.skill_name.clone(),
                    proficiency: 0.0,
                    usage_count: 0,
                    success_rate: 0.0,
                });

            score.usage_count = t.activation_count;
            score.success_rate = t.success_rate();

            // 熟练度 = 成功率 × min(1.0, log10(使用次数) / 2.0)
            let usage_factor = if t.activation_count > 1 {
                (t.activation_count as f64).log10() / 2.0
            } else {
                0.0
            };
            score.proficiency = score.success_rate * usage_factor.min(1.0);
        }

        // 根据遥测数据中的 common_tools 更新工具使用情况
        for t in telemetry {
            for (tool_name, count) in &t.common_tools {
                let stats =
                    self.tool_usage
                        .entry(tool_name.clone())
                        .or_insert_with(|| ToolUsageStats {
                            usage_count: 0,
                            success_count: 0,
                            common_skills: Vec::new(),
                        });
                stats.usage_count += count;
                // 根据技能成功率估算成功次数
                stats.success_count += (*count as f64 * t.success_rate()) as u64;
                if !stats.common_skills.contains(&t.skill_name) {
                    stats.common_skills.push(t.skill_name.clone());
                }
            }
        }

        self.last_updated = epoch_millis();
    }

    /// 获取按熟练度降序排列的前 N 项能力。
    pub fn top_capabilities(&self, n: usize) -> Vec<&CapabilityScore> {
        let mut caps: Vec<_> = self.capabilities.values().collect();
        caps.sort_by(|a, b| {
            b.proficiency
                .partial_cmp(&a.proficiency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        caps.truncate(n);
        caps
    }

    /// 获取按使用次数降序排列的前 N 个工具。
    pub fn top_tools(&self, n: usize) -> Vec<&ToolUsageStats> {
        let mut tools: Vec<_> = self.tool_usage.values().collect();
        tools.sort_by_key(|tool| std::cmp::Reverse(tool.usage_count));
        tools.truncate(n);
        tools
    }

    /// 格式化为用于注入 LLM 系统提示词的文本块。
    pub fn to_prompt_block(&self) -> String {
        if self.capabilities.is_empty() && self.tool_usage.is_empty() {
            return String::new();
        }

        let mut block = String::from("<agent-profile>\n");

        let caps = self.top_capabilities(5);
        if !caps.is_empty() {
            block.push_str("## Capabilities\n");
            for cap in &caps {
                block.push_str(&format!(
                    "- {}: {:.0}% proficiency ({} uses, {:.0}% success)\n",
                    cap.skill_name,
                    cap.proficiency * 100.0,
                    cap.usage_count,
                    cap.success_rate * 100.0,
                ));
            }
        }

        let tools = self.top_tools(5);
        if !tools.is_empty() {
            block.push_str("\n## Tool Expertise\n");
            for tool in &tools {
                let tool_name = self
                    .tool_usage
                    .iter()
                    .find(|(_, v)| std::ptr::eq(*v, *tool))
                    .map(|(k, _)| k.as_str())
                    .unwrap_or("unknown");
                block.push_str(&format!(
                    "- {}: {} uses (contexts: {})\n",
                    tool_name,
                    tool.usage_count,
                    if tool.common_skills.is_empty() {
                        "general".to_string()
                    } else {
                        tool.common_skills.join(", ")
                    },
                ));
            }
        }

        block.push_str("</agent-profile>");
        block
    }
}

impl Default for AgentProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// 单项技能的能力评分。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityScore {
    /// 技能名称。
    pub skill_name: String,
    /// 熟练度评分（0.0 到 1.0），由 成功率 × log(使用次数) 计算得出。
    pub proficiency: f64,
    /// 总激活次数。
    pub usage_count: u64,
    /// 成功率（0.0 到 1.0）。
    pub success_rate: f64,
}

/// 单个工具的使用统计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageStats {
    /// 所有技能中该工具的总使用次数。
    pub usage_count: u64,
    /// 估算的成功使用次数。
    pub success_count: u64,
    /// 常用到该工具的技能列表。
    pub common_skills: Vec<String>,
}

/// 用户偏好画像 —— 跟踪用户偏好、擅长领域和常见任务。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// 用户设置的偏好（键值对）。
    pub preferences: HashMap<String, String>,
    /// 擅长的领域。
    pub expertise_areas: Vec<String>,
    /// 常见任务模式及其频次。
    pub common_tasks: Vec<TaskPattern>,
    /// 最后更新时间戳（epoch 毫秒）。
    pub last_updated: u64,
}

impl UserProfile {
    /// 创建一个新的空画像。
    pub fn new() -> Self {
        Self {
            preferences: HashMap::new(),
            expertise_areas: Vec::new(),
            common_tasks: Vec::new(),
            last_updated: epoch_millis(),
        }
    }

    /// 记录一次任务类型的发生。
    pub fn record_task(&mut self, task_type: &str) {
        if let Some(pattern) = self
            .common_tasks
            .iter_mut()
            .find(|p| p.task_type == task_type)
        {
            pattern.frequency += 1;
            pattern.last_performed = epoch_millis();
        } else {
            self.common_tasks.push(TaskPattern {
                task_type: task_type.to_string(),
                frequency: 1,
                last_performed: epoch_millis(),
            });
        }
        self.last_updated = epoch_millis();
    }

    /// 设置一项用户偏好。
    pub fn set_preference(&mut self, key: &str, value: &str) {
        self.preferences.insert(key.to_string(), value.to_string());
        self.last_updated = epoch_millis();
    }

    /// 获取按频次降序排列的前 N 个常见任务。
    pub fn top_tasks(&self, n: usize) -> Vec<&TaskPattern> {
        let mut tasks: Vec<_> = self.common_tasks.iter().collect();
        tasks.sort_by_key(|task| std::cmp::Reverse(task.frequency));
        tasks.truncate(n);
        tasks
    }

    /// 格式化为用于注入 LLM 系统提示词的文本块。
    pub fn to_prompt_block(&self) -> String {
        if self.preferences.is_empty()
            && self.expertise_areas.is_empty()
            && self.common_tasks.is_empty()
        {
            return String::new();
        }

        let mut block = String::from("<user-profile>\n");

        if !self.preferences.is_empty() {
            block.push_str("## Preferences\n");
            for (key, value) in &self.preferences {
                block.push_str(&format!("- {}: {}\n", key, value));
            }
        }

        if !self.expertise_areas.is_empty() {
            block.push_str("\n## Expertise\n");
            block.push_str(&self.expertise_areas.join(", "));
            block.push('\n');
        }

        let tasks = self.top_tasks(5);
        if !tasks.is_empty() {
            block.push_str("\n## Common Tasks\n");
            for task in &tasks {
                block.push_str(&format!("- {} ({}x)\n", task.task_type, task.frequency));
            }
        }

        block.push_str("</user-profile>");
        block
    }
}

impl Default for UserProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// 一种反复出现的任务模式。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPattern {
    /// 任务类型标识（例如 "coding"、"data-analysis"）。
    pub task_type: String,
    /// 该任务类型已执行的次数。
    pub frequency: u64,
    /// 最后一次执行的时间戳（epoch 毫秒）。
    pub last_performed: u64,
}

/// 基于框架 `Store` trait 实现的画像数据存储。
pub struct ProfileStore {
    /// 底层的通用 KV 存储。
    store: Arc<dyn Store>,
}

impl ProfileStore {
    /// 用给定的 `Store` 创建一个新的画像存储。
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }

    /// 加载 Agent 画像。
    pub async fn load_agent_profile(&self) -> echo_core::error::Result<Option<AgentProfile>> {
        match self.store.get(&["agent", "profile"], "current").await {
            Ok(Some(item)) => {
                let profile: AgentProfile = serde_json::from_value(item.value).map_err(|e| {
                    echo_core::error::ReactError::Other(format!(
                        "Failed to deserialize agent profile: {}",
                        e
                    ))
                })?;
                Ok(Some(profile))
            }
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }

    /// 保存 Agent 画像。
    pub async fn save_agent_profile(&self, profile: &AgentProfile) -> echo_core::error::Result<()> {
        let value = serde_json::to_value(profile).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to serialize agent profile: {}", e))
        })?;
        self.store
            .put(&["agent", "profile"], "current", value)
            .await
            .map_err(|e| {
                echo_core::error::ReactError::Other(format!("Failed to save agent profile: {}", e))
            })
    }

    /// 加载用户画像。
    pub async fn load_user_profile(&self) -> echo_core::error::Result<Option<UserProfile>> {
        match self.store.get(&["user", "profile"], "current").await {
            Ok(Some(item)) => {
                let profile: UserProfile = serde_json::from_value(item.value).map_err(|e| {
                    echo_core::error::ReactError::Other(format!(
                        "Failed to deserialize user profile: {}",
                        e
                    ))
                })?;
                Ok(Some(profile))
            }
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }

    /// 保存用户画像。
    pub async fn save_user_profile(&self, profile: &UserProfile) -> echo_core::error::Result<()> {
        let value = serde_json::to_value(profile).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to serialize user profile: {}", e))
        })?;
        self.store
            .put(&["user", "profile"], "current", value)
            .await
            .map_err(|e| {
                echo_core::error::ReactError::Other(format!("Failed to save user profile: {}", e))
            })
    }
}

/// 获取当前时间的 epoch 毫秒数。
fn epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_telemetry::SkillTelemetry;

    /// 构造一条用于测试的技能遥测记录。
    fn make_telemetry(name: &str, count: u64, success: u64) -> SkillTelemetry {
        SkillTelemetry {
            skill_name: name.to_string(),
            activation_count: count,
            success_count: success,
            failure_count: count - success,
            total_duration_ms: count * 3000,
            common_tools: {
                let mut m = HashMap::new();
                m.insert("bash".to_string(), count);
                m.insert("read_file".to_string(), count / 2);
                m
            },
            common_failures: Vec::new(),
            recent_records: std::collections::VecDeque::new(),
            last_used: epoch_millis(),
            first_used: epoch_millis() - 86400000,
        }
    }

    #[test]
    fn test_agent_profile_new() {
        let p = AgentProfile::new();
        assert!(p.capabilities.is_empty());
        assert!(p.tool_usage.is_empty());
    }

    #[test]
    fn test_agent_profile_update_from_telemetry() {
        let mut p = AgentProfile::new();
        let telemetry = vec![
            make_telemetry("coding", 100, 92),
            make_telemetry("paper-search", 10, 8),
        ];
        p.update_from_telemetry(&telemetry);

        assert_eq!(p.capabilities.len(), 2);
        assert_eq!(p.tool_usage.len(), 2);

        let coding = p.capabilities.get("coding").unwrap();
        assert_eq!(coding.usage_count, 100);
        assert!((coding.success_rate - 0.92).abs() < f64::EPSILON);
        assert!(coding.proficiency > 0.0);

        // bash 被两个技能共用
        let bash = p.tool_usage.get("bash").unwrap();
        assert_eq!(bash.usage_count, 110); // 100 + 10
    }

    #[test]
    fn test_agent_profile_top_capabilities() {
        let mut p = AgentProfile::new();
        let telemetry = vec![
            make_telemetry("coding", 100, 95),
            make_telemetry("paper-search", 5, 4),
            make_telemetry("data-viz", 50, 45),
        ];
        p.update_from_telemetry(&telemetry);

        let top = p.top_capabilities(2);
        assert_eq!(top.len(), 2);
        // coding 应排第一（因使用次数和成功率都很高，熟练度最高）
        assert_eq!(top[0].skill_name, "coding");
    }

    #[test]
    fn test_agent_profile_prompt_block() {
        let mut p = AgentProfile::new();
        let telemetry = vec![make_telemetry("coding", 100, 95)];
        p.update_from_telemetry(&telemetry);

        let block = p.to_prompt_block();
        assert!(block.contains("<agent-profile>"));
        assert!(block.contains("coding"));
        assert!(block.contains("Capabilities"));
        assert!(block.contains("</agent-profile>"));
    }

    #[test]
    fn test_agent_profile_empty_prompt() {
        let p = AgentProfile::new();
        assert!(p.to_prompt_block().is_empty());
    }

    #[test]
    fn test_user_profile_new() {
        let p = UserProfile::new();
        assert!(p.preferences.is_empty());
        assert!(p.expertise_areas.is_empty());
        assert!(p.common_tasks.is_empty());
    }

    #[test]
    fn test_user_profile_set_preference() {
        let mut p = UserProfile::new();
        p.set_preference("language", "zh-CN");
        p.set_preference("code_style", "concise");
        assert_eq!(p.preferences.len(), 2);
        assert_eq!(p.preferences.get("language").unwrap(), "zh-CN");
    }

    #[test]
    fn test_user_profile_record_task() {
        let mut p = UserProfile::new();
        p.record_task("coding");
        p.record_task("coding");
        p.record_task("data-analysis");
        assert_eq!(p.common_tasks.len(), 2);
        assert_eq!(p.common_tasks[0].frequency, 2);
    }

    #[test]
    fn test_user_profile_top_tasks() {
        let mut p = UserProfile::new();
        p.record_task("coding");
        p.record_task("coding");
        p.record_task("coding");
        p.record_task("data-analysis");
        let top = p.top_tasks(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].task_type, "coding");
        assert_eq!(top[0].frequency, 3);
    }

    #[test]
    fn test_user_profile_prompt_block() {
        let mut p = UserProfile::new();
        p.set_preference("language", "zh-CN");
        p.expertise_areas.push("Rust".to_string());
        p.record_task("coding");
        let block = p.to_prompt_block();
        assert!(block.contains("<user-profile>"));
        assert!(block.contains("language: zh-CN"));
        assert!(block.contains("Rust"));
        assert!(block.contains("</user-profile>"));
    }

    #[test]
    fn test_user_profile_empty_prompt() {
        let p = UserProfile::new();
        assert!(p.to_prompt_block().is_empty());
    }
}
