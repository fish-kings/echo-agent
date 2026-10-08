/*!
 * 技能执行遥测——在技能激活期间跟踪工具的可靠性。
 *
 * 提供 `SkillTelemetryStore`，它封装框架的 `Store` trait，
 * 将遥测数据持久化到 `["agent", "skill_telemetry"]` 命名空间下。
 */

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use tracing::warn;

use echo_core::memory::Store;

/// 技能激活期间观察到的一次工具执行。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillExecutionRecord {
    /// 被激活的技能名称。
    pub skill_name: String,
    /// 会话标识符。
    pub session_id: String,
    /// 技能激活时间（epoch 毫秒）。
    pub activated_at: u64,
    /// 技能处于激活状态的时长（毫秒）。
    pub duration_ms: u64,
    /// 本次技能激活期间使用的工具。
    pub tools_used: Vec<String>,
    /// 本次激活期间的工具调用总次数。
    pub tool_calls_count: usize,
    /// 技能执行是否成功。
    pub success: bool,
    /// 执行失败时的错误信息。
    pub error_message: Option<String>,
}

/// 单个技能的聚合遥测指标。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTelemetry {
    /// 技能名称。
    pub skill_name: String,
    /// 该技能激活期间的工具观察总次数。
    pub activation_count: u64,
    /// 成功激活的次数。
    pub success_count: u64,
    /// 失败激活的次数。
    pub failure_count: u64,
    /// 累计总时长（毫秒）。
    pub total_duration_ms: u64,
    /// 与该技能常一起使用的工具及其使用次数。
    pub common_tools: HashMap<String, u64>,
    /// 常见失败模式。
    pub common_failures: Vec<FailurePattern>,
    /// 最近一次使用的时间戳（epoch 毫秒）。
    pub last_used: u64,
    /// 首次使用的时间戳（epoch 毫秒）。
    pub first_used: u64,
    /**
     * 最近的执行结果（最近 20 次，true = 成功，false = 失败）。
     * 用于在健康监控中计算 recent_success_rate。
     */
    #[serde(default)]
    pub recent_records: VecDeque<bool>,
}

impl SkillTelemetry {
    /// 为技能创建新的空遥测数据。
    pub fn new(skill_name: &str) -> Self {
        let now = epoch_millis();
        Self {
            skill_name: skill_name.to_string(),
            activation_count: 0,
            success_count: 0,
            failure_count: 0,
            total_duration_ms: 0,
            common_tools: HashMap::new(),
            common_failures: Vec::new(),
            last_used: now,
            first_used: now,
            recent_records: VecDeque::new(),
        }
    }

    /// 记录一次执行。
    pub fn record(&mut self, record: &SkillExecutionRecord) {
        self.activation_count += 1;
        if record.success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
            // 跟踪失败模式
            if let Some(ref msg) = record.error_message {
                let snippet = truncate_to_200(msg);
                if let Some(pattern) = self
                    .common_failures
                    .iter_mut()
                    .find(|f| f.error_snippet == snippet)
                {
                    pattern.count += 1;
                    pattern.last_occurred = record.activated_at;
                } else {
                    self.common_failures.push(FailurePattern {
                        error_snippet: snippet,
                        count: 1,
                        last_occurred: record.activated_at,
                    });
                }
            }
        }
        self.total_duration_ms += record.duration_ms;
        for tool in &record.tools_used {
            *self.common_tools.entry(tool.clone()).or_insert(0) += 1;
        }
        self.last_used = record.activated_at;
        if record.activated_at < self.first_used {
            self.first_used = record.activated_at;
        }

        // 维护 recent_records FIFO 队列（最近 20 次执行）
        self.recent_records.push_front(record.success);
        if self.recent_records.len() > 20 {
            self.recent_records.pop_back();
        }
    }

    /// 成功率（0.0 到 1.0）。
    pub fn success_rate(&self) -> f64 {
        if self.activation_count == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.activation_count as f64
    }

    /// 平均时长（毫秒）。
    pub fn avg_duration_ms(&self) -> u64 {
        if self.activation_count == 0 {
            return 0;
        }
        self.total_duration_ms / self.activation_count
    }

    /**
     * 基于最近 20 次执行的近期成功率（0.0 到 1.0）。
     * 没有近期记录时返回 0.0。
     */
    pub fn recent_success_rate(&self) -> f64 {
        if self.recent_records.is_empty() {
            return 0.0;
        }
        let successes = self.recent_records.iter().filter(|&&s| s).count();
        successes as f64 / self.recent_records.len() as f64
    }
}

/// 技能执行期间观察到的重复失败模式。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    /// 错误信息的前 200 个字符。
    pub error_snippet: String,
    /// 该失败出现的次数。
    pub count: u64,
    /// 最近一次出现的时间戳（epoch 毫秒）。
    pub last_occurred: u64,
}

/**
 * 技能遥测数据存储，底层由框架的 `Store` trait 支撑。
 *
 * 聚合遥测数据存放在 `["agent", "skill_telemetry"]` 命名空间下，以技能名称为键。
 */
#[derive(Clone)]
pub struct SkillTelemetryStore {
    /// 底层持久化存储
    store: Arc<dyn Store>,
    /// 串行化读-改-写操作的写锁
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

/// 进程级共享的遥测写锁，保证并发记录时不会丢失更新。
static TELEMETRY_WRITE_LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();

/// 获取（必要时初始化）共享的遥测写锁。
fn telemetry_write_lock() -> Arc<tokio::sync::Mutex<()>> {
    TELEMETRY_WRITE_LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

impl SkillTelemetryStore {
    /// 基于给定的 `Store` 创建遥测存储。
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self {
            store,
            write_lock: telemetry_write_lock(),
        }
    }

    /// 记录一次技能执行，并更新聚合遥测数据。
    pub async fn record_execution(
        &self,
        record: &SkillExecutionRecord,
    ) -> echo_core::error::Result<()> {
        let _guard = self.write_lock.lock().await;
        let mut telemetry = self
            .get_telemetry(&record.skill_name)
            .await?
            .unwrap_or_else(|| SkillTelemetry::new(&record.skill_name));

        telemetry.record(record);

        let value = serde_json::to_value(&telemetry).map_err(|e| {
            echo_core::error::ReactError::Other(format!("Failed to serialize telemetry: {}", e))
        })?;

        self.store
            .put(&["agent", "skill_telemetry"], &record.skill_name, value)
            .await
            .map_err(|e| {
                echo_core::error::ReactError::Other(format!("Failed to store telemetry: {}", e))
            })?;

        Ok(())
    }

    /// 获取指定技能的聚合遥测数据。
    pub async fn get_telemetry(
        &self,
        skill_name: &str,
    ) -> echo_core::error::Result<Option<SkillTelemetry>> {
        match self
            .store
            .get(&["agent", "skill_telemetry"], skill_name)
            .await
        {
            Ok(Some(item)) => {
                let telemetry: SkillTelemetry =
                    serde_json::from_value(item.value).map_err(|e| {
                        echo_core::error::ReactError::Other(format!(
                            "Failed to deserialize telemetry for '{}': {}",
                            skill_name, e
                        ))
                    })?;
                Ok(Some(telemetry))
            }
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }

    /// 列出所有技能的遥测数据（按最近使用时间倒序）。
    pub async fn list_all(&self) -> echo_core::error::Result<Vec<SkillTelemetry>> {
        let items = self
            .store
            .list(&["agent", "skill_telemetry"])
            .await
            .unwrap_or_default();

        let mut result = Vec::new();
        for item in items {
            match serde_json::from_value::<SkillTelemetry>(item.value) {
                Ok(t) => result.push(t),
                Err(e) => {
                    warn!("Failed to deserialize telemetry entry: {}", e);
                }
            }
        }

        result.sort_by_key(|telemetry| std::cmp::Reverse(telemetry.last_used));
        Ok(result)
    }
}

/// 获取当前时间的 epoch 毫秒数。
fn epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 将字符串截断为最多 200 个字符。
fn truncate_to_200(s: &str) -> String {
    if s.chars().count() <= 200 {
        s.to_string()
    } else {
        s.chars().take(200).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一条测试用执行记录。
    fn make_record(name: &str, success: bool, duration: u64) -> SkillExecutionRecord {
        SkillExecutionRecord {
            skill_name: name.to_string(),
            session_id: "test-session".to_string(),
            activated_at: epoch_millis(),
            duration_ms: duration,
            tools_used: vec!["bash".into(), "read_file".into()],
            tool_calls_count: 2,
            success,
            error_message: if success {
                None
            } else {
                Some("timeout error".into())
            },
        }
    }

    /// 新建遥测数据应为空。
    #[test]
    fn test_telemetry_new() {
        let t = SkillTelemetry::new("test-skill");
        assert_eq!(t.skill_name, "test-skill");
        assert_eq!(t.activation_count, 0);
        assert_eq!(t.success_rate(), 0.0);
    }

    /// 记录成功执行应更新成功计数。
    #[test]
    fn test_telemetry_record_success() {
        let mut t = SkillTelemetry::new("coding");
        let r = make_record("coding", true, 5000);
        t.record(&r);
        assert_eq!(t.activation_count, 1);
        assert_eq!(t.success_count, 1);
        assert_eq!(t.failure_count, 0);
        assert!((t.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    /// 记录失败执行应更新失败计数与失败模式。
    #[test]
    fn test_telemetry_record_failure() {
        let mut t = SkillTelemetry::new("coding");
        let r = make_record("coding", false, 3000);
        t.record(&r);
        assert_eq!(t.activation_count, 1);
        assert_eq!(t.success_count, 0);
        assert_eq!(t.failure_count, 1);
        assert_eq!(t.common_failures.len(), 1);
        assert_eq!(t.common_failures[0].error_snippet, "timeout error");
    }

    /// 失败片段截断应对 UTF-8 安全。
    #[test]
    fn failure_snippet_truncation_is_utf8_safe() {
        let input = "错".repeat(250);
        let truncated = truncate_to_200(&input);
        assert_eq!(truncated.chars().count(), 200);
    }

    /// 多次记录应正确聚合时长与工具使用。
    #[test]
    fn test_telemetry_aggregate() {
        let mut t = SkillTelemetry::new("coding");
        t.record(&make_record("coding", true, 5000));
        t.record(&make_record("coding", true, 3000));
        t.record(&make_record("coding", false, 1000));

        assert_eq!(t.activation_count, 3);
        assert_eq!(t.success_count, 2);
        assert_eq!(t.failure_count, 1);
        assert_eq!(t.total_duration_ms, 9000);
        assert_eq!(t.avg_duration_ms(), 3000);
        // bash 使用 3 次，read_file 使用 3 次
        assert_eq!(t.common_tools.get("bash"), Some(&3));
        assert_eq!(t.common_tools.get("read_file"), Some(&3));
    }

    /// 相同错误应合并为同一失败模式。
    #[test]
    fn test_failure_pattern_dedup() {
        let mut t = SkillTelemetry::new("coding");
        t.record(&make_record("coding", false, 1000));
        t.record(&make_record("coding", false, 2000));
        assert_eq!(t.common_failures.len(), 1);
        assert_eq!(t.common_failures[0].count, 2);
    }

    /// 验证 200 字符截断。
    #[test]
    fn test_truncate_to_200() {
        let short = "hello";
        assert_eq!(truncate_to_200(short), "hello");

        let long = "a".repeat(300);
        assert_eq!(truncate_to_200(&long).len(), 200);
    }

    /// 近期记录应只保留最近 20 次。
    #[test]
    fn test_recent_records_fifo() {
        let mut t = SkillTelemetry::new("coding");

        // 记录 25 次执行（超过 20 次的 FIFO 上限）
        for i in 0..25 {
            let r = make_record("coding", i % 2 == 0, 1000);
            t.record(&r);
        }

        // 应只保留最近 20 次
        assert_eq!(t.recent_records.len(), 20);
        assert_eq!(t.activation_count, 25);
    }

    /// 无近期记录时近期成功率为 0。
    #[test]
    fn test_recent_success_rate_empty() {
        let t = SkillTelemetry::new("coding");
        assert_eq!(t.recent_success_rate(), 0.0);
    }

    /// 全部成功时近期成功率为 1。
    #[test]
    fn test_recent_success_rate_all_success() {
        let mut t = SkillTelemetry::new("coding");
        for _ in 0..5 {
            t.record(&make_record("coding", true, 1000));
        }
        assert!((t.recent_success_rate() - 1.0).abs() < f64::EPSILON);
    }

    /// 成功与失败混合时近期成功率正确。
    #[test]
    fn test_recent_success_rate_mixed() {
        let mut t = SkillTelemetry::new("coding");
        // 3 次成功，2 次失败
        t.record(&make_record("coding", true, 1000));
        t.record(&make_record("coding", true, 1000));
        t.record(&make_record("coding", false, 1000));
        t.record(&make_record("coding", true, 1000));
        t.record(&make_record("coding", false, 1000));
        assert!((t.recent_success_rate() - 0.6).abs() < f64::EPSILON);
    }
}
