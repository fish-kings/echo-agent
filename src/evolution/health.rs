/*!
 * Skill 健康监控与评分。
 *
 * 通过分析遥测数据（包括成功率、使用模式和失败模式）来监控
 * 活跃 Skill 的健康状况。提供可用于驱动 Skill 生命周期决策
 * （例如废弃）的健康评分。
 */

use chrono::{DateTime, Utc};
use std::sync::Arc;

use echo_core::memory::store::Store;
use echo_state::skill_telemetry::{SkillTelemetry, SkillTelemetryStore};
use serde::{Deserialize, Serialize};

use super::audit::{ChangeEntryBuilder, ChangeLog, ChangeType, EntityType};
use super::layer::EvolutionObserver;
use crate::error::Result;

// 重导出 SkillDescriptor 以便在本模块中使用。
pub use echo_execution::skills::external::SkillDescriptor;

/// Skill 的健康状态分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// 健康（评分 >= 0.7）。
    Healthy,
    /// 需要关注（评分 0.4-0.7）。
    NeedsAttention,
    /// 不健康（评分 < 0.4）。
    Unhealthy,
}

impl HealthStatus {
    /// 将健康评分（0.0-1.0）转换为健康状态。
    pub fn from_score(score: f64) -> Self {
        if score >= 0.7 {
            Self::Healthy
        } else if score >= 0.4 {
            Self::NeedsAttention
        } else {
            Self::Unhealthy
        }
    }

    /// 返回健康状态的可读描述。
    pub fn description(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::NeedsAttention => "needs attention",
            Self::Unhealthy => "unhealthy",
        }
    }
}

/// 健康评分各组成部分的详细拆解。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthBreakdown {
    /// 总体成功率（0.0-1.0）。
    pub success_rate: f64,
    /// 最近 20 次执行的成功率（0.0-1.0）。
    pub recent_success_rate: f64,
    /// 归一化到 0.0-1.0 范围的使用频率。
    pub usage_frequency: f64,
    /// 基于距上次使用天数计算的新鲜度（0.0-1.0，1.0 表示非常近期）。
    pub freshness: f64,
    /// 根据低失败率推断出的用户认可度（0.0-1.0）。
    pub user_approval: f64,
    /// 命令有效性 —— 不存在严重失败（0.0-1.0）。
    pub command_validity: f64,
}

impl HealthBreakdown {
    /**
     * 使用加权公式计算总体健康评分：
     * health = success_rate * 0.3 + recent_success_rate * 0.2 + usage_frequency * 0.1
     *        + freshness * 0.15 + user_approval * 0.15 + command_validity * 0.1
     */
    pub fn overall_score(&self) -> f64 {
        self.success_rate * 0.3
            + self.recent_success_rate * 0.2
            + self.usage_frequency * 0.1
            + self.freshness * 0.15
            + self.user_approval * 0.15
            + self.command_validity * 0.1
    }
}

/// 单个 Skill 的健康报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillHealthReport {
    /// Skill 名称。
    pub skill_name: String,
    /// 总体健康评分（0.0-1.0）。
    pub health_score: f64,
    /// 健康状态分类。
    pub status: HealthStatus,
    /// 评分各组成部分的详细拆解。
    pub breakdown: HealthBreakdown,
    /// 根据健康分析得出的可读建议。
    pub recommendations: Vec<String>,
    /// 本报告的生成时间。
    #[serde(with = "crate::utils::time::local_rfc3339")]
    pub analyzed_at: DateTime<Utc>,
}

/// 通过分析遥测数据来监控 Skill 健康状况。
pub struct SkillHealthMonitor {
    /// 遥测数据存储
    telemetry_store: SkillTelemetryStore,
    /// 可选的 evolution 事件观察者
    observer: Option<Arc<dyn EvolutionObserver>>,
}

impl SkillHealthMonitor {
    /// 创建一个新的健康监控器。
    pub fn new(store: Arc<dyn Store>) -> Self {
        let telemetry_store = SkillTelemetryStore::new(store);
        Self {
            telemetry_store,
            observer: None,
        }
    }

    /// 将完成的健康报告发布给 evolution 事件观察者。
    pub fn with_evolution_observer(mut self, observer: Arc<dyn EvolutionObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// 分析单个 Skill 的健康状况。
    pub async fn analyze_skill(&self, skill_name: &str) -> Result<Option<SkillHealthReport>> {
        let telemetry = match self.telemetry_store.get_telemetry(skill_name).await? {
            Some(t) => t,
            None => return Ok(None),
        };

        let breakdown = self.compute_breakdown(&telemetry);
        let health_score = breakdown.overall_score();
        let status = HealthStatus::from_score(health_score);
        let recommendations = self.generate_recommendations(&telemetry, &breakdown, status);

        let report = SkillHealthReport {
            skill_name: skill_name.to_string(),
            health_score,
            status,
            breakdown,
            recommendations,
            analyzed_at: Utc::now(),
        };
        if let Some(observer) = &self.observer {
            observer.on_skill_health_check(skill_name).await;
        }
        Ok(Some(report))
    }

    /// 分析所有 Skill 的健康状况并返回汇总结果。
    pub async fn analyze_all_skills(&self) -> Result<Vec<SkillHealthReport>> {
        let all_telemetry = self.telemetry_store.list_all().await?;
        let mut reports = Vec::new();

        for telemetry in all_telemetry {
            let breakdown = self.compute_breakdown(&telemetry);
            let health_score = breakdown.overall_score();
            let status = HealthStatus::from_score(health_score);
            let recommendations = self.generate_recommendations(&telemetry, &breakdown, status);

            let report = SkillHealthReport {
                skill_name: telemetry.skill_name.clone(),
                health_score,
                status,
                breakdown,
                recommendations,
                analyzed_at: Utc::now(),
            };
            if let Some(observer) = &self.observer {
                observer.on_skill_health_check(&report.skill_name).await;
            }
            reports.push(report);
        }

        Ok(reports)
    }

    /// 将一次健康检查记录到变更日志中。
    pub fn record_health_check(
        &self,
        report: &SkillHealthReport,
        change_log: &dyn ChangeLog,
    ) -> Result<()> {
        let entry =
            ChangeEntryBuilder::new(EntityType::Skill, &report.skill_name, ChangeType::Update)
                .reason(format!(
                    "Health check: {} (score: {:.2})",
                    report.status.description(),
                    report.health_score
                ))
                .trigger("skill_health_monitor".to_string())
                .build(change_log);
        change_log.record(entry)?;
        Ok(())
    }

    /// 根据遥测数据计算健康评分拆解。
    fn compute_breakdown(&self, telemetry: &SkillTelemetry) -> HealthBreakdown {
        let success_rate = if telemetry.activation_count > 0 {
            telemetry.success_count as f64 / telemetry.activation_count as f64
        } else {
            0.0
        };

        let recent_success_rate = telemetry.recent_success_rate();

        // 归一化使用频率：假设最多 1000 次激活对应 1.0
        let usage_frequency = (telemetry.activation_count as f64 / 1000.0).min(1.0);

        // 新鲜度：距上次使用的天数，上限为 90 天
        let now = Utc::now().timestamp_millis() as u64;
        let days_since_last_use = if telemetry.last_used > 0 {
            (now.saturating_sub(telemetry.last_used)) / (24 * 60 * 60 * 1000)
        } else {
            90
        };
        let freshness = 1.0 - (days_since_last_use as f64 / 90.0).min(1.0);

        // 用户认可度：失败率的反面
        let user_approval = if telemetry.activation_count > 0 {
            1.0 - (telemetry.failure_count as f64 / telemetry.activation_count as f64)
        } else {
            0.0
        };

        /* 命令有效性：不存在严重失败
         * 若存在包含 "critical" 或 "fatal" 关键词的失败，则降低评分
         */
        let has_critical_failures = telemetry.common_failures.iter().any(|f| {
            let snippet_lower = f.error_snippet.to_lowercase();
            snippet_lower.contains("critical") || snippet_lower.contains("fatal")
        });
        let command_validity = if has_critical_failures { 0.3 } else { 1.0 };

        HealthBreakdown {
            success_rate,
            recent_success_rate,
            usage_frequency,
            freshness,
            user_approval,
            command_validity,
        }
    }

    /// 根据健康分析生成建议。
    fn generate_recommendations(
        &self,
        telemetry: &SkillTelemetry,
        breakdown: &HealthBreakdown,
        status: HealthStatus,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match status {
            HealthStatus::Healthy => {
                // 健康的 Skill 不需要任何建议
            }
            HealthStatus::NeedsAttention => {
                if breakdown.success_rate < 0.7 {
                    recommendations.push(format!(
                        "Success rate is {:.1}%. Review common failures and consider patching.",
                        breakdown.success_rate * 100.0
                    ));
                }
                if breakdown.freshness < 0.5 {
                    recommendations.push(
                        "Skill has not been used recently. Consider if it's still relevant."
                            .to_string(),
                    );
                }
                if breakdown.user_approval < 0.7 {
                    recommendations.push(
                        "High failure rate suggests user dissatisfaction. Review error patterns."
                            .to_string(),
                    );
                }
            }
            HealthStatus::Unhealthy => {
                if breakdown.success_rate < 0.5 {
                    recommendations.push(format!(
                        "Critical: Success rate is only {:.1}%. Skill may need major revision or deprecation.",
                        breakdown.success_rate * 100.0
                    ));
                }
                if breakdown.recent_success_rate < 0.5 {
                    recommendations.push(format!(
                        "Recent success rate is {:.1}%. Skill is performing poorly in recent usage.",
                        breakdown.recent_success_rate * 100.0
                    ));
                }
                if !telemetry.common_failures.is_empty() {
                    let top_failure = &telemetry.common_failures[0];
                    recommendations.push(format!(
                        "Most common failure: '{}' (occurred {} times). Consider adding error handling.",
                        top_failure.error_snippet, top_failure.count
                    ));
                }
                if breakdown.freshness < 0.3 {
                    recommendations.push(
                        "Skill is rarely used and may be obsolete. Consider archiving.".to_string(),
                    );
                }
                recommendations.push(
                    "Consider deprecating this skill or merging with a similar skill.".to_string(),
                );
            }
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_state::memory::store::InMemoryStore;
    use echo_state::skill_telemetry::SkillExecutionRecord;
    use futures::future::BoxFuture;
    use std::sync::Mutex;

    struct HealthObserver {
        names: Arc<Mutex<Vec<String>>>,
    }

    impl EvolutionObserver for HealthObserver {
        fn on_skill_health_check<'a>(&'a self, skill_name: &'a str) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                self.names
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(skill_name.to_string());
            })
        }
    }

    fn make_record(skill_name: &str, success: bool, error: Option<&str>) -> SkillExecutionRecord {
        SkillExecutionRecord {
            skill_name: skill_name.to_string(),
            session_id: "test-session".to_string(),
            activated_at: Utc::now().timestamp_millis() as u64,
            duration_ms: 1000,
            tools_used: vec!["Bash".to_string()],
            tool_calls_count: 1,
            success,
            error_message: error.map(|s| s.to_string()),
        }
    }

    #[tokio::test]
    async fn test_healthy_skill() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let observed_names = Arc::new(Mutex::new(Vec::new()));
        let monitor = SkillHealthMonitor::new(store.clone()).with_evolution_observer(Arc::new(
            HealthObserver {
                names: observed_names.clone(),
            },
        ));

        // 记录 10 次成功执行
        for _ in 0..10 {
            let telemetry_store = SkillTelemetryStore::new(store.clone());
            telemetry_store
                .record_execution(&make_record("test-skill", true, None))
                .await
                .unwrap();
        }

        let report = monitor.analyze_skill("test-skill").await.unwrap().unwrap();
        assert_eq!(report.status, HealthStatus::Healthy);
        assert!(report.health_score >= 0.7);
        assert!(report.recommendations.is_empty());
        assert_eq!(
            *observed_names
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec!["test-skill".to_string()]
        );
    }

    #[tokio::test]
    async fn test_unhealthy_skill() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let monitor = SkillHealthMonitor::new(store.clone());

        // 记录 10 次执行中的 8 次失败
        let telemetry_store = SkillTelemetryStore::new(store.clone());
        for _ in 0..2 {
            telemetry_store
                .record_execution(&make_record("test-skill", true, None))
                .await
                .unwrap();
        }
        for _ in 0..8 {
            telemetry_store
                .record_execution(&make_record("test-skill", false, Some("permission denied")))
                .await
                .unwrap();
        }

        let report = monitor.analyze_skill("test-skill").await.unwrap().unwrap();
        assert_eq!(report.status, HealthStatus::Unhealthy);
        assert!(report.health_score < 0.4);
        assert!(!report.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_needs_attention_skill() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let monitor = SkillHealthMonitor::new(store.clone());

        // 记录 5 次成功与 5 次失败（成功率 50%）
        let telemetry_store = SkillTelemetryStore::new(store.clone());
        for _ in 0..5 {
            telemetry_store
                .record_execution(&make_record("test-skill", true, None))
                .await
                .unwrap();
        }
        for _ in 0..5 {
            telemetry_store
                .record_execution(&make_record("test-skill", false, Some("timeout")))
                .await
                .unwrap();
        }

        let report = monitor.analyze_skill("test-skill").await.unwrap().unwrap();
        assert_eq!(report.status, HealthStatus::NeedsAttention);
        assert!(report.health_score >= 0.4 && report.health_score < 0.7);
    }

    #[test]
    fn test_health_status_from_score() {
        assert_eq!(HealthStatus::from_score(0.8), HealthStatus::Healthy);
        assert_eq!(HealthStatus::from_score(0.7), HealthStatus::Healthy);
        assert_eq!(HealthStatus::from_score(0.5), HealthStatus::NeedsAttention);
        assert_eq!(HealthStatus::from_score(0.4), HealthStatus::NeedsAttention);
        assert_eq!(HealthStatus::from_score(0.3), HealthStatus::Unhealthy);
        assert_eq!(HealthStatus::from_score(0.0), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_breakdown_calculation() {
        let breakdown = HealthBreakdown {
            success_rate: 0.9,
            recent_success_rate: 0.95,
            usage_frequency: 0.5,
            freshness: 0.8,
            user_approval: 0.9,
            command_validity: 1.0,
        };

        let score = breakdown.overall_score();
        /* 0.9*0.3 + 0.95*0.2 + 0.5*0.1 + 0.8*0.15 + 0.9*0.15 + 1.0*0.1
         * = 0.27 + 0.19 + 0.05 + 0.12 + 0.135 + 0.1 = 0.865
         */
        assert!((score - 0.865).abs() < 0.01);
    }

    #[test]
    fn test_critical_failure_reduces_validity() {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let monitor = SkillHealthMonitor::new(store.clone());

        let mut telemetry = SkillTelemetry::new("test-skill");
        telemetry.activation_count = 10;
        telemetry.success_count = 9;
        telemetry.failure_count = 1;
        telemetry
            .common_failures
            .push(echo_state::skill_telemetry::FailurePattern {
                error_snippet: "CRITICAL: system failure".to_string(),
                count: 1,
                last_occurred: Utc::now().timestamp_millis() as u64,
            });

        let breakdown = monitor.compute_breakdown(&telemetry);
        assert_eq!(breakdown.command_validity, 0.3);
    }
}
