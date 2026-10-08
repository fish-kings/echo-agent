//! 护栏（Guard）系统核心 trait 与类型

/// 内容安全护栏（PII 检测与脱敏）
#[cfg(feature = "guard")]
pub mod content;
/// 基于 LLM 的语义护栏
#[cfg(feature = "guard")]
pub mod llm;
/// 基于规则（关键词/正则）的护栏
#[cfg(feature = "guard")]
pub mod rule;

use crate::error::Result;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// 护栏检查方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardDirection {
    /// 用户输入方向检查
    Input,
    /// 模型输出方向检查
    Output,
    /// 工具输入参数检查
    ToolInput,
    /// 工具输出结果检查
    ToolOutput,
}

impl std::fmt::Display for GuardDirection {
    /// 输出方向的 snake_case 标识
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardDirection::Input => write!(f, "input"),
            GuardDirection::Output => write!(f, "output"),
            GuardDirection::ToolInput => write!(f, "tool_input"),
            GuardDirection::ToolOutput => write!(f, "tool_output"),
        }
    }
}

/// 护栏检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardResult {
    /// 检查通过
    Pass,
    /// 内容被拦截
    Block {
        /// 拦截原因
        reason: String,
    },
    /// 从所有护栏收集到的多条警告。
    Warn {
        /// 警告原因列表
        reasons: Vec<String>,
    },
    /// 内容已被安全转换，必须替换原始负载。
    Transform {
        /// 传递给后续护栏和消费方的替换内容。
        content: String,
        /// 描述本次转换的原因。
        reasons: Vec<String>,
    },
}

impl GuardResult {
    /// 判断结果是否为拦截
    pub fn is_blocked(&self) -> bool {
        matches!(self, GuardResult::Block { .. })
    }
}

/// 护栏 trait
pub trait Guard: Send + Sync {
    /// 获取护栏名称
    fn name(&self) -> &str;

    /**
    检查内容

    # 参数
    * `content` - 待检查的内容
    * `direction` - 检查方向
    */
    fn check<'a>(
        &'a self,
        content: &'a str,
        direction: GuardDirection,
    ) -> BoxFuture<'a, Result<GuardResult>>;
}

/// 护栏管理器
pub struct GuardManager {
    /// 按注册顺序排列的护栏链
    guards: Vec<Arc<dyn Guard>>,
}

impl Default for GuardManager {
    /// 创建空的护栏管理器
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GuardManager {
    /// 克隆护栏管理器（共享底层护栏实例）
    fn clone(&self) -> Self {
        Self {
            guards: self.guards.clone(),
        }
    }
}

impl GuardManager {
    /// 创建空的护栏管理器
    pub fn new() -> Self {
        Self { guards: Vec::new() }
    }

    /// 添加一个护栏
    pub fn add(&mut self, guard: Arc<dyn Guard>) {
        self.guards.push(guard);
    }

    /// 从护栏列表创建管理器
    pub fn from_guards(guards: Vec<Arc<dyn Guard>>) -> Self {
        Self { guards }
    }

    /// 判断是否为空（未添加任何护栏）
    pub fn is_empty(&self) -> bool {
        self.guards.is_empty()
    }

    /**
    运行护栏链并返回最终决策（包括转换后的内容）。

    护栏按注册顺序执行，因为执行转换的护栏会改变后续每个护栏需要检查的内容。
    */
    pub async fn check_all(&self, content: &str, direction: GuardDirection) -> Result<GuardResult> {
        if self.guards.is_empty() {
            return Ok(GuardResult::Pass);
        }

        let mut current = content.to_string();
        let mut warnings = Vec::new();
        let mut transformed = false;

        for guard in &self.guards {
            let guard_name = guard.name();
            match guard.check(&current, direction).await {
                Ok(GuardResult::Block { reason }) => {
                    tracing::warn!(
                        guard = guard_name,
                        direction = %direction,
                        reason = %reason,
                        "Guard blocked content"
                    );
                    return Ok(GuardResult::Block { reason });
                }
                Ok(GuardResult::Warn { reasons }) => {
                    warnings.extend(reasons);
                }
                Ok(GuardResult::Transform { content, reasons }) => {
                    current = content;
                    transformed = true;
                    warnings.extend(reasons);
                }
                Ok(GuardResult::Pass) => {}
                Err(e) => {
                    tracing::error!(guard = guard_name, error = %e, "Guard check error");
                    warnings.push(format!("{} error: {}", guard_name, e));
                }
            }
        }

        if transformed {
            Ok(GuardResult::Transform {
                content: current,
                reasons: warnings,
            })
        } else if !warnings.is_empty() {
            Ok(GuardResult::Warn { reasons: warnings })
        } else {
            Ok(GuardResult::Pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 将 "secret" 替换为 "[redacted]" 的测试护栏
    struct ReplaceGuard;

    impl Guard for ReplaceGuard {
        fn name(&self) -> &str {
            "replace"
        }

        fn check<'a>(
            &'a self,
            content: &'a str,
            _direction: GuardDirection,
        ) -> BoxFuture<'a, Result<GuardResult>> {
            Box::pin(async move {
                Ok(GuardResult::Transform {
                    content: content.replace("secret", "[redacted]"),
                    reasons: vec!["redacted secret".to_string()],
                })
            })
        }
    }

    /// 内容中仍含 "secret" 时拦截的测试护栏
    struct RejectUnredactedGuard;

    impl Guard for RejectUnredactedGuard {
        fn name(&self) -> &str {
            "reject-unredacted"
        }

        fn check<'a>(
            &'a self,
            content: &'a str,
            _direction: GuardDirection,
        ) -> BoxFuture<'a, Result<GuardResult>> {
            Box::pin(async move {
                if content.contains("secret") {
                    Ok(GuardResult::Block {
                        reason: "unredacted secret".to_string(),
                    })
                } else {
                    Ok(GuardResult::Pass)
                }
            })
        }
    }

    #[tokio::test]
    async fn transformations_feed_subsequent_guards() -> Result<()> {
        let manager = GuardManager::from_guards(vec![
            Arc::new(ReplaceGuard),
            Arc::new(RejectUnredactedGuard),
        ]);

        let result = manager.check_all("a secret", GuardDirection::Input).await?;
        match result {
            GuardResult::Transform { content, reasons } => {
                assert_eq!(content, "a [redacted]");
                assert_eq!(reasons, vec!["redacted secret"]);
            }
            other => {
                return Err(crate::error::ReactError::Other(format!(
                    "expected transform, got {other:?}"
                )));
            }
        }
        Ok(())
    }
}
