/*!
工具权限模型

提供一套多层级的权限控制系统：
- PermissionMode：权限模式（default/plan/auto/bypass 等）
- PermissionRule：规则系统（allow/deny/ask）
- RuleSource：规则来源优先级
- RuleRegistry：规则注册表

参考自 Claude Code 的权限架构设计
*/

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ── 工具权限类型 ──────────────────────────────────────────────────────────────

/// 工具权限类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolPermission {
    /// 读取文件/目录权限
    Read,
    /// 写入文件/目录权限
    Write,
    /// 网络访问权限
    Network,
    /// 执行命令/代码权限
    Execute,
    /// 敏感操作权限（例如访问密钥、环境变量等）
    Sensitive,
}

impl std::fmt::Display for ToolPermission {
    /// 将权限类型格式化为小写的字符串标识（如 `"read"`、`"write"`）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolPermission::Read => write!(f, "read"),
            ToolPermission::Write => write!(f, "write"),
            ToolPermission::Network => write!(f, "network"),
            ToolPermission::Execute => write!(f, "execute"),
            ToolPermission::Sensitive => write!(f, "sensitive"),
        }
    }
}

// ── 权限模式 ──────────────────────────────────────────────────────────────────

/**
权限模式 - 控制权限检查行为

参考自 Claude Code 的 PermissionMode 设计：
- Default：危险操作需要用户确认
- Plan：内部只读执行模式（不是面向用户的审批模式）
- AcceptEdits：自动接受编辑操作
- BypassPermissions：绕过所有检查（可通过 bypass_disabled 禁用）
- Auto：由 AI 分类器自动决策
- Bubble：子 Agent 的权限向上冒泡给父进程
- DontAsk：静默拒绝未命中允许规则的工具（不向用户提示）
- StrictConfirm：写入/执行/网络/敏感操作前均需询问
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    /// 默认模式：危险操作需要用户确认
    #[default]
    Default,
    /**
    内部只读执行模式：不允许写入和执行操作。
    面向用户的规划/任务路由场景应改用交互模式。
    */
    Plan,
    /// 自动接受文件编辑操作
    #[serde(
        rename = "auto-edit",
        alias = "accept-edits",
        alias = "autoedit",
        alias = "acceptedits"
    )]
    AcceptEdits,
    /// 绕过所有权限检查（请谨慎使用）
    #[serde(
        rename = "full-auto",
        alias = "fullauto",
        alias = "bypass",
        alias = "bypass-permissions",
        alias = "bypasspermissions"
    )]
    BypassPermissions,
    /// 由 AI 分类器自动决策（需要实现 Classifier）
    Auto,
    /// 子 Agent 的权限向上冒泡给父进程
    Bubble,
    /**
    静默模式：匹配到的规则自动允许，未匹配的静默拒绝

    介于 Default 与 BypassPermissions 之间的中间模式，
    适用于 CI/CD 等无人值守的执行场景。
    */
    #[serde(rename = "dont-ask", alias = "dontask")]
    DontAsk,
    /// 严格交互模式：读取操作被允许，变更或外部操作均需询问。
    #[serde(
        rename = "strict",
        alias = "strict-confirm",
        alias = "strict-confirmation"
    )]
    StrictConfirm,
}

impl PermissionMode {
    /// 供配置和 UI 传输层使用的稳定 kebab-case 标识符。
    pub const fn id(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::AcceptEdits => "auto-edit",
            Self::BypassPermissions => "full-auto",
            Self::Auto => "auto",
            Self::Bubble => "bubble",
            Self::DontAsk => "dont-ask",
            Self::StrictConfirm => "strict",
        }
    }

    /// 检查当前模式是否允许写入操作
    pub fn allows_write(&self) -> bool {
        match self {
            PermissionMode::BypassPermissions => true,
            PermissionMode::AcceptEdits => true,
            /*
            DontAsk 依赖规则匹配来决定：只有命中显式允许规则的写入才被允许；
            否则静默拒绝。
            */
            PermissionMode::DontAsk => false,
            PermissionMode::Plan => false,
            PermissionMode::StrictConfirm => false,
            _ => false,
        }
    }

    /// 检查当前模式是否需要用户交互
    pub fn requires_interaction(&self) -> bool {
        match self {
            PermissionMode::BypassPermissions => false,
            PermissionMode::Auto => false,
            PermissionMode::DontAsk => false, // 静默拒绝，无需交互
            PermissionMode::AcceptEdits => false, // 编辑自动接受，其它操作仍需确认
            PermissionMode::StrictConfirm => true,
            _ => true,
        }
    }

    /// 检查当前模式是否使用分类器
    pub fn uses_classifier(&self) -> bool {
        matches!(self, PermissionMode::Auto)
    }
}

impl std::fmt::Display for PermissionMode {
    /// 输出该模式的稳定 kebab-case 标识符。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

impl std::str::FromStr for PermissionMode {
    type Err = String;

    /// 解析字符串标识符（含历史别名）为对应的权限模式。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" | "ask" => Ok(Self::Default),
            "plan" => Ok(Self::Plan),
            "auto-edit" | "autoedit" | "accept-edits" | "acceptedits" => Ok(Self::AcceptEdits),
            "full-auto" | "fullauto" | "bypass" | "bypass-permissions" | "bypasspermissions" => {
                Ok(Self::BypassPermissions)
            }
            "auto" => Ok(Self::Auto),
            "bubble" => Ok(Self::Bubble),
            "dont-ask" | "dontask" => Ok(Self::DontAsk),
            "strict" | "strict-confirm" | "strict-confirmation" => Ok(Self::StrictConfirm),
            other => Err(format!(
                "invalid permission mode '{other}'; expected default, plan, auto-edit, full-auto, auto, bubble, dont-ask, or strict"
            )),
        }
    }
}

// ── 权限决策 ──────────────────────────────────────────────────────────────────

/// 权限决策
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    /// 允许执行
    Allow,
    /// 拒绝执行
    Deny {
        /// 拒绝原因
        reason: String,
    },
    /// 需要用户审批
    RequireApproval,
    /// 需要用户审批，并附带建议
    Ask {
        /// 建议列表
        suggestions: Vec<String>,
    },
}

impl PermissionDecision {
    /// 检查是否为允许决策
    pub fn is_allowed(&self) -> bool {
        matches!(self, PermissionDecision::Allow)
    }

    /// 检查是否为拒绝决策
    pub fn is_denied(&self) -> bool {
        matches!(self, PermissionDecision::Deny { .. })
    }

    /// 检查是否需要用户审批
    pub fn requires_approval(&self) -> bool {
        matches!(
            self,
            PermissionDecision::RequireApproval | PermissionDecision::Ask { .. }
        )
    }
}

// ── 规则来源优先级 ─────────────────────────────────────────────────────────────

/**
规则来源优先级（数值越大优先级越高）

参考自 Claude Code 的 PERMISSION_RULE_SOURCES 设计。
在拒绝优先（deny-first）的评估策略下，来源优先级仅影响同类型规则
（deny/ask/allow）之间的排序；deny 规则始终优先于 ask 和 allow 规则。
*/
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum RuleSource {
    /// 默认规则（最低优先级）
    #[default]
    Default = 0,
    /// 本地设置（.echo/settings.local.json）
    LocalSettings = 1,
    /// 项目设置（.echo/settings.json）
    ProjectSettings = 2,
    /// 用户设置（~/.echo/settings.json）
    UserSettings = 3,
    /// 管理员策略（用户不可覆盖，用于企业部署）
    Managed = 4,
    /// CLI 命令行参数
    CliArg = 5,
    /// 会话级临时规则（最高优先级）
    Session = 6,
}

impl std::str::FromStr for RuleSource {
    type Err = String;

    /// 将 camelCase / snake_case 字符串标识符解析为对应的规则来源。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default" => Ok(Self::Default),
            "localSettings" | "local_settings" => Ok(Self::LocalSettings),
            "projectSettings" | "project_settings" => Ok(Self::ProjectSettings),
            "userSettings" | "user_settings" | "manual" => Ok(Self::UserSettings),
            "managed" => Ok(Self::Managed),
            "cliArg" | "cli_arg" => Ok(Self::CliArg),
            "session" => Ok(Self::Session),
            _ => Err(format!("unknown permission rule source: {value}")),
        }
    }
}

impl std::fmt::Display for RuleSource {
    /// 输出该来源对应的 camelCase 字符串标识符。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleSource::Default => write!(f, "default"),
            RuleSource::LocalSettings => write!(f, "localSettings"),
            RuleSource::ProjectSettings => write!(f, "projectSettings"),
            RuleSource::UserSettings => write!(f, "userSettings"),
            RuleSource::Managed => write!(f, "managed"),
            RuleSource::CliArg => write!(f, "cliArg"),
            RuleSource::Session => write!(f, "session"),
        }
    }
}

// ── 规则匹配器 ────────────────────────────────────────────────────────────────

/// 规则匹配器 - 定义一条规则如何匹配工具调用
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RuleMatcher {
    /// 精确匹配工具名称
    Tool {
        /// 待匹配的工具名称
        name: String,
    },
    /// 通配符模式匹配（支持 "Bash(git:*)" 等形式）
    Pattern {
        /// 通配符匹配模式
        pattern: String,
    },
    /// 按权限类型匹配
    Permission {
        /// 待匹配的权限类型
        permission: ToolPermission,
    },
    /// 匹配所有工具
    All,
}

impl std::str::FromStr for RuleMatcher {
    type Err = String;

    /**
    将字符串形式的匹配器规格（如 `"tool:xxx"`、`"pattern:xxx"`、`"perm:xxx"`、
    `"*"`）解析为对应的匹配器。
    */
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "*" || value == "all" {
            return Ok(Self::All);
        }
        if let Some(name) = value.strip_prefix("tool:") {
            if name.is_empty() {
                return Err("tool permission matcher requires a name".to_string());
            }
            return Ok(Self::Tool {
                name: name.to_string(),
            });
        }
        if let Some(pattern) = value.strip_prefix("pattern:") {
            if pattern.is_empty() {
                return Err("pattern permission matcher cannot be empty".to_string());
            }
            return Ok(Self::Pattern {
                pattern: pattern.to_string(),
            });
        }
        let flag = value
            .strip_prefix("perm:")
            .or_else(|| value.strip_prefix("permission:"));
        if let Some(flag) = flag {
            let permission = match flag {
                "read" => ToolPermission::Read,
                "write" => ToolPermission::Write,
                "network" => ToolPermission::Network,
                "execute" => ToolPermission::Execute,
                "sensitive" => ToolPermission::Sensitive,
                _ => return Err(format!("unknown permission matcher: {flag}")),
            };
            return Ok(Self::Permission { permission });
        }
        Err(format!("unsupported permission matcher: {value}"))
    }
}

impl RuleMatcher {
    /** 检查一个匹配器字符串是否与当前匹配器一致（用于删除规则时定位）。

    与 `parse_rule()` 中 `RuleMatcher::Pattern` 的构造语义保持一致：
    删除规则时使用的匹配器字符串与新增规则时使用的完全相同，以便据此定位规则。
    */
    pub fn matches_matcher_str(&self, matcher_str: &str) -> bool {
        match self {
            RuleMatcher::Tool { name } => name == matcher_str,
            RuleMatcher::Pattern { pattern } => pattern == matcher_str,
            RuleMatcher::Permission { .. } => false,
            RuleMatcher::All => matcher_str == "*" || matcher_str == "all",
        }
    }

    /// 检查是否与指定工具匹配
    pub fn matches(&self, tool_name: &str, permissions: &[ToolPermission]) -> bool {
        match self {
            RuleMatcher::Tool { name } => tool_name == name,
            RuleMatcher::Pattern { pattern } => {
                if pattern == "*" {
                    return true;
                }
                // 优先精确匹配
                if tool_name == pattern {
                    return true;
                }
                /* 对形如 "Bash(rm:*)" 的模式使用 glob 匹配；
                只有在 glob 匹配成功时才返回 true，未命中时继续走前缀检查。 */
                #[cfg(feature = "permission")]
                {
                    if let Ok(glob) = globset::Glob::new(pattern) {
                        let matcher = glob.compile_matcher();
                        if matcher.is_match(tool_name) {
                            return true;
                        }
                    }
                }
                /* 兜底方案：在没有 globset 的情况下处理 "prefix*)" 形式的模式。
                例如 "Bash(rm:*)" 匹配 "Bash(rm:rf)"。 */
                if pattern.ends_with("*)") {
                    let prefix = &pattern[..pattern.len() - 2];
                    if tool_name.starts_with(prefix) {
                        return true;
                    }
                }
                // 兜底方案："Bash" 前缀匹配 "Bash(git:*)"
                if tool_name.starts_with(pattern)
                    && tool_name.len() > pattern.len()
                    && tool_name.as_bytes()[pattern.len()] == b'('
                {
                    return true;
                }
                false
            }
            RuleMatcher::Permission { permission } => permissions.contains(permission),
            RuleMatcher::All => true,
        }
    }
}

impl std::fmt::Display for RuleMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleMatcher::Tool { name } => write!(f, "tool:{}", name),
            RuleMatcher::Pattern { pattern } => write!(f, "pattern:{}", pattern),
            RuleMatcher::Permission { permission } => write!(f, "permission:{}", permission),
            RuleMatcher::All => write!(f, "all"),
        }
    }
}

// ── 规则行为 ──────────────────────────────────────────────────────────────────

/// 规则行为 - 匹配成功后采取的动作
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RuleBehavior {
    /// 允许执行
    Allow,
    /// 拒绝执行
    Deny {
        /// 拒绝原因
        reason: String,
    },
    /// 需要用户确认
    Ask {
        /// 建议列表
        suggestions: Vec<String>,
    },
}

impl std::str::FromStr for RuleBehavior {
    type Err = String;

    /// 将字符串标识符（`"allow"`、`"deny"`、`"ask"`）解析为对应的规则行为。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "allow" => Ok(Self::Allow),
            "deny" => Ok(Self::Deny {
                reason: "denied by rule".to_string(),
            }),
            "ask" => Ok(Self::Ask {
                suggestions: vec!["allow".to_string(), "deny".to_string()],
            }),
            _ => Err(format!("unknown permission rule behavior: {value}")),
        }
    }
}

impl RuleBehavior {
    /// 转换为 PermissionDecision
    pub fn to_decision(&self) -> PermissionDecision {
        match self {
            RuleBehavior::Allow => PermissionDecision::Allow,
            RuleBehavior::Deny { reason } => PermissionDecision::Deny {
                reason: reason.clone(),
            },
            RuleBehavior::Ask { suggestions } => PermissionDecision::Ask {
                suggestions: suggestions.clone(),
            },
        }
    }
}

// ── 权限规则 ──────────────────────────────────────────────────────────────────

/// 权限规则 - 单条规则的完整定义
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionRule {
    /// 规则匹配器
    pub matcher: RuleMatcher,
    /// 规则行为
    pub behavior: RuleBehavior,
    /// 规则来源
    pub source: RuleSource,
    /// 规则描述（可选）
    #[serde(default)]
    pub description: Option<String>,
}

impl PermissionRule {
    /// 创建一条允许规则
    pub fn allow(matcher: RuleMatcher, source: RuleSource) -> Self {
        Self {
            matcher,
            behavior: RuleBehavior::Allow,
            source,
            description: None,
        }
    }

    /// 创建一条拒绝规则
    pub fn deny(matcher: RuleMatcher, reason: String, source: RuleSource) -> Self {
        Self {
            matcher,
            behavior: RuleBehavior::Deny { reason },
            source,
            description: None,
        }
    }

    /// 创建一条询问规则
    pub fn ask(matcher: RuleMatcher, suggestions: Vec<String>, source: RuleSource) -> Self {
        Self {
            matcher,
            behavior: RuleBehavior::Ask { suggestions },
            source,
            description: None,
        }
    }

    /// 检查是否与指定的工具调用匹配
    pub fn matches(&self, tool_name: &str, permissions: &[ToolPermission]) -> bool {
        self.matcher.matches(tool_name, permissions)
    }
}

// ── 规则注册表 ────────────────────────────────────────────────────────────────

/** 规则注册表 - 管理所有权限规则

规则按来源优先级排序；优先级更高的规则优先匹配。匹配顺序：
1. 按来源优先级从高到低
2. 同一来源内，按插入顺序
*/
#[derive(Debug, Clone, Default)]
pub struct RuleRegistry {
    /// 已注册的规则列表，按来源优先级排序
    rules: Vec<PermissionRule>,
}

impl RuleRegistry {
    /// 创建一个空的规则注册表
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一条规则（自动按优先级排序）
    pub fn add_rule(&mut self, rule: PermissionRule) {
        // 按来源优先级插入排序
        let pos = self
            .rules
            .iter()
            .position(|r| r.source < rule.source)
            .unwrap_or(self.rules.len());

        self.rules.insert(pos, rule);
    }

    /// 批量添加规则
    pub fn add_rules(&mut self, rules: Vec<PermissionRule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// 精确移除一条规则，不影响其它来源所拥有的规则。
    pub fn remove_rule(&mut self, rule: &PermissionRule) -> bool {
        let before = self.rules.len();
        self.rules.retain(|candidate| candidate != rule);
        before != self.rules.len()
    }

    /** 检查一次工具调用，返回匹配到的规则行为

    评估顺序遵循拒绝优先（deny-first）原则（参考自 Claude Code）：
    1. 拒绝规则 —— 任意来源的拒绝都优先于所有允许
    2. 询问规则 —— 按来源优先级
    3. 允许规则 —— 按来源优先级

    这保证了一条低优先级的拒绝规则永远不会被高优先级的允许规则覆盖。

    实现为单次遍历：命中拒绝立即返回；询问与允许则先记录下来，
    循环结束后返回其中优先级最高的匹配结果。
    */
    pub fn check(&self, tool_name: &str, permissions: &[ToolPermission]) -> Option<RuleBehavior> {
        let mut first_ask: Option<RuleBehavior> = None;
        let mut first_allow: Option<RuleBehavior> = None;

        for rule in &self.rules {
            if !rule.matches(tool_name, permissions) {
                continue;
            }
            match &rule.behavior {
                RuleBehavior::Deny { .. } => return Some(rule.behavior.clone()),
                RuleBehavior::Ask { .. } => {
                    if first_ask.is_none() {
                        first_ask = Some(rule.behavior.clone());
                    }
                }
                RuleBehavior::Allow => {
                    if first_allow.is_none() {
                        first_allow = Some(rule.behavior.clone());
                    }
                }
            }
        }

        first_ask.or(first_allow)
    }

    /// 获取指定来源的所有规则
    pub fn rules_by_source(&self, source: RuleSource) -> Vec<&PermissionRule> {
        self.rules.iter().filter(|r| r.source == source).collect()
    }

    /// 移除指定来源的所有规则
    pub fn remove_by_source(&mut self, source: RuleSource) {
        self.rules.retain(|r| r.source != source);
    }

    /** 移除所有匹配指定匹配器字符串的规则

    返回被移除的规则数量。匹配语义与 `parse_rule()` 中 `AddRule` 使用的语义相同。
    */
    pub fn remove_by_matcher(&mut self, matcher_str: &str) -> usize {
        let before = self.rules.len();
        self.rules
            .retain(|r| !r.matcher.matches_matcher_str(matcher_str));
        before - self.rules.len()
    }

    /// 清空所有规则
    pub fn clear(&mut self) {
        self.rules.clear();
    }

    /// 获取规则数量
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// 获取所有规则
    pub fn all_rules(&self) -> &[PermissionRule] {
        &self.rules
    }
}

// ── 权限策略 Trait ────────────────────────────────────────────────────────────

/// 权限策略 trait
pub trait PermissionPolicy: Send + Sync {
    /// 检查一次工具调用所需的权限，返回对应的权限决策。
    fn check<'a>(
        &'a self,
        tool_name: &'a str,
        permissions: &'a [ToolPermission],
    ) -> BoxFuture<'a, PermissionDecision>;
}

/// 默认权限策略（保留向后兼容性）
pub struct DefaultPermissionPolicy {
    /// 已直接授予、无需审批的权限集合
    granted: HashSet<ToolPermission>,
    /// 需要用户审批的权限集合
    approval_required: HashSet<ToolPermission>,
}

impl Default for DefaultPermissionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultPermissionPolicy {
    /// 创建一个默认权限策略：Execute 和 Sensitive 权限需要审批，其余未授权权限将被拒绝。
    pub fn new() -> Self {
        let mut approval_required = HashSet::new();
        approval_required.insert(ToolPermission::Execute);
        approval_required.insert(ToolPermission::Sensitive);

        Self {
            granted: HashSet::new(),
            approval_required,
        }
    }

    /// 直接授予指定权限（无需审批）
    pub fn grant(mut self, perm: ToolPermission) -> Self {
        self.granted.insert(perm);
        self.approval_required.remove(&perm);
        self
    }

    /// 将指定权限标记为需要用户审批
    pub fn require_approval(mut self, perm: ToolPermission) -> Self {
        self.approval_required.insert(perm);
        self.granted.remove(&perm);
        self
    }

    /// 授予全部权限类型（无需审批）
    pub fn grant_all(mut self) -> Self {
        self.granted.insert(ToolPermission::Read);
        self.granted.insert(ToolPermission::Write);
        self.granted.insert(ToolPermission::Network);
        self.granted.insert(ToolPermission::Execute);
        self.granted.insert(ToolPermission::Sensitive);
        self.approval_required.clear();
        self
    }
}

impl PermissionPolicy for DefaultPermissionPolicy {
    fn check<'a>(
        &'a self,
        _tool_name: &'a str,
        permissions: &'a [ToolPermission],
    ) -> BoxFuture<'a, PermissionDecision> {
        Box::pin(async move {
            if permissions.is_empty() {
                return PermissionDecision::Allow;
            }

            let mut need_approval = Vec::new();
            let mut denied = Vec::new();

            for perm in permissions {
                if self.granted.contains(perm) {
                    continue;
                }
                if self.approval_required.contains(perm) {
                    need_approval.push(*perm);
                } else {
                    denied.push(*perm);
                }
            }

            if !denied.is_empty() {
                let names: Vec<String> = denied.iter().map(|p| p.to_string()).collect();
                return PermissionDecision::Deny {
                    reason: format!("Unauthorized permissions: {}", names.join(", ")),
                };
            }

            if !need_approval.is_empty() {
                return PermissionDecision::RequireApproval;
            }

            PermissionDecision::Allow
        })
    }
}

// ── 单元测试 ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_mode_default() {
        let mode = PermissionMode::default();
        assert_eq!(mode, PermissionMode::Default);
        assert!(mode.requires_interaction());
        assert!(!mode.uses_classifier());
    }

    #[test]
    fn test_permission_mode_bypass() {
        let mode = PermissionMode::BypassPermissions;
        assert!(mode.allows_write());
        assert!(!mode.requires_interaction());
    }

    #[test]
    fn test_permission_mode_plan() {
        let mode = PermissionMode::Plan;
        assert!(!mode.allows_write());
        assert!(mode.requires_interaction());
    }

    #[test]
    fn test_permission_mode_auto() {
        let mode = PermissionMode::Auto;
        assert!(!mode.requires_interaction());
        assert!(mode.uses_classifier());
    }

    #[test]
    fn permission_mode_ids_and_aliases_are_framework_owned() {
        let cases = [
            ("default", PermissionMode::Default),
            ("ask", PermissionMode::Default),
            ("auto-edit", PermissionMode::AcceptEdits),
            ("acceptEdits", PermissionMode::AcceptEdits),
            ("full-auto", PermissionMode::BypassPermissions),
            ("bypassPermissions", PermissionMode::BypassPermissions),
            ("dont-ask", PermissionMode::DontAsk),
            ("dontAsk", PermissionMode::DontAsk),
            ("strict", PermissionMode::StrictConfirm),
        ];
        for (wire, expected) in cases {
            assert_eq!(wire.parse::<PermissionMode>(), Ok(expected));
            assert_eq!(expected.id(), expected.to_string());
        }
    }

    #[test]
    fn test_rule_source_ordering() {
        assert!(RuleSource::Session > RuleSource::CliArg);
        assert!(RuleSource::CliArg > RuleSource::UserSettings);
        assert!(RuleSource::UserSettings > RuleSource::ProjectSettings);
        assert!(RuleSource::ProjectSettings > RuleSource::LocalSettings);
        assert!(RuleSource::LocalSettings > RuleSource::Default);
    }

    #[test]
    fn test_rule_matcher_tool() {
        let matcher = RuleMatcher::Tool {
            name: "Bash".to_string(),
        };
        assert!(matcher.matches("Bash", &[]));
        assert!(!matcher.matches("Read", &[]));
    }

    #[test]
    fn test_rule_matcher_pattern() {
        let matcher = RuleMatcher::Pattern {
            pattern: "Bash".to_string(),
        };
        assert!(matcher.matches("Bash", &[]));
        assert!(matcher.matches("Bash(git:*)", &[]));
        assert!(!matcher.matches("BashExtra", &[]));
    }

    #[test]
    fn test_rule_matcher_wildcard() {
        let matcher = RuleMatcher::Pattern {
            pattern: "*".to_string(),
        };
        assert!(matcher.matches("Bash", &[]));
        assert!(matcher.matches("Read", &[]));
        assert!(matcher.matches("Write", &[]));
    }

    #[test]
    fn test_rule_matcher_permission() {
        let matcher = RuleMatcher::Permission {
            permission: ToolPermission::Execute,
        };
        assert!(matcher.matches("shell", &[ToolPermission::Execute]));
        assert!(!matcher.matches("read", &[ToolPermission::Read]));
    }

    #[test]
    fn test_permission_rule_create() {
        let rule = PermissionRule::allow(
            RuleMatcher::Tool {
                name: "Read".to_string(),
            },
            RuleSource::UserSettings,
        );
        assert_eq!(rule.behavior, RuleBehavior::Allow);
        assert_eq!(rule.source, RuleSource::UserSettings);
    }

    #[test]
    fn test_rule_registry_add() {
        let mut registry = RuleRegistry::new();

        // 添加低优先级规则
        registry.add_rule(PermissionRule::deny(
            RuleMatcher::All,
            "default deny".to_string(),
            RuleSource::Default,
        ));

        // 添加高优先级规则
        registry.add_rule(PermissionRule::allow(
            RuleMatcher::Tool {
                name: "Read".to_string(),
            },
            RuleSource::UserSettings,
        ));

        // 高优先级规则应该在前面
        assert_eq!(registry.rules[0].source, RuleSource::UserSettings);
        assert_eq!(registry.rules[1].source, RuleSource::Default);
    }

    #[test]
    fn test_rule_registry_check() {
        let mut registry = RuleRegistry::new();

        // 默认拒绝所有
        registry.add_rule(PermissionRule::deny(
            RuleMatcher::All,
            "default deny".to_string(),
            RuleSource::Default,
        ));

        // 用户设置允许 Read
        registry.add_rule(PermissionRule::allow(
            RuleMatcher::Tool {
                name: "Read".to_string(),
            },
            RuleSource::UserSettings,
        ));

        /*
        deny-first: 即使有高优先级 Allow，Deny All 仍然匹配
        Read 匹配 Deny(All) → 被拒绝
        */
        let result = registry.check("Read", &[]);
        assert_eq!(
            result,
            Some(RuleBehavior::Deny {
                reason: "default deny".to_string()
            })
        );

        // Bash 也匹配 Deny(All) → 被拒绝
        let result = registry.check("Bash", &[]);
        assert!(matches!(result, Some(RuleBehavior::Deny { .. })));
    }

    #[test]
    fn test_rule_registry_allow_without_deny() {
        let mut registry = RuleRegistry::new();

        // 只有 allow 规则
        registry.add_rule(PermissionRule::allow(
            RuleMatcher::Tool {
                name: "Read".to_string(),
            },
            RuleSource::UserSettings,
        ));

        // Read 匹配 Allow → 允许
        let result = registry.check("Read", &[]);
        assert_eq!(result, Some(RuleBehavior::Allow));

        // Bash 无匹配规则
        let result = registry.check("Bash", &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn remove_rule_keeps_other_sources_and_matchers() {
        let mut registry = RuleRegistry::new();
        let removable = PermissionRule::allow(
            RuleMatcher::Tool {
                name: "Read".to_string(),
            },
            RuleSource::UserSettings,
        );
        registry.add_rule(removable.clone());
        registry.add_rule(PermissionRule::deny(
            RuleMatcher::All,
            "default deny".to_string(),
            RuleSource::Default,
        ));

        assert!(registry.remove_rule(&removable));
        assert!(!registry.remove_rule(&removable));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_rule_registry_deny_first_ordering() {
        let mut registry = RuleRegistry::new();

        // 高优先级 allow
        registry.add_rule(PermissionRule::allow(
            RuleMatcher::Pattern {
                pattern: "Bash".to_string(),
            },
            RuleSource::UserSettings,
        ));
        // 低优先级 deny
        registry.add_rule(PermissionRule::deny(
            RuleMatcher::Pattern {
                pattern: "Bash(rm:*)".to_string(),
            },
            "dangerous".to_string(),
            RuleSource::Default,
        ));

        // Bash(rm:rf) — deny 匹配，deny 优先于 allow
        let result = registry.check("Bash(rm:rf)", &[]);
        assert!(matches!(result, Some(RuleBehavior::Deny { .. })));

        // Bash(ls) — 只有 allow 匹配
        let result = registry.check("Bash(ls)", &[]);
        assert_eq!(result, Some(RuleBehavior::Allow));
    }

    #[test]
    fn test_rule_registry_ask_between_deny_and_allow() {
        let mut registry = RuleRegistry::new();

        registry.add_rule(PermissionRule::allow(
            RuleMatcher::Pattern {
                pattern: "Bash".to_string(),
            },
            RuleSource::UserSettings,
        ));
        registry.add_rule(PermissionRule::ask(
            RuleMatcher::Pattern {
                pattern: "Bash(rm:*)".to_string(),
            },
            vec!["Confirm".to_string()],
            RuleSource::Default,
        ));

        // Bash(rm:rf) — Ask 规则匹配，优先于 Allow
        let result = registry.check("Bash(rm:rf)", &[]);
        assert!(matches!(result, Some(RuleBehavior::Ask { .. })));

        // Bash(git:status) — 只有 Allow 规则匹配
        let result = registry.check("Bash(git:status)", &[]);
        assert_eq!(result, Some(RuleBehavior::Allow));
    }

    #[test]
    fn test_permission_decision_is_allowed() {
        assert!(PermissionDecision::Allow.is_allowed());
        assert!(
            !PermissionDecision::Deny {
                reason: "test".to_string()
            }
            .is_allowed()
        );
    }

    #[test]
    fn test_permission_decision_requires_approval() {
        assert!(PermissionDecision::RequireApproval.requires_approval());
        assert!(
            PermissionDecision::Ask {
                suggestions: vec!["yes".to_string()]
            }
            .requires_approval()
        );
        assert!(!PermissionDecision::Allow.requires_approval());
    }

    #[tokio::test]
    async fn test_empty_permissions_allowed() {
        let policy = DefaultPermissionPolicy::new();
        let decision = policy.check("tool", &[]).await;
        assert!(matches!(decision, PermissionDecision::Allow));
    }

    #[tokio::test]
    async fn test_granted_permission() {
        let policy = DefaultPermissionPolicy::new().grant(ToolPermission::Read);
        let decision = policy.check("tool", &[ToolPermission::Read]).await;
        assert!(matches!(decision, PermissionDecision::Allow));
    }

    #[tokio::test]
    async fn test_execute_requires_approval() {
        let policy = DefaultPermissionPolicy::new();
        let decision = policy.check("tool", &[ToolPermission::Execute]).await;
        assert!(matches!(decision, PermissionDecision::RequireApproval));
    }

    #[tokio::test]
    async fn test_ungranted_denied() {
        let policy = DefaultPermissionPolicy::new();
        let decision = policy.check("tool", &[ToolPermission::Write]).await;
        assert!(matches!(decision, PermissionDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_grant_all() {
        let policy = DefaultPermissionPolicy::new().grant_all();
        let decision = policy
            .check(
                "tool",
                &[
                    ToolPermission::Read,
                    ToolPermission::Write,
                    ToolPermission::Execute,
                    ToolPermission::Sensitive,
                ],
            )
            .await;
        assert!(matches!(decision, PermissionDecision::Allow));
    }

    #[test]
    fn rule_wire_values_parse_into_framework_types() {
        assert_eq!(
            "tool:read_file".parse::<RuleMatcher>(),
            Ok(RuleMatcher::Tool {
                name: "read_file".to_string()
            })
        );
        assert_eq!(
            "deny".parse::<RuleBehavior>(),
            Ok(RuleBehavior::Deny {
                reason: "denied by rule".to_string(),
            })
        );
        assert_eq!("session".parse::<RuleSource>(), Ok(RuleSource::Session));
        assert!("tool:".parse::<RuleMatcher>().is_err());
    }
}
