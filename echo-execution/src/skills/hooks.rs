/*! Hook 系统 —— Echo Agent 框架的生命周期钩子。

Hook 允许技能（skill）和用户在 agent 生命周期的特定节点，通过运行命令、注入提示词
（prompt）、发起 HTTP 调用或调用 MCP 工具来扩展 agent 的行为。

## 架构角色

本模块提供共享的 hook 分发与扩展层。宿主应用是本地的、由用户控制的应用，因此用户配置的
命令、本地 HTTP 端点和 MCP 工具都被当作可信的扩展来对待。注册阶段仍然会拒绝格式错误的
配置以及意外的超长超时设置。

| 设计概念 | 本模块中的实现 |
|---------------|-------------------------------|
| `ToolExecutionEngine` | [`HookRegistry`] —— 所有 hook 事件的集中分发器 |
| `ToolHook` trait | [`HookAction`] 枚举 —— Command、Prompt、Permission、Http、McpTool、Agent、ActivateSkill |
| `HookDecision`（Allow/Deny/Modify） | [`HookResult`] —— `block`、`updated_input`、`permission_decision` |
| `ToolPolicy` | [`HooksDefinition`] + [`HookRule`] —— matcher → actions 映射（可通过 YAML 配置） |

工具 hook 通过共享的工具执行流水线接入：
- `PreToolUseHookStage`
- 用于 `PermissionRequest` 的权限阶段
- `PostToolUseHookStage`，负责把成功和失败的结果路由到各自不同的事件

## Hook 事件

| 事件 | 触发时机 | 可修改内容 |
|-------|------|-----------|
| `PreToolUse` | 工具执行之前 | 输入、权限（允许/阻止） |
| `PostToolUse` | 工具执行成功之后 | 输出、是否继续 |
| `PostToolUseFailure` | 工具执行失败之后 | 错误反馈 |
| `PermissionRequest` | 出现权限确认对话框时 | 自动批准/拒绝 |
| `PermissionDenied` | 权限被拒绝时 | 重试信号 |
| `SessionStart` | 会话开始或恢复时 | 注入上下文 |
| `SessionEnd` | 会话结束时 | 清理 |
| `Stop` | Agent 完成响应时 | 继续原因 |
| `StopFailure` | Agent 遇到不可恢复错误时 | 告警/恢复 |
| `Notification` | Agent 需要用户关注时 | 权限快捷方式 |
| `UserPromptSubmit` | 用户提交提示词时 | 注入上下文、阻止 |
| `PreCompact` | 上下文压缩之前 | 注入上下文 |
| `PostCompact` | 上下文压缩之后 | 注入上下文 |
| `ConfigChange` | 配置文件变更时 | 阻止/重新加载 |
| `InstructionsLoaded` | 技能/指令加载完成时 | 加载后校验 |
| `PostToolBatch` | 一批并行工具调用完成之后 | 聚合 |
| `SubagentStart` | 子 agent 派发之前 | 注入上下文 |
| `SubagentStop` | 子 agent 完成之后 | 注入结果 |
| `TaskCreated` | 任务节点进入可执行图时 | 注入上下文 |
| `TaskStarted` | 调度器认领任务开始执行时 | 注入上下文 |
| `TaskCompleted` | 任务到达终态时 | 注入结果 |
| `PluginLoaded` | 插件组件被激活时 | 加载后集成 |
| `PluginDisabled` | 插件组件被停用时 | 清理 |
| `PostMemoryWrite` | 记忆被持久化时 | 后续处理 |
| `MemoryLayerChange` | 记忆变更层级时 | 后续处理 |
| `SkillCandidateDetected` | 检测到技能候选项时 | 评审/通知 |
| `SkillLifecycleTransition` | 技能生命周期状态变化时 | 后续处理 |
| `SkillHealthCheck` | 技能健康检查完成时 | 后续处理 |
| `SkillPatchApplied` | 技能补丁应用完成时 | 后续处理 |
| `SkillMergeApplied` | 技能合并完成时 | 后续处理 |
| `RulePromoted` | 记忆被提升为 AGENTS.md 规则时 | 后续处理 |

## Hook 类型

| 类型 | 行为 |
|------|----------|
| `command` | 执行一条 shell 命令；JSON 上下文通过 stdin 传入 |
| `prompt` | 为 LLM 注入一条提示消息 |
| `permission` | 直接返回一个权限决策（allow/deny/ask） |
| `http` | 向某个 URL POST 事件数据并解析响应 |
| `mcp_tool` | 调用某个 MCP 服务器上的工具 |
| `agent` | 调用一个已配置的子 agent |
| `activate_skill` | 直接激活一个已发现的技能 |

## 格式（应用配置）

技能文件本身不携带私有的 Hook 扩展。Hook 由宿主应用配置或插件的 Hook 组件提供。

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "${SKILL_DIR}/validate.sh"
          timeout: 5
    - matcher: "Write"
      hooks:
        - type: prompt
          prompt: "Check file permissions before writing"
  PostToolUse:
    - matcher: "Edit|Write"
      hooks:
        - type: command
          command: "jq -r '.tool_input.file_path' | xargs prettier --write"
  Stop:
    - hooks:
        - type: command
          command: "osascript -e 'display notification \"Done\"'"
  SessionStart:
    - matcher: "startup"
      hooks:
        - type: prompt
          prompt: "Remember to use bun, not npm."
```
*/

// ── 从 echo-core 重新导出核心类型 ─────────────────────────────────

pub use echo_core::hooks::{
    CompressHookStats, HookContext, HookEvent, HookEventCategory, HookResult, HookSource,
    UnifiedHookExecutorFn,
};

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use echo_core::tools::permission::{PermissionDecision, PermissionMode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug, info, warn};

use crate::sandbox::{SandboxCommand, SandboxExecutor};
use crate::skills::minimal_hook_env_with_context;

// ── （HookEvent、HookContext、HookResult、CompressHookStats、HookSource 现已移至 echo-core）──

// ── Hook 动作 ────────────────────────────────────────────────────────

/// hook 动作的默认超时时间（秒）。
const fn default_hook_timeout() -> u64 {
    600
}

/// hook 允许的最大超时时间（秒）。用于防止意外的超长超时。
const MAX_HOOK_TIMEOUT: u64 = 3600;

/// 命令字符串的最大长度（以 Unicode 标量值计）。
const MAX_COMMAND_CHARS: usize = 32 * 1024;

/// 单个 hook 动作。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum HookAction {
    /// 执行一条 shell 命令。命令通过 stdin 接收 JSON 格式的上下文。
    Command {
        /// 要执行的 shell 命令字符串。
        command: String,
        /// 使用的 shell 类型（例如 "bash"、"powershell"），默认为 bash。
        #[serde(default)]
        shell: Option<String>,
        /// 命令执行超时时间（秒）。
        #[serde(default = "default_hook_timeout")]
        timeout: u64,
    },
    /// 为 LLM 注入一条提示词供其参考。
    Prompt {
        /// 注入的提示词内容。
        prompt: String,
    },
    /// 直接返回一个权限决策。
    Permission {
        /// 决策："allow" | "deny" | "ask"
        decision: String,
        /// 拒绝（deny）时的原因
        #[serde(default)]
        reason: Option<String>,
        /// 询问（ask）时的建议选项
        #[serde(default)]
        suggestions: Vec<String>,
    },
    /// 向某个 URL POST 事件数据并解析响应。
    Http {
        /// 请求目标 URL。
        url: String,
        /// HTTP 方法，默认为 POST。
        #[serde(default)]
        method: Option<String>,
        /// 附加的请求头。
        #[serde(default)]
        headers: Option<HashMap<String, String>>,
        /// 请求超时时间（秒）。
        #[serde(default = "default_hook_timeout")]
        timeout: u64,
    },
    /// 调用某个 MCP 服务器上的工具。
    #[serde(rename = "mcp_tool")]
    McpTool {
        /// MCP 服务器名称。
        server: String,
        /// 要调用的工具名称。
        tool: String,
        /// 传给工具的参数。
        #[serde(default)]
        arguments: Option<Value>,
        /// 调用超时时间（秒）。
        #[serde(default = "default_hook_timeout")]
        timeout: u64,
    },
    /// 派生一个子 agent 来处理该 hook 动作。
    Subagent {
        /// 要调用的子 agent 名称。
        name: String,
        /// 传给子 agent 的任务描述。
        #[serde(default)]
        task: Option<String>,
        /// 超时时间（秒）。
        #[serde(default = "default_hook_timeout")]
        timeout: u64,
    },
    /// 不经过 LLM，直接激活一个技能。
    /// 匹配成功时，hook 引擎会调用 `ReactAgent::activate_skill(skill)`。
    /// `reason` 会作为系统提示展示给模型，说明激活原因。
    #[serde(rename = "activate_skill")]
    ActivateSkill {
        /// 要激活的技能名称（必须匹配一个已发现的技能）。
        skill: String,
        /// 展示给模型的可读原因说明。
        #[serde(default)]
        reason: String,
    },
}

impl HookAction {
    /// 校验该 hook 动作，对不安全/无效的配置返回错误。
    ///
    /// 在 hook 注册期间调用，用于尽早发现配置错误。
    pub fn validate(&self) -> Result<(), String> {
        match self {
            HookAction::Command {
                command, timeout, ..
            } => {
                if command.is_empty() {
                    return Err("Command hook has empty command string".into());
                }
                let command_chars = command.chars().count();
                if command_chars > MAX_COMMAND_CHARS {
                    return Err(format!(
                        "Command hook exceeds max length ({} > {} characters)",
                        command_chars, MAX_COMMAND_CHARS
                    ));
                }
                if *timeout > MAX_HOOK_TIMEOUT {
                    return Err(format!(
                        "Command hook timeout {}s exceeds maximum {}s",
                        timeout, MAX_HOOK_TIMEOUT
                    ));
                }
            }
            HookAction::Prompt { prompt } => {
                if prompt.is_empty() {
                    return Err("Prompt hook has empty prompt string".into());
                }
            }
            HookAction::Permission { decision, .. } => {
                if !matches!(decision.as_str(), "allow" | "deny" | "ask") {
                    return Err(format!(
                        "Permission hook has invalid decision '{}' (expected: allow, deny, ask)",
                        decision
                    ));
                }
            }
            HookAction::Http { url, timeout, .. } => {
                if url.is_empty() {
                    return Err("Http hook has empty url".into());
                }
                let parsed = reqwest::Url::parse(url)
                    .map_err(|error| format!("Http hook has invalid url: {error}"))?;
                match parsed.scheme() {
                    "https" => {}
                    "http" if is_local_http_url(&parsed) => {}
                    "http" => {
                        return Err(
                            "Http hook must use https unless it targets a local address".into()
                        );
                    }
                    scheme => {
                        return Err(format!(
                            "Http hook has unsupported url scheme '{scheme}' (expected http or https)"
                        ));
                    }
                }
                if *timeout > MAX_HOOK_TIMEOUT {
                    return Err(format!(
                        "Http hook timeout {}s exceeds maximum {}s",
                        timeout, MAX_HOOK_TIMEOUT
                    ));
                }
            }
            HookAction::McpTool {
                server,
                tool,
                timeout,
                ..
            } => {
                if server.is_empty() {
                    return Err("McpTool hook has empty server name".into());
                }
                if tool.is_empty() {
                    return Err("McpTool hook has empty tool name".into());
                }
                if *timeout > MAX_HOOK_TIMEOUT {
                    return Err(format!(
                        "McpTool hook timeout {}s exceeds maximum {}s",
                        timeout, MAX_HOOK_TIMEOUT
                    ));
                }
            }
            HookAction::Subagent { name, timeout, .. } => {
                if name.is_empty() {
                    return Err("Subagent hook has empty subagent name".into());
                }
                if *timeout > MAX_HOOK_TIMEOUT {
                    return Err(format!(
                        "Subagent hook timeout {}s exceeds maximum {}s",
                        timeout, MAX_HOOK_TIMEOUT
                    ));
                }
            }
            HookAction::ActivateSkill { skill, .. } => {
                if skill.is_empty() {
                    return Err("ActivateSkill hook has empty skill name".into());
                }
            }
        }
        Ok(())
    }

    /// 稳定的动作名称，用于诊断信息和 dry-run 输出。
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Command { .. } => "command",
            Self::Prompt { .. } => "prompt",
            Self::Permission { .. } => "permission",
            Self::Http { .. } => "http",
            Self::McpTool { .. } => "mcp_tool",
            Self::Subagent { .. } => "subagent",
            Self::ActivateSkill { .. } => "activate_skill",
        }
    }
}

/// 判断给定 URL 是否指向本地/私有地址（用于放行非 https 的 http hook）。
fn is_local_http_url(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    let ip_literal = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(address) = ip_literal.parse::<std::net::IpAddr>() {
        return match address {
            std::net::IpAddr::V4(address) => {
                address.is_loopback()
                    || address.is_private()
                    || address.is_link_local()
                    || address.is_unspecified()
            }
            std::net::IpAddr::V6(address) => {
                address.is_loopback()
                    || address.is_unique_local()
                    || address.is_unicast_link_local()
                    || address.is_unspecified()
            }
        };
    }

    let normalized_host = host.to_ascii_lowercase();
    normalized_host == "localhost"
        || normalized_host.ends_with(".localhost")
        || !normalized_host.contains('.')
        || normalized_host.ends_with(".local")
        || normalized_host.ends_with(".lan")
}

// ── Hook 规则 ──────────────────────────────────────────────────────────

/** 一条 hook 规则：一个匹配模式（matcher）+ 一个或多个动作。

`matcher` 字段：
- 对于工具事件：匹配工具名称（精确匹配、glob 通配，或以 `|` 分隔的多个候选）。
- 对于生命周期事件：匹配事件提示（例如 `"startup"`、`"permission_prompt"`）。
- `"*"` 或空字符串匹配任意情况。
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookRule {
    /// 匹配模式。
    #[serde(default)]
    pub matcher: String,

    /// 匹配成功后要执行的动作列表。
    pub hooks: Vec<HookAction>,
}

// ── Hooks 定义 ───────────────────────────────────────────────────────

/** 来自编程式 Skill 描述符或宿主配置的完整 hooks 定义。

使用 `HashMap<HookEvent, Vec<HookRule>>`，这样新增任何事件类型都无需修改本结构体即可
自动支持。
*/
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HooksDefinition {
    /// 事件 -> 规则列表 的映射，支持所有 `HookEvent` 变体。
    #[serde(flatten)]
    pub rules: HashMap<HookEvent, Vec<HookRule>>,
}

impl HooksDefinition {
    /// 获取指定事件对应的规则列表。
    pub fn rules_for(&self, event: HookEvent) -> &[HookRule] {
        self.rules.get(&event).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// 判断是否未定义任何 hook。
    pub fn is_empty(&self) -> bool {
        self.rules.values().all(|v| v.is_empty())
    }

    /// 为指定事件添加规则。
    pub fn add_rules(&mut self, event: HookEvent, rules: Vec<HookRule>) {
        let (rules, _) = validate_hook_rules(event, rules, "definition");
        if !rules.is_empty() {
            self.rules.entry(event).or_default().extend(rules);
        }
    }

    /// 将另一个定义合并进当前定义。
    pub fn merge(&mut self, other: HooksDefinition) {
        for (event, rules) in other.rules {
            self.add_rules(event, rules);
        }
    }
}

/// 校验一批规则中的所有动作，剔除无效动作后返回保留的规则以及被跳过的数量。
fn validate_hook_rules(
    event: HookEvent,
    rules: Vec<HookRule>,
    source: &str,
) -> (Vec<HookRule>, usize) {
    let mut kept_rules = Vec::with_capacity(rules.len());
    let mut skipped = 0usize;
    for mut rule in rules {
        rule.hooks.retain(|action| match action.validate() {
            Ok(()) => true,
            Err(_) => {
                warn!(
                    source,
                    ?event,
                    action = action.kind(),
                    "Invalid hook action skipped"
                );
                skipped = skipped.saturating_add(1);
                false
            }
        });
        if !rule.hooks.is_empty() {
            kept_rules.push(rule);
        }
    }
    (kept_rules, skipped)
}

/// 校验整份 hooks 定义（遍历所有事件），返回清理后的定义以及被跳过的动作总数。
fn validate_hook_definition(definition: HooksDefinition, source: &str) -> (HooksDefinition, usize) {
    let mut clean = HooksDefinition::default();
    let mut skipped = 0usize;
    for (event, rules) in definition.rules {
        let (rules, event_skipped) = validate_hook_rules(event, rules, source);
        skipped = skipped.saturating_add(event_skipped);
        if !rules.is_empty() {
            clean.rules.insert(event, rules);
        }
    }
    (clean, skipped)
}

// ── MCP 工具执行器 ─────────────────────────────────────────────────

/** 用于从 hook 中执行 MCP 工具调用的类型擦除回调。

agent 层通过 [`HookRegistry::set_mcp_executor`] 注入该回调，使得 [`HookAction::McpTool`]
hook 能够调用 agent 的 MCP 管理器，而无需让 echo-execution 依赖 echo-integration。
*/
pub type McpExecutorFn = Arc<
    dyn Fn(String, String, Option<Value>) -> Pin<Box<dyn Future<Output = HookResult> + Send>>
        + Send
        + Sync,
>;

/** 用于从 hook 中派生子 agent 的类型擦除回调。

agent 层通过 [`HookRegistry::set_subagent_executor`] 注入该回调，使得
[`HookAction::Subagent`] hook 能够按名称派发子 agent，而无需让 echo-execution 依赖
agent 层。接收 (subagent_name, task_prompt)，返回子 agent 的输出文本。
*/
pub type SubagentExecutorFn = Arc<
    dyn Fn(String, String) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
>;

/** 类型擦除的编程式 hook 执行器。

嵌入方（例如把 hook 事件转发给宿主语言实现的 SDK 扩展桥接层）通过注入闭包的方式，
使编程式 hook 能够遵循与数据驱动规则相同的合并、修改与短路语义 —— 同时不让
echo-execution 依赖任何具体宿主。与已有的
[`McpExecutorFn`]/[`SubagentExecutorFn`] 注入模式保持一致。
*/
pub type ProgrammaticHookFn =
    Arc<dyn Fn(HookContext) -> Pin<Box<dyn Future<Output = HookResult> + Send>> + Send + Sync>;

tokio::task_local! {
    /// 当前任务内子 agent hook 的嵌套深度，用于防止子 agent hook 递归触发自身。
    static SUBAGENT_HOOK_DEPTH: u8;
}

// ── Hook 注册表 ──────────────────────────────────────────────────────────

/// 汇总所有来源（技能与用户配置）的 hook 注册表。
// 手动实现 Clone，以保证类型擦除的运行时回调在克隆快照中仍然保留。
#[derive(Default)]
pub struct HookRegistry {
    /// 来源 -> hooks 定义 的映射。
    sources: HashMap<HookSource, RegisteredHook>,
    /// 用于执行 hook 命令的可选沙箱管理器。
    sandbox: Option<Arc<dyn SandboxExecutor>>,
    /// 用于 Http hook 动作的可选 HTTP 客户端。
    http_client: Option<reqwest::Client>,
    /// 用于 McpTool hook 动作的可选 MCP 工具执行器。
    mcp_executor: Option<McpExecutorFn>,
    /// 用于 Subagent hook 动作的可选子 agent 执行器。
    subagent_executor: Option<SubagentExecutorFn>,
    /// 编程式（闭包）hook 列表，按名称排序以保证确定性的执行顺序。
    programmatic: Vec<ProgrammaticHookEntry>,
}

/// 一个已注册的编程式 hook：一个具名闭包，加上它订阅的事件列表
/// （空列表表示订阅所有事件）。
#[derive(Clone)]
struct ProgrammaticHookEntry {
    /// hook 名称，用于去重和排序。
    name: String,
    /// 订阅的事件列表；为空表示订阅所有事件。
    events: Vec<HookEvent>,
    /// 实际执行的闭包。
    executor: ProgrammaticHookFn,
}

impl Clone for HookRegistry {
    fn clone(&self) -> Self {
        Self {
            sources: self.sources.clone(),
            sandbox: self.sandbox.clone(),
            http_client: self.http_client.clone(),
            mcp_executor: self.mcp_executor.clone(),
            subagent_executor: self.subagent_executor.clone(),
            programmatic: self.programmatic.clone(),
        }
    }
}

/// 一个来源已注册的 hook 定义及其运行所需的目录信息。
#[derive(Debug, Clone)]
struct RegisteredHook {
    /// 该来源的 hooks 定义。
    definition: HooksDefinition,
    /// 该来源对应的目录，用作命令的工作目录及环境变量来源。
    source_dir: String,
    /// 该来源对应的插件数据目录（仅插件来源存在）。
    plugin_data_dir: Option<String>,
}

/// 针对某个 hook 上下文将会执行的一个动作，不产生任何副作用（用于 dry-run）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookDryRunMatch {
    /// 该动作所属的 hook 来源（如 "user_config"、"skill:xxx"）。
    pub source: String,
    /// 命中的匹配模式。
    pub matcher: String,
    /// 动作类型名称。
    pub action: String,
}

/// 匹配某个 hook 上下文但不实际执行任何动作的结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HookDryRunResult {
    /// 被匹配的事件类型。
    pub event: HookEvent,
    /// 所有命中的动作列表。
    pub matches: Vec<HookDryRunMatch>,
}

impl HookRegistry {
    /// 创建一个空的 hook 注册表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 以构建器（builder）风格附加一个沙箱管理器，用于执行 hook 命令。
    pub fn with_sandbox_manager(mut self, manager: Arc<dyn SandboxExecutor>) -> Self {
        self.sandbox = Some(manager);
        self
    }

    /// 附加或替换沙箱管理器。
    pub fn set_sandbox_manager(&mut self, manager: Arc<dyn SandboxExecutor>) {
        self.sandbox = Some(manager);
    }

    /// 注册来自某个技能的 hooks。
    pub fn register(&mut self, skill_name: &str, skill_dir: &str, definition: HooksDefinition) {
        let source = format!("skill:{skill_name}");
        let (definition, skipped) = validate_hook_definition(definition, &source);
        if definition.is_empty() {
            return;
        }
        info!(
            skill = skill_name,
            rule_count = definition.rules.values().map(|v| v.len()).sum::<usize>(),
            skipped,
            "Registered skill hooks"
        );
        self.sources.insert(
            HookSource::Skill(skill_name.to_string()),
            RegisteredHook {
                definition,
                source_dir: skill_dir.to_string(),
                plugin_data_dir: None,
            },
        );
    }

    /// 注册来自用户配置的 hooks。
    pub fn register_user_hooks(&mut self, definition: HooksDefinition) {
        let (definition, skipped) = validate_hook_definition(definition, "user_config");
        if definition.is_empty() {
            return;
        }
        info!(
            rule_count = definition.rules.values().map(|v| v.len()).sum::<usize>(),
            skipped, "Registered user hooks from config"
        );
        self.sources.insert(
            HookSource::UserConfig,
            RegisteredHook {
                definition,
                source_dir: String::new(),
                plugin_data_dir: None,
            },
        );
    }

    /** 注册由已安装插件贡献的 hooks。

    这是 [`register`](Self::register) 面向插件来源的对应版本：它将定义存放在
    [`HookSource::Plugin`] 下，使插件 hook 与技能、用户配置 hook 保持独立的来源身份
    （对应审计项 P0-2）。此前插件 hook 被归档在 `HookSource::Skill("plugin:…")` 下，
    这样做的问题是：（1）从未真正用到 `HookSource::Plugin` 这个变体；（2）导致插件
    hook 在 `list_sources` / 热重载来源替换时与技能 hook 无法区分。

    每个动作都会预先校验（见 [`HookAction::validate`]）；无效动作会被记录日志并丢弃，
    而不是中止整个插件的注册（单个格式错误的 hook 不应该拖累插件的其他组件）。
    对同一插件重复注册会替换其先前的定义，从而支持“重载时重建”而无需手动反注册。
    */
    pub fn register_plugin_hooks(
        &mut self,
        plugin_name: &str,
        source_dir: &str,
        plugin_data_dir: &str,
        definition: HooksDefinition,
    ) -> bool {
        if definition.is_empty() {
            return false;
        }
        let source = format!("plugin:{plugin_name}");
        let total = definition.rules.values().map(|v| v.len()).sum::<usize>();
        let (clean, skipped) = validate_hook_definition(definition, &source);
        if clean.is_empty() {
            warn!(
                plugin = plugin_name,
                total,
                skipped,
                "Plugin registered no valid hooks after validation; nothing registered"
            );
            return false;
        }
        info!(
            plugin = plugin_name,
            rule_count = clean.rules.values().map(|v| v.len()).sum::<usize>(),
            skipped,
            "Registered plugin hooks"
        );
        self.sources.insert(
            HookSource::Plugin(plugin_name.to_string()),
            RegisteredHook {
                definition: clean,
                source_dir: source_dir.to_string(),
                plugin_data_dir: Some(plugin_data_dir.to_string()),
            },
        );
        true
    }

    /// 反注册来自某个来源的 hooks。
    pub fn unregister(&mut self, source: &HookSource) -> bool {
        self.sources.remove(source).is_some()
    }

    /// 按名称注册（或替换）一个编程式 hook。`events` 限定该执行器关注的事件范围；
    /// 空切片表示订阅所有事件。编程式 hook 会在所有数据驱动来源之后按名称顺序运行，
    /// 合并与短路语义与数据驱动规则相同。
    pub fn set_programmatic_hook(
        &mut self,
        name: &str,
        events: &[HookEvent],
        executor: ProgrammaticHookFn,
    ) {
        let entry = ProgrammaticHookEntry {
            name: name.to_string(),
            events: events.to_vec(),
            executor,
        };
        match self
            .programmatic
            .iter_mut()
            .find(|existing| existing.name == entry.name)
        {
            Some(existing) => *existing = entry,
            None => {
                self.programmatic.push(entry);
                self.programmatic
                    .sort_by(|left, right| left.name.cmp(&right.name));
            }
        }
    }

    /// 按名称移除一个编程式 hook；若该 hook 确实已注册则返回 true。
    pub fn remove_programmatic_hook(&mut self, name: &str) -> bool {
        let before = self.programmatic.len();
        self.programmatic.retain(|entry| entry.name != name);
        before != self.programmatic.len()
    }

    /// 清除所有用户配置的 hooks（保留技能 hook 不受影响）。
    pub fn clear_user_hooks(&mut self) -> bool {
        self.sources.remove(&HookSource::UserConfig).is_some()
    }

    /// 列出所有已注册的 hook 来源及各自的规则数量。
    pub fn list_sources(&self) -> Vec<(String, usize)> {
        self.sources
            .iter()
            .map(|(source, registered)| {
                let name = match source {
                    HookSource::UserConfig => "user_config".to_string(),
                    HookSource::Skill(name) => format!("skill:{}", name),
                    HookSource::Plugin(name) => format!("plugin:{}", name),
                };
                let count = registered.definition.rules.values().map(|v| v.len()).sum();
                (name, count)
            })
            .collect()
    }

    /// 判断是否未注册任何 hook。
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty() && self.programmatic.is_empty()
    }

    /// 判断是否为指定事件注册了任何 hook。
    pub fn has_hooks_for(&self, event: HookEvent) -> bool {
        self.sources
            .values()
            .any(|r| !r.definition.rules_for(event).is_empty())
            || self
                .programmatic
                .iter()
                .any(|entry| entry.events.is_empty() || entry.events.contains(&event))
    }

    /// 针对一个具体上下文匹配 hooks，但不执行任何动作。
    pub fn dry_run(&self, context: &HookContext) -> HookDryRunResult {
        let mut matches = Vec::new();
        for source in self.sorted_sources() {
            let Some(registered) = self.sources.get(source) else {
                continue;
            };
            for rule in registered.definition.rules_for(context.event) {
                if !matches_hook(&rule.matcher, context) {
                    continue;
                }
                for action in &rule.hooks {
                    matches.push(HookDryRunMatch {
                        source: source.to_string(),
                        matcher: rule.matcher.clone(),
                        action: action.kind().to_string(),
                    });
                }
            }
        }
        HookDryRunResult {
            event: context.event,
            matches,
        }
    }

    /// 按确定的优先级顺序排列 hook 来源：用户配置最先，其次是按名称排序的插件，
    /// 最后是按名称排序的技能。
    fn sorted_sources(&self) -> Vec<&HookSource> {
        let mut sources: Vec<&HookSource> = self.sources.keys().collect();
        sources.sort_by(|a, b| match (a, b) {
            (HookSource::UserConfig, _) => std::cmp::Ordering::Less,
            (_, HookSource::UserConfig) => std::cmp::Ordering::Greater,
            (HookSource::Plugin(a), HookSource::Plugin(b)) => a.cmp(b),
            (HookSource::Plugin(_), _) => std::cmp::Ordering::Less,
            (_, HookSource::Plugin(_)) => std::cmp::Ordering::Greater,
            (HookSource::Skill(a), HookSource::Skill(b)) => a.cmp(b),
        });
        sources
    }

    /// 设置用于 Http hook 动作的 HTTP 客户端。
    pub fn set_http_client(&mut self, client: reqwest::Client) {
        self.http_client = Some(client);
    }

    /** 设置用于 McpTool hook 动作的 MCP 工具执行器。

    该执行器接收 (server_name, tool_name, arguments)，应当调用对应 MCP 服务器上的
    工具，并返回一个 [`HookResult`]。
    */
    pub fn set_mcp_executor(&mut self, executor: McpExecutorFn) {
        self.mcp_executor = Some(executor);
    }

    /** 设置用于 Subagent hook 动作的子 agent 执行器。

    该执行器接收 (agent_name, task_prompt)，应当派发指定名称的子 agent，并在
    成功时返回其输出文本。
    */
    pub fn set_subagent_executor(&mut self, executor: SubagentExecutorFn) {
        self.subagent_executor = Some(executor);
    }

    // -- 公开的执行方法 --

    /// 执行所有匹配的 PreToolUse hooks。
    pub async fn run_pre_tool_use(
        &self,
        tool_name: &str,
        tool_input: &Value,
        session_id: &str,
    ) -> HookResult {
        let context = HookContext::for_pre_tool_use(tool_name, tool_input, session_id, "");
        self.run_hooks(&context).await
    }

    /// 执行所有匹配的 PostToolUse hooks。
    pub async fn run_post_tool_use(
        &self,
        tool_name: &str,
        tool_input: &Value,
        tool_output: &str,
        session_id: &str,
    ) -> HookResult {
        let context =
            HookContext::for_post_tool_use(tool_name, tool_input, tool_output, session_id, "");
        self.run_hooks(&context).await
    }

    /// 执行所有匹配的 PostToolUseFailure hooks。
    pub async fn run_post_tool_use_failure(
        &self,
        tool_name: &str,
        tool_input: &Value,
        tool_error: &str,
        session_id: &str,
    ) -> HookResult {
        let context = HookContext::for_post_tool_use_failure(
            tool_name, tool_input, tool_error, session_id, "",
        );
        self.run_hooks(&context).await
    }

    /// 执行某个生命周期事件对应的 hooks。
    pub async fn run_lifecycle_hooks(&self, context: &HookContext) -> HookResult {
        self.run_hooks(context).await
    }

    // -- 核心执行引擎 --

    /// 依次遍历所有数据驱动来源与编程式 hook，匹配并执行动作，合并出最终结果。
    async fn run_hooks(&self, context: &HookContext) -> HookResult {
        let event = context.event;
        let mut combined = HookResult::default();

        // 来源排序：用户配置优先，然后按字母顺序排列技能
        for source in self.sorted_sources() {
            let Some(registered) = self.sources.get(source) else {
                continue;
            };
            let rules = registered.definition.rules_for(event);
            let runtime = HookActionRuntime {
                plugin_data_dir: registered.plugin_data_dir.as_deref(),
                sandbox: self.sandbox.as_ref(),
                http_client: self.http_client.as_ref(),
                mcp_executor: self.mcp_executor.as_ref(),
                subagent_executor: self.subagent_executor.as_ref(),
            };

            for rule in rules {
                if !matches_hook(&rule.matcher, context) {
                    continue;
                }

                debug!(
                    source = %source,
                    event = ?event,
                    "Hook matched"
                );

                for action in &rule.hooks {
                    let result =
                        execute_action(action, &registered.source_dir, context, &runtime).await;

                    merge_result(&mut combined, result);

                    if combined.stop_propagation || combined.block {
                        return combined;
                    }
                }
            }
        }

        // 编程式 hook 在所有数据驱动来源之后按名称顺序运行，采用相同的合并/短路
        // 语义。空的事件订阅表示关注所有事件。
        for entry in &self.programmatic {
            if !entry.events.is_empty() && !entry.events.contains(&event) {
                continue;
            }
            let result = (entry.executor)(context.clone()).await;
            merge_result(&mut combined, result);
            if combined.stop_propagation || combined.block {
                return combined;
            }
        }

        combined
    }
}

// ── 匹配器 ────────────────────────────────────────────────────────────

/// 所有 hook 事件通用的匹配函数。
fn matches_hook(matcher: &str, context: &HookContext) -> bool {
    // "*" 或空字符串匹配任意情况
    if matcher == "*" || matcher.is_empty() {
        return true;
    }

    // 工具事件：与 tool_name 匹配
    if context.event.is_tool_event()
        && let Some(ref tool_name) = context.tool_name
        && matches_tool_name(matcher, tool_name)
    {
        return true;
    }

    // 非工具事件（生命周期、子 agent、任务）：与 context.matcher 提示匹配
    if let Some(ref hint) = context.matcher {
        // 精确匹配
        if matcher == hint.as_str() {
            return true;
        }
        // 以 `|` 分隔的多个候选（例如 "Edit|Write" 或 "startup|resume"）
        for part in matcher.split('|') {
            let part = part.trim();
            if part == hint.as_str() {
                return true;
            }
            // 尝试将每个候选作为 glob 模式匹配
            if let Ok(pattern) = glob::Pattern::new(part)
                && pattern.matches(hint)
            {
                return true;
            }
        }
        // 尝试将完整 matcher 作为 glob 模式匹配
        if let Ok(pattern) = glob::Pattern::new(matcher)
            && pattern.matches(hint)
        {
            return true;
        }
    }

    false
}

/// 将工具名称与模式进行匹配（精确匹配、glob 通配，或带括号的前缀匹配）。
fn matches_tool_name(matcher: &str, tool_name: &str) -> bool {
    if matcher == tool_name {
        return true;
    }
    // 以 `|` 分隔的多个候选
    if matcher.contains('|') {
        return matcher
            .split('|')
            .any(|part| matches_tool_name(part.trim(), tool_name));
    }
    // glob 匹配
    if let Ok(pattern) = glob::Pattern::new(matcher)
        && pattern.matches(tool_name)
    {
        return true;
    }
    // 前缀匹配，例如模式 "Bash" 匹配 "Bash(git:*)"
    if tool_name.starts_with(matcher)
        && tool_name.len() > matcher.len()
        && tool_name.as_bytes().get(matcher.len()) == Some(&b'(')
    {
        return true;
    }
    false
}

// ── Action Execution ───────────────────────────────────────────────────

struct HookActionRuntime<'a> {
    plugin_data_dir: Option<&'a str>,
    sandbox: Option<&'a Arc<dyn SandboxExecutor>>,
    http_client: Option<&'a reqwest::Client>,
    mcp_executor: Option<&'a McpExecutorFn>,
    subagent_executor: Option<&'a SubagentExecutorFn>,
}

async fn execute_action(
    action: &HookAction,
    source_dir: &str,
    context: &HookContext,
    runtime: &HookActionRuntime<'_>,
) -> HookResult {
    match action {
        HookAction::Command {
            command,
            shell,
            timeout,
        } => {
            execute_command_hook(
                command,
                shell.as_deref(),
                *timeout,
                source_dir,
                runtime.plugin_data_dir,
                context,
                runtime.sandbox,
            )
            .await
        }
        HookAction::Prompt { prompt } => {
            let mut result = HookResult::default();
            result.messages.push(prompt.clone());
            result
        }
        HookAction::Permission {
            decision,
            reason,
            suggestions,
        } => {
            let mut result = HookResult::default();
            match decision.as_str() {
                "allow" => {
                    result.permission_decision = Some(PermissionDecision::Allow);
                }
                "deny" => {
                    let reason_text = reason.clone().unwrap_or_else(|| "Hook denied".to_string());
                    result.block = true;
                    result.block_reason = Some(reason_text.clone());
                    result.permission_decision = Some(PermissionDecision::Deny {
                        reason: reason_text,
                    });
                }
                "ask" => {
                    result.permission_decision = Some(PermissionDecision::Ask {
                        suggestions: suggestions.clone(),
                    });
                }
                _ => {
                    warn!(
                        event = %context.event.as_str(),
                        action = "permission",
                        "Unknown permission decision from hook"
                    );
                }
            }
            // Permission decisions are reduced across every matching source.
            // In particular, an early allow/ask must not hide a later deny.
            // A deny still sets `block` above and therefore stops execution
            // after the highest-priority decision has been observed; callers
            // can still request an explicit propagation stop with `continue`.
            result
        }
        HookAction::Http {
            url,
            method,
            headers,
            timeout,
        } => {
            execute_http_hook(
                url,
                method.as_deref(),
                headers.as_ref(),
                *timeout,
                context,
                runtime.http_client,
            )
            .await
        }
        HookAction::McpTool {
            server,
            tool,
            arguments,
            timeout,
        } => match runtime.mcp_executor {
            Some(executor) => {
                let fut = executor(server.clone(), tool.clone(), arguments.clone());
                if *timeout > 0 {
                    match tokio::time::timeout(Duration::from_secs(*timeout), fut).await {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(
                                event = %context.event.as_str(),
                                action = "mcp_tool",
                                timeout_secs = *timeout,
                                "McpTool hook timed out"
                            );
                            hook_execution_message(format!(
                                "McpTool hook {server}::{tool} timed out after {timeout}s"
                            ))
                        }
                    }
                } else {
                    fut.await
                }
            }
            None => {
                warn!(
                    event = %context.event.as_str(),
                    action = "mcp_tool",
                    "McpTool hook action configured but no mcp_executor registered"
                );
                hook_execution_message(format!(
                    "McpTool hook {server}::{tool} skipped because no MCP executor is registered"
                ))
            }
        },
        HookAction::Subagent {
            name,
            task,
            timeout,
            ..
        } => {
            let depth = SUBAGENT_HOOK_DEPTH.try_with(|depth| *depth).unwrap_or(0);
            if depth > 0 {
                return hook_execution_message(format!(
                    "Subagent hook '{name}' skipped because a subagent hook is already running"
                ));
            }
            let task_text = task.clone().unwrap_or_default();
            match runtime.subagent_executor {
                Some(executor) => {
                    let fut = executor(name.clone(), task_text.clone());
                    let fut = SUBAGENT_HOOK_DEPTH.scope(depth.saturating_add(1), fut);
                    let result = if *timeout > 0 {
                        match tokio::time::timeout(Duration::from_secs(*timeout), fut).await {
                            Ok(r) => r,
                            Err(_) => {
                                Err(format!("Subagent hook '{name}' timed out after {timeout}s"))
                            }
                        }
                    } else {
                        fut.await
                    };
                    match result {
                        Ok(output) => {
                            let mut hr = HookResult::default();
                            hr.messages.push(format!(
                                "Subagent '{name}' output: {}",
                                truncate_hook_message(&output)
                            ));
                            hr
                        }
                        Err(e) => {
                            warn!(
                                event = %context.event.as_str(),
                                action = "subagent",
                                "Subagent hook failed"
                            );
                            hook_execution_message(format!("Subagent hook '{name}' error: {e}"))
                        }
                    }
                }
                None => {
                    warn!(
                        event = %context.event.as_str(),
                        action = "subagent",
                        "Subagent hook action triggered but no subagent executor is registered"
                    );
                    hook_execution_message(format!(
                        "Subagent hook '{name}' skipped because no subagent executor is registered"
                    ))
                }
            }
        }
        HookAction::ActivateSkill { skill, reason } => {
            HookResult::with_activate_skill(skill.clone(), reason.clone())
        }
    }
}

// -- Command hook execution --

async fn execute_command_hook(
    command: &str,
    shell: Option<&str>,
    timeout_secs: u64,
    source_dir: &str,
    plugin_data_dir: Option<&str>,
    context: &HookContext,
    sandbox: Option<&Arc<dyn SandboxExecutor>>,
) -> HookResult {
    // Build JSON context for stdin (include hook_event_name for compatibility)
    let mut stdin_value = serde_json::to_value(context).unwrap_or_default();
    stdin_value["hook_event_name"] = json!(context.event.as_str());
    let stdin_json = stdin_value;

    let timeout = Duration::from_secs(timeout_secs);

    // -- Sandbox execution path --
    if let Some(manager) = sandbox {
        let (program, args) = build_hook_shell_command(command, shell);
        let mut sandbox_cmd = SandboxCommand::program(&program, args).with_timeout(timeout);

        if !source_dir.is_empty() && Path::new(source_dir).exists() {
            sandbox_cmd = sandbox_cmd.with_working_dir(source_dir);
        }

        // Use minimal environment
        let mut env = minimal_hook_env_with_context(
            source_dir,
            &context.session_id,
            context.event.as_str(),
            &context.cwd,
        );
        add_plugin_hook_env(&mut env, source_dir, plugin_data_dir);
        for (k, v) in env {
            sandbox_cmd = sandbox_cmd.with_env(k, v);
        }

        // Pipe stdin JSON
        if let Ok(json_str) = serde_json::to_string(&stdin_json) {
            sandbox_cmd = sandbox_cmd.with_stdin(json_str);
        }

        return match manager.execute(sandbox_cmd).await {
            Ok(result) => {
                if !result.stderr.is_empty() {
                    debug!(
                        event = %context.event.as_str(),
                        action = "command",
                        stderr_bytes = result.stderr.len(),
                        "Hook stderr (sandboxed)"
                    );
                }
                parse_hook_output(&result.stdout, &result.stderr, result.exit_code)
            }
            Err(_) => {
                warn!(
                    event = %context.event.as_str(),
                    action = "command",
                    sandboxed = true,
                    "Hook sandbox error"
                );
                hook_execution_message("Hook sandbox execution failed".to_string())
            }
        };
    }

    // -- Fallback: direct process execution (no sandbox) --
    let (program, args) = build_hook_shell_command(command, shell);
    let mut cmd = tokio::process::Command::new(&program);
    for arg in &args {
        cmd.arg(arg);
    }
    cmd.kill_on_drop(true);

    if !source_dir.is_empty() && Path::new(source_dir).exists() {
        cmd.current_dir(source_dir);
    }

    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    // Preserve the user's trusted local environment and add stable hook fields.
    let mut env = minimal_hook_env_with_context(
        source_dir,
        &context.session_id,
        context.event.as_str(),
        &context.cwd,
    );
    add_plugin_hook_env(&mut env, source_dir, plugin_data_dir);
    for (k, v) in env {
        cmd.env(k, v);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            warn!(
                event = %context.event.as_str(),
                action = "command",
                error_kind = ?e.kind(),
                "Failed to spawn hook command"
            );
            return hook_execution_message(format!("Failed to spawn hook command: {e}"));
        }
    };

    let mut child = child;
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let json_str = serde_json::to_string(&stdin_json).unwrap_or_default();
        let _ = stdin.write_all(json_str.as_bytes()).await;
        let _ = stdin.write_all(b"\n").await;
        drop(stdin);
    }

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            if !output.stderr.is_empty() {
                debug!(
                    event = %context.event.as_str(),
                    action = "command",
                    stderr_bytes = output.stderr.len(),
                    "Hook stderr"
                );
            }

            parse_hook_output(&stdout, &stderr, output.status.code().unwrap_or(-1))
        }
        Ok(Err(e)) => {
            warn!(
                event = %context.event.as_str(),
                action = "command",
                error_kind = ?e.kind(),
                "Hook command execution error"
            );
            hook_execution_message(format!("Hook command execution failed: {e}"))
        }
        Err(_) => {
            warn!(
                event = %context.event.as_str(),
                action = "command",
                timeout_secs = timeout_secs,
                "Hook command timed out"
            );
            hook_execution_message(format!("Hook command timed out after {timeout_secs}s"))
        }
    }
}

// -- HTTP hook execution --

async fn execute_http_hook(
    url: &str,
    method: Option<&str>,
    headers: Option<&HashMap<String, String>>,
    timeout_secs: u64,
    context: &HookContext,
    client: Option<&reqwest::Client>,
) -> HookResult {
    let method = reqwest::Method::from_bytes(method.unwrap_or("POST").as_bytes())
        .unwrap_or(reqwest::Method::POST);
    let payload = serde_json::to_value(context).unwrap_or_else(|_| json!({}));
    let client = client.cloned().unwrap_or_default();
    let mut request = client
        .request(method, url)
        .timeout(Duration::from_secs(timeout_secs.max(1)))
        .json(&payload);
    if let Some(headers) = headers {
        for (name, value) in headers {
            request = request.header(name, value);
        }
    }
    let result = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            warn!(
                action = "http",
                timeout = error.is_timeout(),
                connect = error.is_connect(),
                "Http hook request failed"
            );
            return hook_execution_message(format!("Http hook request failed: {error}"));
        }
    };

    match result.status().is_success() {
        true => {
            let text = result.text().await.unwrap_or_default();
            parse_hook_output(&text, "", 0)
        }
        false => {
            warn!(status = %result.status(), "Http hook non-2xx response");
            hook_execution_message(format!(
                "Http hook returned non-success status {}",
                result.status()
            ))
        }
    }
}

// ── Output Parsing ─────────────────────────────────────────────────────

/// Parse JSON output from a hook command.
///
/// Hooks can return JSON to control execution:
/// ```json
/// {
///   "decision": "block",
///   "reason": "Unsafe command detected",
///   "continue": false,
///   "permission_decision": "allow" | "deny" | "ask",
///   "permission_suggestions": ["Allow", "Deny"],
///   "continue_reason": "Check if tests pass",
///   "injected_context": "Remember to use bun",
///   "metadata": {}
/// }
/// ```
fn parse_hook_output(stdout: &str, stderr: &str, exit_code: i32) -> HookResult {
    let mut result = HookResult::default();

    // Exit code semantics (aligned with Claude Code convention):
    //  exit 0: pass (no block intent)
    //  exit 1: no block (hook produced output but no explicit block)
    //  exit 2: block (explicit block signal)
    //  other non-zero: warning, no block
    match exit_code {
        0 | 1 => {} // no implicit block
        2 => {
            result.block = true;
            let reason = stderr.trim();
            result.block_reason = Some(if reason.is_empty() {
                "Hook exited with code 2 (explicit block)".to_string()
            } else {
                truncate_hook_message(reason)
            });
        }
        _ => {
            // Other non-zero: log warning, don't block
            warn!(
                exit_code = exit_code,
                "Hook exited with unexpected code, treating as warning (not block)"
            );
            let detail = stderr.trim();
            result.messages.push(if detail.is_empty() {
                format!("Hook exited with code {exit_code}")
            } else {
                format!(
                    "Hook exited with code {exit_code}: {}",
                    truncate_hook_message(detail)
                )
            });
        }
    }

    // Try to parse JSON output
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return result;
    }

    if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
        let hook_specific = json.get("hookSpecificOutput").unwrap_or(&json);
        if let Some(message) = json.get("systemMessage").and_then(Value::as_str) {
            result.messages.push(truncate_hook_message(message));
        }

        // Parse decision field
        if let Some(decision) = json.get("decision").and_then(|v| v.as_str()) {
            match decision {
                "block" => {
                    result.block = true;
                    if let Some(reason) = json.get("reason").and_then(|v| v.as_str()) {
                        result.block_reason = Some(truncate_hook_message(reason));
                    }
                }
                "allow" => {
                    result.block = false;
                    result.block_reason = None;
                }
                _ => {}
            }
        }

        // Portable Codex/Claude hook output fields. embedding application keeps its native flat
        // fields below, while accepting these nested fields for plugin reuse.
        if let Some(decision) = hook_specific
            .get("permissionDecision")
            .and_then(Value::as_str)
        {
            match decision {
                "allow" => result.permission_decision = Some(PermissionDecision::Allow),
                "deny" => {
                    let reason = hook_specific
                        .get("permissionDecisionReason")
                        .and_then(Value::as_str)
                        .map(truncate_hook_message)
                        .unwrap_or_else(|| "Hook denied permission".to_string());
                    result.permission_decision = Some(PermissionDecision::Deny { reason });
                }
                "ask" => {
                    result.permission_decision = Some(PermissionDecision::Ask {
                        suggestions: Vec::new(),
                    });
                }
                _ => {}
            }
        }
        if let Some(decision) = hook_specific.get("decision").and_then(Value::as_object)
            && let Some(behavior) = decision.get("behavior").and_then(Value::as_str)
        {
            match behavior {
                "allow" => result.permission_decision = Some(PermissionDecision::Allow),
                "deny" => {
                    let reason = decision
                        .get("message")
                        .and_then(Value::as_str)
                        .map(truncate_hook_message)
                        .unwrap_or_else(|| "Hook denied permission".to_string());
                    result.permission_decision = Some(PermissionDecision::Deny { reason });
                }
                _ => {}
            }
        }

        // Parse permission_decision field
        if let Some(perm_decision) = json.get("permission_decision").and_then(|v| v.as_str()) {
            match perm_decision {
                "allow" => {
                    result.permission_decision = Some(PermissionDecision::Allow);
                }
                "deny" => {
                    let reason = json
                        .get("permission_reason")
                        .and_then(|v| v.as_str())
                        .map(truncate_hook_message)
                        .unwrap_or_else(|| "Hook denied permission".to_string());
                    result.permission_decision = Some(PermissionDecision::Deny { reason });
                }
                "ask" => {
                    let suggestions = json
                        .get("permission_suggestions")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    result.permission_decision = Some(PermissionDecision::Ask { suggestions });
                }
                _ => {}
            }
        }

        // Parse the canonical call-scoped permission mode override. Hooks are
        // user-installed local extensions; shared/managed deployments can still
        // disable BypassPermissions through PermissionService::bypass_disabled.
        if let Some(mode) = json
            .get("permission_mode_override")
            .and_then(|v| v.as_str())
        {
            result.permission_mode_override = match mode {
                "default" => Some(PermissionMode::Default),
                "plan" => Some(PermissionMode::Plan),
                "auto" => Some(PermissionMode::Auto),
                "acceptEdits" => Some(PermissionMode::AcceptEdits),
                "bypassPermissions" => Some(PermissionMode::BypassPermissions),
                "bubble" => Some(PermissionMode::Bubble),
                "dontAsk" => Some(PermissionMode::DontAsk),
                "strict" => Some(PermissionMode::StrictConfirm),
                _ => None,
            };
        }

        if json.get("continue") == Some(&Value::Bool(false)) {
            result.stop_propagation = true;
        }

        if let Some(updated) = hook_specific
            .get("updatedInput")
            .or_else(|| json.get("updatedInput"))
        {
            result.updated_input = Some(updated.clone());
        }

        // Parse lifecycle-specific fields
        if let Some(reason) = json.get("continue_reason").and_then(|v| v.as_str()) {
            result.continue_reason = Some(truncate_hook_message(reason));
        }

        if let Some(ctx) = hook_specific
            .get("additionalContext")
            .or_else(|| json.get("injected_context"))
            .and_then(Value::as_str)
        {
            result.injected_context = Some(truncate_hook_message(ctx));
        }

        // Parse retry field (PermissionDenied hooks)
        if json.get("retry") == Some(&Value::Bool(true)) {
            result.retry = true;
        }

        if let Some(meta) = json.get("metadata")
            && !meta.is_null()
        {
            result.metadata = Some(meta.clone());
        }
    } else if exit_code == 0 {
        // Non-JSON stdout on exit 0: treat as injected context
        result.injected_context = Some(truncate_hook_message(trimmed));
    }

    // Exit code 2 is an out-of-band blocking signal. Structured stdout may add
    // context, but it must not downgrade that process-level decision.
    if exit_code == 2 {
        result.block = true;
        let reason = stderr.trim();
        result.block_reason = Some(if reason.is_empty() {
            "Hook exited with code 2 (explicit block)".to_string()
        } else {
            truncate_hook_message(reason)
        });
    }

    result
}

fn add_plugin_hook_env(
    env: &mut HashMap<String, String>,
    source_dir: &str,
    plugin_data_dir: Option<&str>,
) {
    let Some(plugin_data_dir) = plugin_data_dir else {
        return;
    };
    for key in ["PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT", "ECHO_PLUGIN_ROOT"] {
        env.insert(key.to_string(), source_dir.to_string());
    }
    for key in ["PLUGIN_DATA", "CLAUDE_PLUGIN_DATA", "ECHO_PLUGIN_DATA"] {
        env.insert(key.to_string(), plugin_data_dir.to_string());
    }
}

fn hook_execution_message(message: String) -> HookResult {
    let mut result = HookResult::default();
    result.messages.push(truncate_hook_message(&message));
    result
}

fn truncate_hook_message(message: &str) -> String {
    const MAX_CHARS: usize = 10_000;
    let mut text = message.chars().take(MAX_CHARS).collect::<String>();
    if message.chars().count() > MAX_CHARS {
        text.push_str("...");
    }
    text
}

// ── Shell Command Builder ──────────────────────────────────────────────

fn build_hook_shell_command(command: &str, shell: Option<&str>) -> (String, Vec<String>) {
    let shell_type = shell.unwrap_or("bash");

    if shell_type == "powershell" {
        let program = if which_exists("pwsh") {
            "pwsh"
        } else if cfg!(target_os = "windows") {
            "powershell"
        } else {
            "sh"
        };

        if program == "pwsh" || program == "powershell" {
            (
                program.to_string(),
                vec![
                    "-NoProfile".to_string(),
                    "-NonInteractive".to_string(),
                    "-Command".to_string(),
                    command.to_string(),
                ],
            )
        } else {
            (
                "sh".to_string(),
                vec!["-c".to_string(), command.to_string()],
            )
        }
    } else if cfg!(target_os = "windows") {
        #[cfg(target_os = "windows")]
        {
            if let Some(bash) = crate::skills::external::prompt_exec::find_git_bash_path() {
                (bash, vec!["-c".to_string(), command.to_string()])
            } else {
                (
                    "cmd".to_string(),
                    vec!["/C".to_string(), command.to_string()],
                )
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            (
                "bash".to_string(),
                vec!["-c".to_string(), command.to_string()],
            )
        }
    } else {
        (
            "bash".to_string(),
            vec!["-c".to_string(), command.to_string()],
        )
    }
}

fn which_exists(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

// ── Result Merging ─────────────────────────────────────────────────────

fn merge_result(combined: &mut HookResult, incoming: HookResult) {
    let carries_permission_decision = incoming.permission_decision.is_some();
    if incoming.block {
        combined.block = true;
        combined.block_reason = incoming.block_reason.or(combined.block_reason.take());
    }
    if incoming.updated_input.is_some() {
        combined.updated_input = incoming.updated_input;
    }
    combined.messages.extend(incoming.messages);
    // An external command/HTTP/programmatic hook may return both a permission
    // decision and `continue: false`. Permission safety is reduced across all
    // matching sources, so allow/ask/require-approval cannot use that stop bit
    // to hide a later deny. Results without a permission decision retain the
    // ordinary stop-propagation contract.
    if incoming.stop_propagation && !carries_permission_decision {
        combined.stop_propagation = true;
    }

    // Permission decision with priority: deny > ask > require_approval > allow
    if let Some(new_decision) = incoming.permission_decision {
        let should_replace = match (&combined.permission_decision, &new_decision) {
            // If we already have deny, keep it
            (Some(PermissionDecision::Deny { .. }), _) => false,
            // New decision is deny -- always take it
            (_, PermissionDecision::Deny { .. }) => true,
            // If we already have ask, keep it (ask > allow)
            (Some(PermissionDecision::Ask { .. }), _) => false,
            // New decision is ask -- take it over allow
            (_, PermissionDecision::Ask { .. }) => true,
            // If we already have RequireApproval, keep it
            (Some(PermissionDecision::RequireApproval), _) => false,
            // New decision is RequireApproval -- take it over allow
            (_, PermissionDecision::RequireApproval) => true,
            // Both are allow -- either is fine
            (Some(PermissionDecision::Allow), PermissionDecision::Allow) => false,
            // No existing decision -- take the new one
            (None, _) => true,
        };
        if should_replace {
            combined.permission_decision = Some(new_decision);
        }
    }

    // permission_mode_override: last non-none wins
    if incoming.permission_mode_override.is_some() {
        combined.permission_mode_override = incoming.permission_mode_override;
    }

    // continue_reason: non-None overrides None
    if incoming.continue_reason.is_some() {
        combined.continue_reason = incoming.continue_reason;
    }

    // injected_context: concatenate with newline
    if let Some(ctx) = incoming.injected_context {
        combined.injected_context = Some(match combined.injected_context.take() {
            Some(existing) => format!("{}\n{}", existing, ctx),
            None => ctx,
        });
    }

    // retry: OR semantics (any true → combined true)
    if incoming.retry {
        combined.retry = true;
    }

    // activate_skill: first-wins (the first non-None activation request is kept;
    // subsequent activate_skill results from other hooks are ignored to avoid
    // conflicting multi-skill activation).
    if combined.activate_skill.is_none() {
        combined.activate_skill = incoming.activate_skill;
    }

    // metadata: deep merge
    if let Some(meta) = incoming.metadata {
        combined.metadata = Some(match combined.metadata.take() {
            Some(existing) => {
                let mut merged = existing;
                if let (Value::Object(a), Value::Object(b)) = (&mut merged, &meta) {
                    for (k, v) in b {
                        a.insert(k.clone(), v.clone());
                    }
                }
                merged
            }
            None => meta,
        });
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- HookEvent tests --

    #[test]
    fn test_hook_event_is_tool_event() {
        assert!(HookEvent::PreToolUse.is_tool_event());
        assert!(HookEvent::PostToolUse.is_tool_event());
        assert!(HookEvent::PostToolUseFailure.is_tool_event());
        assert!(HookEvent::PermissionRequest.is_tool_event());
        assert!(HookEvent::PermissionDenied.is_tool_event());
        assert!(!HookEvent::SessionStart.is_tool_event());
        assert!(!HookEvent::Stop.is_tool_event());
        assert!(!HookEvent::Notification.is_tool_event());
        assert!(!HookEvent::StopFailure.is_tool_event());
    }

    #[test]
    fn test_hook_event_supports_matcher() {
        assert!(HookEvent::PreToolUse.supports_matcher());
        assert!(HookEvent::SessionStart.supports_matcher());
        assert!(HookEvent::Notification.supports_matcher());
        assert!(HookEvent::SubagentStart.supports_matcher());
        assert!(HookEvent::TaskCreated.supports_matcher());
        assert!(HookEvent::PostToolBatch.supports_matcher());
        assert!(HookEvent::UserPromptSubmit.supports_matcher());
        assert!(!HookEvent::StopFailure.supports_matcher());
    }

    #[test]
    fn test_hook_event_serde() {
        let event = HookEvent::PreToolUse;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, "\"PreToolUse\"");

        let event = HookEvent::SessionStart;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, "\"SessionStart\"");

        let parsed: HookEvent = serde_json::from_str("\"PostToolUseFailure\"").unwrap();
        assert_eq!(parsed, HookEvent::PostToolUseFailure);
    }

    // -- HookContext tests --

    #[test]
    fn test_hook_context_for_pre_tool_use() {
        let ctx =
            HookContext::for_pre_tool_use("Bash", &json!({"command": "ls"}), "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::PreToolUse);
        assert_eq!(ctx.tool_name.as_deref(), Some("Bash"));
        assert_eq!(ctx.session_id, "sess-1");
        assert!(ctx.tool_output.is_none());
        assert!(ctx.matcher.is_none());
    }

    #[test]
    fn test_hook_context_for_session_start() {
        let ctx = HookContext::for_session_start("startup", "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::SessionStart);
        assert_eq!(ctx.matcher.as_deref(), Some("startup"));
        assert!(ctx.tool_name.is_none());
    }

    #[test]
    fn test_hook_context_for_stop() {
        let ctx = HookContext::for_stop(None, "sess-1", "agent", false);
        assert_eq!(ctx.event, HookEvent::Stop);
        assert!(ctx.matcher.is_none());
        assert!(ctx.tool_name.is_none());
    }

    #[test]
    fn test_hook_context_for_notification() {
        let ctx = HookContext::for_notification("permission_prompt", "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::Notification);
        assert_eq!(ctx.matcher.as_deref(), Some("permission_prompt"));
    }

    #[test]
    fn test_hook_context_serialization() {
        let ctx =
            HookContext::for_pre_tool_use("Bash", &json!({"command": "ls"}), "sess-1", "agent");
        let json_str = serde_json::to_string(&ctx).unwrap();
        assert!(json_str.contains("\"event\":\"PreToolUse\""));
        assert!(json_str.contains("\"tool_name\":\"Bash\""));
        // None fields should be omitted
        assert!(!json_str.contains("\"tool_output\""));
        assert!(!json_str.contains("\"matcher\""));
    }

    // -- matches_hook tests --

    #[test]
    fn test_matches_hook_wildcard() {
        let ctx = HookContext::for_pre_tool_use("Bash", &json!({}), "", "");
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
    }

    #[test]
    fn test_matches_hook_tool_name_exact() {
        let ctx = HookContext::for_pre_tool_use("Bash", &json!({}), "", "");
        assert!(matches_hook("Bash", &ctx));
        assert!(!matches_hook("Read", &ctx));
    }

    #[test]
    fn test_matches_hook_tool_name_pipe_separated() {
        let ctx = HookContext::for_pre_tool_use("Edit", &json!({}), "", "");
        assert!(matches_hook("Edit|Write", &ctx));
        assert!(matches_hook("Write|Edit", &ctx));
        assert!(!matches_hook("Bash|Read", &ctx));
    }

    #[test]
    fn test_matches_hook_tool_name_prefix() {
        let ctx = HookContext::for_pre_tool_use("Bash(git:*)", &json!({}), "", "");
        assert!(matches_hook("Bash", &ctx));
    }

    #[test]
    fn test_matches_hook_lifecycle_matcher() {
        let ctx = HookContext::for_session_start("startup", "", "");
        assert!(matches_hook("startup", &ctx));
        assert!(!matches_hook("resume", &ctx));
    }

    #[test]
    fn test_matches_hook_lifecycle_pipe_separated() {
        let ctx = HookContext::for_session_start("resume", "", "");
        assert!(matches_hook("startup|resume", &ctx));
    }

    #[test]
    fn test_matches_hook_lifecycle_glob() {
        let ctx = HookContext::for_notification("permission_prompt", "", "");
        assert!(matches_hook("permission*", &ctx));
    }

    #[test]
    fn test_matches_hook_no_matcher_event() {
        let ctx = HookContext::for_stop(None, "", "", false);
        // Stop doesn't support matcher, so only "*" and "" match
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
        assert!(!matches_hook("something", &ctx));
    }

    #[test]
    fn test_matches_hook_subagent_start() {
        let ctx = HookContext::for_subagent_start("coder", "sync", "implement X", "", "");
        assert!(matches_hook("coder", &ctx));
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
        assert!(!matches_hook("planner", &ctx));
        // Pipe-separated alternatives
        assert!(matches_hook("planner|coder|reviewer", &ctx));
    }

    #[test]
    fn test_matches_hook_subagent_stop() {
        let ctx = HookContext::for_subagent_stop(
            "coder",
            "sync",
            "success",
            echo_core::hooks::SubagentStopStatus::Completed,
            "",
            "",
        );
        assert!(matches_hook("coder", &ctx));
        assert!(!matches_hook("planner", &ctx));
    }

    #[test]
    fn test_matches_hook_task_created() {
        let ctx = HookContext::for_task_created("t-1", "build API", "", "");
        assert!(matches_hook("build API", &ctx));
        assert!(matches_hook("*", &ctx));
        assert!(!matches_hook("deploy", &ctx));
    }

    #[test]
    fn test_matches_hook_task_completed() {
        let ctx = HookContext::for_task_completed(
            "t-1",
            "build API",
            "success",
            echo_core::hooks::TaskTerminalStatus::Completed,
            "",
            "",
        );
        assert!(matches_hook("build API", &ctx));
        assert!(!matches_hook("deploy", &ctx));
    }

    #[test]
    fn test_matches_hook_subagent_glob() {
        let ctx = HookContext::for_subagent_start("code-reviewer", "sync", "review", "", "");
        assert!(matches_hook("code*", &ctx));
        assert!(!matches_hook("test*", &ctx));
    }

    // -- HooksDefinition tests --

    #[test]
    fn test_hooks_definition_empty() {
        let def = HooksDefinition::default();
        assert!(def.is_empty());
    }

    #[test]
    fn test_hooks_definition_add_rules() {
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "test".into(),
                }],
            }],
        );
        assert!(!def.is_empty());
        assert_eq!(def.rules_for(HookEvent::PreToolUse).len(), 1);
        assert_eq!(def.rules_for(HookEvent::PostToolUse).len(), 0);
    }

    #[test]
    fn test_register_plugin_hooks_distinct_source() {
        // P0-2: plugin hooks must register under HookSource::Plugin, not be
        // collapsed into HookSource::Skill. Verify the source identity shows up
        // distinctly in list_sources.
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "audit plugin".into(),
                }],
            }],
        );

        registry.register_plugin_hooks(
            "data-analysis-pack",
            "/plugins/data",
            "/plugin-state/data-analysis-pack",
            def,
        );

        let sources = registry.list_sources();
        // Should appear as plugin:data-analysis-pack, never as skill:plugin:...
        assert!(
            sources
                .iter()
                .any(|(name, _)| name == "plugin:data-analysis-pack"),
            "plugin hook source must be plugin-prefixed, got {sources:?}"
        );
        assert!(
            !sources
                .iter()
                .any(|(name, _)| name.starts_with("skill:plugin:")),
            "plugin hook must not be filed as skill source, got {sources:?}"
        );
    }

    #[test]
    fn test_register_plugin_hooks_drops_invalid_actions() {
        // P0-2: an invalid action is dropped, valid siblings survive, and the
        // plugin is still registered under the plugin source.
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PostToolUse,
            vec![HookRule {
                matcher: "*".into(),
                hooks: vec![
                    // valid
                    HookAction::Prompt {
                        prompt: "keep".into(),
                    },
                    // invalid: empty command
                    HookAction::Command {
                        command: String::new(),
                        shell: None,
                        timeout: 10,
                    },
                ],
            }],
        );

        registry.register_plugin_hooks("mixed-plugin", "/p", "/data/p", def);
        let sources = registry.list_sources();
        assert!(
            sources
                .iter()
                .any(|(name, count)| name == "plugin:mixed-plugin" && *count == 1),
            "valid sibling rule should survive (1 rule), got {sources:?}"
        );
    }

    #[test]
    fn test_register_plugin_hooks_empty_noop() {
        let mut registry = HookRegistry::new();
        registry.register_plugin_hooks("empty-plugin", "/p", "/data/p", HooksDefinition::default());
        assert!(registry.is_empty(), "empty definition must not register");
    }

    #[test]
    fn test_hooks_definition_merge() {
        let mut def1 = HooksDefinition::default();
        def1.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-1".into(),
                }],
            }],
        );

        let mut def2 = HooksDefinition::default();
        def2.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Read".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-2".into(),
                }],
            }],
        );
        def2.add_rules(
            HookEvent::SessionStart,
            vec![HookRule {
                matcher: "startup".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "welcome".into(),
                }],
            }],
        );

        def1.merge(def2);
        assert_eq!(def1.rules_for(HookEvent::PreToolUse).len(), 2);
        assert_eq!(def1.rules_for(HookEvent::SessionStart).len(), 1);
    }

    #[test]
    fn test_hooks_definition_deserialize_yaml() {
        let yaml = r#"
PreToolUse:
  - matcher: "Bash"
    hooks:
      - type: command
        command: "echo test"
        timeout: 5
      - type: prompt
        prompt: "Be careful"
PostToolUse:
  - matcher: "*"
    hooks:
      - type: command
        command: "echo done"
SessionStart:
  - matcher: "startup"
    hooks:
      - type: prompt
        prompt: "Welcome!"
"#;
        let def: HooksDefinition = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(!def.is_empty());
        assert_eq!(def.rules_for(HookEvent::PreToolUse).len(), 1);
        assert_eq!(def.rules_for(HookEvent::PreToolUse)[0].hooks.len(), 2);
        assert_eq!(def.rules_for(HookEvent::PostToolUse).len(), 1);
        assert_eq!(def.rules_for(HookEvent::SessionStart).len(), 1);
    }

    #[test]
    fn test_hooks_definition_deserialize_with_http_action() {
        let yaml = r#"
PostToolUse:
  - matcher: "Bash"
    hooks:
      - type: http
        url: "https://audit.example.com/tool-usage"
        timeout: 3
"#;
        let def: HooksDefinition = serde_yaml_ng::from_str(yaml).unwrap();
        let rules = def.rules_for(HookEvent::PostToolUse);
        assert_eq!(rules.len(), 1);
        assert!(
            matches!(&rules[0].hooks[0], HookAction::Http { url, .. } if url == "https://audit.example.com/tool-usage")
        );
    }

    #[test]
    fn test_hooks_definition_deserialize_with_mcp_tool_action() {
        let yaml = r#"
Notification:
  - matcher: "permission_prompt"
    hooks:
      - type: mcp_tool
        server: "slack"
        tool: "send_message"
        arguments:
          channel: "agent-approvals"
"#;
        let def: HooksDefinition = serde_yaml_ng::from_str(yaml).unwrap();
        let rules = def.rules_for(HookEvent::Notification);
        assert_eq!(rules.len(), 1);
        assert!(
            matches!(&rules[0].hooks[0], HookAction::McpTool { server, .. } if server == "slack")
        );
    }

    // -- HookResult tests --

    #[test]
    fn test_hook_result_allow() {
        let result = HookResult::allow();
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_allowed());
    }

    #[test]
    fn test_hook_result_deny() {
        let result = HookResult::deny("test reason".to_string());
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_denied());
        assert!(result.block);
    }

    #[test]
    fn test_hook_result_ask() {
        let result = HookResult::ask(vec!["Option A".to_string()]);
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().requires_approval());
    }

    #[test]
    fn test_hook_result_should_continue() {
        let mut result = HookResult::default();
        assert!(!result.should_continue());
        result.continue_reason = Some("Check tests".to_string());
        assert!(result.should_continue());
    }

    // -- parse_hook_output tests --

    #[test]
    fn test_parse_hook_output_empty() {
        let result = parse_hook_output("", "", 0);
        assert!(!result.block);
    }

    #[test]
    fn test_parse_hook_output_block() {
        let result = parse_hook_output(r#"{"decision": "block", "reason": "unsafe"}"#, "", 0);
        assert!(result.block);
        assert_eq!(result.block_reason, Some("unsafe".into()));
    }

    #[test]
    fn test_parse_hook_output_allow() {
        let result = parse_hook_output(r#"{"decision": "allow"}"#, "", 1);
        assert!(!result.block);
    }

    #[test]
    fn test_parse_hook_output_nonzero_exit_code_1_no_block() {
        // exit 1 = no block (hook produced output but no block intent)
        let result = parse_hook_output("", "", 1);
        assert!(!result.block);
    }

    #[test]
    fn test_parse_hook_output_exit_code_2_block() {
        // exit 2 = explicit block signal
        let result = parse_hook_output("", "", 2);
        assert!(result.block);
        assert_eq!(
            result.block_reason,
            Some("Hook exited with code 2 (explicit block)".to_string())
        );
    }

    #[test]
    fn test_parse_hook_output_exit_code_2_uses_stderr_reason() {
        let result = parse_hook_output("", "Blocked by policy", 2);
        assert!(result.block);
        assert_eq!(result.block_reason.as_deref(), Some("Blocked by policy"));
    }

    #[test]
    fn test_parse_hook_output_exit_code_2_cannot_be_downgraded() {
        let result = parse_hook_output(
            r#"{"decision": "allow", "reason": "ignore"}"#,
            "Blocked by process",
            2,
        );
        assert!(result.block);
        assert_eq!(result.block_reason.as_deref(), Some("Blocked by process"));
    }

    #[test]
    fn test_parse_hook_output_exit_code_other_no_block() {
        // Other non-zero = warning, no block
        let result = parse_hook_output("", "", 3);
        assert!(!result.block);
        let result = parse_hook_output("", "", 127);
        assert!(!result.block);
    }

    #[test]
    fn test_parse_hook_output_retry_field() {
        let result = parse_hook_output(r#"{"retry": true}"#, "", 0);
        assert!(result.retry);
    }

    #[test]
    fn test_parse_hook_output_retry_false_by_default() {
        let result = parse_hook_output("", "", 0);
        assert!(!result.retry);
    }

    #[test]
    fn test_parse_hook_output_updated_input() {
        let result = parse_hook_output(r#"{"updatedInput": {"command": "safe-command"}}"#, "", 0);
        assert!(!result.block);
        assert_eq!(
            result.updated_input,
            Some(json!({"command": "safe-command"}))
        );
    }

    #[test]
    fn test_parse_hook_output_accepts_portable_nested_fields() {
        let result = parse_hook_output(
            r#"{"systemMessage":"audit","hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{"command":"echo safe"},"additionalContext":"reviewed"}}"#,
            "",
            0,
        );
        assert!(
            result
                .permission_decision
                .as_ref()
                .is_some_and(PermissionDecision::is_allowed)
        );
        assert_eq!(result.updated_input, Some(json!({"command": "echo safe"})));
        assert_eq!(result.injected_context.as_deref(), Some("reviewed"));
        assert_eq!(result.messages, vec!["audit"]);
    }

    #[test]
    fn test_parse_hook_output_accepts_permission_request_decision() {
        let result = parse_hook_output(
            r#"{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"deny","message":"Blocked by repository policy"}}}"#,
            "",
            0,
        );
        assert!(
            result
                .permission_decision
                .as_ref()
                .is_some_and(PermissionDecision::is_denied)
        );
    }

    #[test]
    fn test_parse_hook_output_continue_reason() {
        let result =
            parse_hook_output(r#"{"continue_reason": "Run tests before stopping"}"#, "", 0);
        assert_eq!(
            result.continue_reason,
            Some("Run tests before stopping".to_string())
        );
    }

    #[test]
    fn test_parse_hook_output_injected_context() {
        let result = parse_hook_output(r#"{"injected_context": "Remember to use bun"}"#, "", 0);
        assert_eq!(
            result.injected_context,
            Some("Remember to use bun".to_string())
        );
    }

    #[test]
    fn test_parse_hook_output_non_json_as_context() {
        let result = parse_hook_output("Remember: use bun, not npm", "", 0);
        assert_eq!(
            result.injected_context,
            Some("Remember: use bun, not npm".to_string())
        );
    }

    #[test]
    fn test_parse_hook_output_permission_decision_allow() {
        let result = parse_hook_output(r#"{"permission_decision": "allow"}"#, "", 0);
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_allowed());
    }

    #[test]
    fn test_parse_hook_output_permission_decision_deny() {
        let result = parse_hook_output(
            r#"{"permission_decision": "deny", "permission_reason": "unsafe"}"#,
            "",
            0,
        );
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_denied());
    }

    #[test]
    fn test_parse_hook_output_permission_mode_override() {
        for (wire_value, expected) in [
            ("default", PermissionMode::Default),
            ("plan", PermissionMode::Plan),
            ("auto", PermissionMode::Auto),
            ("acceptEdits", PermissionMode::AcceptEdits),
            ("bypassPermissions", PermissionMode::BypassPermissions),
            ("bubble", PermissionMode::Bubble),
            ("dontAsk", PermissionMode::DontAsk),
            ("strict", PermissionMode::StrictConfirm),
        ] {
            let output = format!(r#"{{"permission_mode_override":"{wire_value}"}}"#);
            let result = parse_hook_output(&output, "", 0);
            assert_eq!(result.permission_mode_override, Some(expected));
        }
    }

    #[test]
    fn test_parse_hook_output_ignores_legacy_permission_mode() {
        let result = parse_hook_output(r#"{"permission_mode": "auto"}"#, "", 0);
        assert_eq!(result.permission_mode_override, None);
    }

    // -- merge_result tests --

    #[test]
    fn test_merge_result_permission_priority_deny_wins() {
        let mut combined = HookResult {
            permission_decision: Some(PermissionDecision::Allow),
            ..HookResult::default()
        };

        let incoming = HookResult {
            permission_decision: Some(PermissionDecision::Ask {
                suggestions: vec!["Allow".to_string()],
            }),
            ..HookResult::default()
        };
        merge_result(&mut combined, incoming);
        assert!(matches!(
            combined.permission_decision.clone().unwrap(),
            PermissionDecision::Ask { .. }
        ));

        let incoming2 = HookResult {
            permission_decision: Some(PermissionDecision::Deny {
                reason: "unsafe".to_string(),
            }),
            ..HookResult::default()
        };
        merge_result(&mut combined, incoming2);
        assert!(matches!(
            combined.permission_decision.clone().unwrap(),
            PermissionDecision::Deny { .. }
        ));

        let incoming3 = HookResult {
            permission_decision: Some(PermissionDecision::Allow),
            ..HookResult::default()
        };
        merge_result(&mut combined, incoming3);
        assert!(matches!(
            combined.permission_decision.clone().unwrap(),
            PermissionDecision::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn permission_hooks_reduce_across_sources_before_allow_or_ask_short_circuit() {
        let mut registry = HookRegistry::new();

        let permission_rule = |decision: &str| HookRule {
            matcher: "Bash".to_string(),
            hooks: vec![HookAction::Permission {
                decision: decision.to_string(),
                reason: Some(format!("{decision} decision")),
                suggestions: vec!["allow".to_string()],
            }],
        };

        let mut user = HooksDefinition::default();
        user.add_rules(HookEvent::PreToolUse, vec![permission_rule("allow")]);
        registry.register_user_hooks(user);

        let mut plugin = HooksDefinition::default();
        plugin.add_rules(HookEvent::PreToolUse, vec![permission_rule("ask")]);
        assert!(registry.register_plugin_hooks("policy", "/tmp/policy", "/tmp/data", plugin));

        let mut skill = HooksDefinition::default();
        skill.add_rules(HookEvent::PreToolUse, vec![permission_rule("deny")]);
        registry.register("guard", "/tmp/guard", skill);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "git status"}), "session")
            .await;

        assert!(matches!(
            result.permission_decision,
            Some(PermissionDecision::Deny { ref reason }) if reason == "deny decision"
        ));
        assert!(result.block);
        assert_eq!(result.block_reason.as_deref(), Some("deny decision"));
    }

    #[test]
    fn permission_output_stop_cannot_hide_later_deny() {
        let mut combined = HookResult::default();
        for decision in ["allow", "ask"] {
            let incoming = parse_hook_output(
                &format!(r#"{{"permission_decision":"{decision}","continue":false}}"#),
                "",
                0,
            );
            assert!(incoming.stop_propagation);
            merge_result(&mut combined, incoming);
            assert!(!combined.stop_propagation);
        }
        merge_result(&mut combined, HookResult::deny("later deny".to_string()));
        assert!(matches!(
            combined.permission_decision,
            Some(PermissionDecision::Deny { ref reason }) if reason == "later deny"
        ));

        let mut non_permission = HookResult::default();
        merge_result(
            &mut non_permission,
            HookResult {
                stop_propagation: true,
                ..HookResult::default()
            },
        );
        assert!(non_permission.stop_propagation);
    }

    #[tokio::test]
    async fn command_permission_stop_cannot_hide_later_source_deny() {
        #[cfg(target_os = "windows")]
        let command = |decision: &str| {
            format!(r#"echo {{"permission_decision":"{decision}","continue":false}}"#)
        };
        #[cfg(not(target_os = "windows"))]
        let command = |decision: &str| {
            format!(r#"printf '%s' '{{"permission_decision":"{decision}","continue":false}}'"#)
        };

        let mut registry = HookRegistry::new();
        let mut user = HooksDefinition::default();
        user.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".to_string(),
                hooks: vec![
                    HookAction::Command {
                        command: command("allow"),
                        shell: None,
                        timeout: 5,
                    },
                    HookAction::Command {
                        command: command("ask"),
                        shell: None,
                        timeout: 5,
                    },
                ],
            }],
        );
        registry.register_user_hooks(user);

        let mut skill = HooksDefinition::default();
        skill.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".to_string(),
                hooks: vec![HookAction::Permission {
                    decision: "deny".to_string(),
                    reason: Some("later source deny".to_string()),
                    suggestions: Vec::new(),
                }],
            }],
        );
        registry.register("guard", "/tmp/guard", skill);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "git status"}), "session")
            .await;
        assert!(matches!(
            result.permission_decision,
            Some(PermissionDecision::Deny { ref reason }) if reason == "later source deny"
        ));
        assert!(result.block);
    }

    #[test]
    fn test_merge_result_continue_reason() {
        let mut combined = HookResult::default();
        merge_result(
            &mut combined,
            HookResult {
                continue_reason: Some("Check tests".to_string()),
                ..HookResult::default()
            },
        );
        assert_eq!(combined.continue_reason, Some("Check tests".to_string()));

        // Later non-None overrides
        merge_result(
            &mut combined,
            HookResult {
                continue_reason: Some("Different reason".to_string()),
                ..HookResult::default()
            },
        );
        assert_eq!(
            combined.continue_reason,
            Some("Different reason".to_string())
        );
    }

    #[test]
    fn test_merge_result_injected_context_concatenates() {
        let mut combined = HookResult::default();
        merge_result(
            &mut combined,
            HookResult {
                injected_context: Some("First".to_string()),
                ..HookResult::default()
            },
        );
        assert_eq!(combined.injected_context, Some("First".to_string()));

        merge_result(
            &mut combined,
            HookResult {
                injected_context: Some("Second".to_string()),
                ..HookResult::default()
            },
        );
        assert_eq!(combined.injected_context, Some("First\nSecond".to_string()));
    }

    #[test]
    fn test_merge_result_metadata_deep_merge() {
        let mut combined = HookResult {
            metadata: Some(json!({"a": 1, "b": 2})),
            ..HookResult::default()
        };
        merge_result(
            &mut combined,
            HookResult {
                metadata: Some(json!({"b": 3, "c": 4})),
                ..HookResult::default()
            },
        );
        let meta = combined.metadata.unwrap();
        assert_eq!(meta["a"], 1);
        assert_eq!(meta["b"], 3); // overwritten
        assert_eq!(meta["c"], 4);
    }

    #[test]
    fn test_merge_result_permission_mode_override_last_wins() {
        let mut combined = HookResult {
            permission_mode_override: Some(PermissionMode::Auto),
            ..HookResult::default()
        };
        merge_result(
            &mut combined,
            HookResult {
                permission_mode_override: Some(PermissionMode::Plan),
                ..HookResult::default()
            },
        );
        assert_eq!(
            combined.permission_mode_override,
            Some(PermissionMode::Plan)
        );

        merge_result(
            &mut combined,
            HookResult {
                permission_mode_override: None,
                ..HookResult::default()
            },
        );
        assert_eq!(
            combined.permission_mode_override,
            Some(PermissionMode::Plan)
        );
    }

    #[test]
    fn test_merge_result_retry_or_semantics() {
        let mut combined = HookResult::default();
        assert!(!combined.retry);

        merge_result(
            &mut combined,
            HookResult {
                retry: true,
                ..HookResult::default()
            },
        );
        assert!(combined.retry);

        // Once true, stays true even if incoming is false
        merge_result(
            &mut combined,
            HookResult {
                retry: false,
                ..HookResult::default()
            },
        );
        assert!(combined.retry);
    }

    // -- HookRegistry tests --

    #[test]
    fn test_hook_registry_empty() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_hook_registry_register_skill() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "test".into(),
                }],
            }],
        );
        registry.register("test-skill", "/tmp/test", def);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_hook_registry_register_user_hooks() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::Stop,
            vec![HookRule {
                matcher: String::new(),
                hooks: vec![HookAction::Prompt {
                    prompt: "Don't stop".into(),
                }],
            }],
        );
        registry.register_user_hooks(def);
        assert!(!registry.is_empty());
        assert!(registry.has_hooks_for(HookEvent::Stop));
        assert!(!registry.has_hooks_for(HookEvent::PreToolUse));
    }

    #[test]
    fn all_hook_sources_filter_invalid_actions_and_support_dry_run() {
        fn definition() -> HooksDefinition {
            HooksDefinition {
                rules: HashMap::from([(
                    HookEvent::PreToolUse,
                    vec![HookRule {
                        matcher: "Bash".to_string(),
                        hooks: vec![
                            HookAction::Prompt {
                                prompt: String::new(),
                            },
                            HookAction::Prompt {
                                prompt: "valid".to_string(),
                            },
                        ],
                    }],
                )]),
            }
        }

        let mut registry = HookRegistry::new();
        registry.register_user_hooks(definition());
        registry.register("formatting", "/tmp/skill", definition());
        assert!(registry.register_plugin_hooks(
            "local-tools",
            "/tmp/plugin",
            "/tmp/plugin-data",
            definition()
        ));

        let result = registry.dry_run(&HookContext::for_dry_run(HookEvent::PreToolUse, "Bash"));
        assert_eq!(result.matches.len(), 3);
        assert!(
            result
                .matches
                .iter()
                .all(|matched| matched.action == "prompt")
        );
        assert_eq!(
            result
                .matches
                .iter()
                .map(|matched| matched.source.as_str())
                .collect::<Vec<_>>(),
            vec!["user_config", "plugin:local-tools", "skill:formatting"]
        );
    }

    #[test]
    fn test_hook_registry_unregister() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "test".into(),
                }],
            }],
        );
        registry.register("test-skill", "/tmp/test", def);
        assert!(!registry.is_empty());

        let removed = registry.unregister(&HookSource::Skill("test-skill".to_string()));
        assert!(removed);
        assert!(registry.is_empty());
    }

    #[tokio::test]
    async fn test_hook_registry_no_match() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Write".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "check".into(),
                }],
            }],
        );
        registry.register("test", "/tmp", def);

        let result = registry.run_pre_tool_use("Read", &json!({}), "").await;
        assert!(result.messages.is_empty());
    }

    #[tokio::test]
    async fn test_hook_registry_prompt_match() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "Verify the command is safe".into(),
                }],
            }],
        );
        registry.register("security", "/tmp", def);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "ls"}), "")
            .await;
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0], "Verify the command is safe");
    }

    #[tokio::test]
    async fn test_hook_registry_command_execution() {
        if cfg!(target_os = "windows") {
            return;
        }
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Command {
                    command: r#"echo '{"decision":"allow"}'"#.into(),
                    shell: None,
                    timeout: 5,
                }],
            }],
        );
        registry.register("test", "/tmp", def);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "ls"}), "")
            .await;
        assert!(!result.block);
    }

    #[tokio::test]
    async fn plugin_command_hook_receives_portable_root_and_data_env() {
        if cfg!(target_os = "windows") {
            return;
        }
        let plugin_root = std::env::temp_dir().join(format!(
            "echo-hook-plugin-{} (dev); local",
            std::process::id()
        ));
        let plugin_data =
            std::env::temp_dir().join(format!("echo-hook-plugin-data-{}", std::process::id()));
        let mut registry = HookRegistry::new();
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::SessionStart,
            vec![HookRule {
                matcher: "startup".to_string(),
                hooks: vec![HookAction::Command {
                    command: "printf '%s|%s' \"$PLUGIN_ROOT\" \"$PLUGIN_DATA\"".to_string(),
                    shell: None,
                    timeout: 5,
                }],
            }],
        );
        assert!(registry.register_plugin_hooks(
            "env-test",
            &plugin_root.display().to_string(),
            &plugin_data.display().to_string(),
            definition,
        ));

        let result = registry
            .run_lifecycle_hooks(&HookContext::for_session_start("startup", "session", "eko"))
            .await;

        assert_eq!(
            result.injected_context,
            Some(format!(
                "{}|{}",
                plugin_root.display(),
                plugin_data.display()
            ))
        );
    }

    #[tokio::test]
    async fn subagent_hook_dispatches_through_registered_executor() {
        let mut registry = HookRegistry::new();
        registry.set_subagent_executor(Arc::new(|name, task| {
            Box::pin(async move { Ok(format!("{name}:{task}")) })
        }));
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::SessionStart,
            vec![HookRule {
                matcher: "startup".to_string(),
                hooks: vec![HookAction::Subagent {
                    name: "reviewer".to_string(),
                    task: Some("Review the change".to_string()),
                    timeout: 5,
                }],
            }],
        );
        registry.register_user_hooks(definition);

        let result = registry
            .run_lifecycle_hooks(&HookContext::for_session_start("startup", "session", "eko"))
            .await;

        assert_eq!(
            result.messages,
            vec!["Subagent 'reviewer' output: reviewer:Review the change"]
        );
    }

    #[tokio::test]
    async fn test_hook_registry_lifecycle_hooks() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::SessionStart,
            vec![HookRule {
                matcher: "startup".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "Welcome!".into(),
                }],
            }],
        );
        registry.register("test", "/tmp", def);

        let ctx = HookContext::for_session_start("startup", "sess-1", "agent");
        let result = registry.run_lifecycle_hooks(&ctx).await;
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0], "Welcome!");

        // Non-matching matcher
        let ctx = HookContext::for_session_start("resume", "sess-1", "agent");
        let result = registry.run_lifecycle_hooks(&ctx).await;
        assert!(result.messages.is_empty());
    }

    #[tokio::test]
    async fn test_hook_registry_stop_hook_with_continue() {
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::Stop,
            vec![HookRule {
                matcher: String::new(),
                hooks: vec![HookAction::Command {
                    command: r#"echo '{"continue_reason": "Run tests first"}'"#.into(),
                    shell: None,
                    timeout: 5,
                }],
            }],
        );
        registry.register("test", "/tmp", def);

        let ctx = HookContext::for_stop(None, "sess-1", "agent", false);
        let result = registry.run_lifecycle_hooks(&ctx).await;
        assert_eq!(result.continue_reason, Some("Run tests first".to_string()));
    }

    #[tokio::test]
    async fn test_hook_registry_user_config_priority() {
        let mut registry = HookRegistry::new();

        // Register skill hook first
        let mut skill_def = HooksDefinition::default();
        skill_def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-skill".into(),
                }],
            }],
        );
        registry.register("z-skill", "/tmp", skill_def);

        // Register user config
        let mut user_def = HooksDefinition::default();
        user_def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-user".into(),
                }],
            }],
        );
        registry.register_user_hooks(user_def);

        let result = registry.run_pre_tool_use("Bash", &json!({}), "").await;
        // User config runs first, then skill
        assert_eq!(result.messages, vec!["from-user", "from-skill"]);
    }

    #[tokio::test]
    async fn test_hook_registry_runs_in_deterministic_skill_name_order() {
        let mut registry = HookRegistry::new();
        let mut z_def = HooksDefinition::default();
        z_def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-z".into(),
                }],
            }],
        );
        registry.register("z-skill", "/tmp", z_def);

        let mut a_def = HooksDefinition::default();
        a_def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Prompt {
                    prompt: "from-a".into(),
                }],
            }],
        );
        registry.register("a-skill", "/tmp", a_def);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "ls"}), "")
            .await;
        assert_eq!(result.messages, vec!["from-a", "from-z"]);
    }

    #[tokio::test]
    async fn test_hook_command_receives_session_id() {
        if cfg!(target_os = "windows") {
            return;
        }
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".into(),
                hooks: vec![HookAction::Command {
                    command: r#"printf '{"updatedInput":{"session_id":"%s"}}' "$SESSION_ID""#
                        .into(),
                    shell: None,
                    timeout: 5,
                }],
            }],
        );
        registry.register("session-skill", "/tmp", def);

        let result = registry
            .run_pre_tool_use("Bash", &json!({"command": "ls"}), "sess-123")
            .await;
        assert_eq!(
            result.updated_input,
            Some(json!({"session_id": "sess-123"}))
        );
    }

    #[test]
    fn test_matches_tool_name_glob() {
        assert!(matches_tool_name("Bash(*)", "Bash(git:*)"));
        assert!(matches_tool_name("Bash", "Bash(git:*)"));
        assert!(matches_tool_name("*", "Read"));
        assert!(!matches_tool_name("Bash", "Read"));
        assert!(!matches_tool_name("Write", "Bash(git:*)"));
    }

    // -- HookAction serde tests --

    #[test]
    fn test_hook_action_http_deserialize() {
        let json = r#"{"type":"http","url":"https://example.com","timeout":5}"#;
        let action: HookAction = serde_json::from_str(json).unwrap();
        assert!(
            matches!(action, HookAction::Http { url, timeout: 5, .. } if url == "https://example.com")
        );
    }

    #[test]
    fn test_hook_action_mcp_tool_deserialize() {
        let json = r#"{"type":"mcp_tool","server":"slack","tool":"send_message","arguments":{"channel":"test"}}"#;
        let action: HookAction = serde_json::from_str(json).unwrap();
        assert!(
            matches!(action, HookAction::McpTool { server, tool, .. } if server == "slack" && tool == "send_message")
        );
    }

    #[test]
    fn test_hook_action_subagent_deserialize() -> Result<(), String> {
        let json = r#"{"type":"subagent","name":"reviewer","task":"Review the change"}"#;
        let action: HookAction = serde_json::from_str(json).map_err(|error| error.to_string())?;
        assert!(
            matches!(action, HookAction::Subagent { name, task, .. } if name == "reviewer" && task.as_deref() == Some("Review the change"))
        );
        Ok(())
    }

    // -- HookAction validation tests --

    #[test]
    fn test_hook_action_validate_command_ok() {
        let action = HookAction::Command {
            command: "echo hello".into(),
            shell: None,
            timeout: 10,
        };
        assert!(action.validate().is_ok());
    }

    #[test]
    fn test_hook_action_validate_command_empty() {
        let action = HookAction::Command {
            command: "".into(),
            shell: None,
            timeout: 10,
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_command_timeout_exceeds_max() {
        let action = HookAction::Command {
            command: "echo hi".into(),
            shell: None,
            timeout: 99999,
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_prompt_empty() {
        let action = HookAction::Prompt { prompt: "".into() };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_permission_invalid() {
        let action = HookAction::Permission {
            decision: "maybe".into(),
            reason: None,
            suggestions: vec![],
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_permission_valid() {
        for dec in &["allow", "deny", "ask"] {
            let action = HookAction::Permission {
                decision: (*dec).into(),
                reason: None,
                suggestions: vec![],
            };
            assert!(
                action.validate().is_ok(),
                "decision '{}' should be valid",
                dec
            );
        }
    }

    #[test]
    fn test_hook_action_validate_http_empty_url() {
        let action = HookAction::Http {
            url: "".into(),
            method: None,
            headers: None,
            timeout: 10,
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_http_timeout_exceeds_max() {
        let action = HookAction::Http {
            url: "https://example.com".into(),
            method: None,
            headers: None,
            timeout: 99999,
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn http_hooks_allow_local_cleartext_but_reject_remote_cleartext() {
        for url in [
            "http://localhost:3000/hooks",
            "http://printer/hooks",
            "http://eko.local/hooks",
            "http://service.lan/hooks",
            "http://127.0.0.1:8080/hooks",
            "http://192.168.1.20/hooks",
            "http://[::1]:8080/hooks",
        ] {
            let action = HookAction::Http {
                url: url.to_string(),
                method: None,
                headers: None,
                timeout: 10,
            };
            assert!(
                action.validate().is_ok(),
                "local URL should be valid: {url}"
            );
        }
        let remote = HookAction::Http {
            url: "http://example.com/hooks".to_string(),
            method: None,
            headers: None,
            timeout: 10,
        };
        assert!(remote.validate().is_err());
    }

    #[tokio::test]
    async fn mcp_hooks_allow_user_configured_tool_names() {
        let mut registry = HookRegistry::new();
        registry.set_mcp_executor(Arc::new(|server, tool, _arguments| {
            Box::pin(async move {
                let mut result = HookResult::default();
                result.messages.push(format!("{server}:{tool}"));
                result
            })
        }));
        let mut definition = HooksDefinition::default();
        definition.add_rules(
            HookEvent::PreToolUse,
            vec![HookRule {
                matcher: "Bash".to_string(),
                hooks: vec![HookAction::McpTool {
                    server: "local-dev".to_string(),
                    tool: "bash".to_string(),
                    arguments: None,
                    timeout: 10,
                }],
            }],
        );
        registry.register_user_hooks(definition);

        let result = registry.run_pre_tool_use("Bash", &json!({}), "").await;

        assert_eq!(result.messages, vec!["local-dev:bash"]);
    }

    #[test]
    fn test_hook_action_validate_mcp_empty_server() {
        let action = HookAction::McpTool {
            server: "".into(),
            tool: "send".into(),
            arguments: None,
            timeout: 10,
        };
        assert!(action.validate().is_err());
    }

    #[test]
    fn test_hook_action_validate_mcp_timeout_exceeds_max() {
        let action = HookAction::McpTool {
            server: "s".into(),
            tool: "t".into(),
            arguments: None,
            timeout: 99999,
        };
        assert!(action.validate().is_err());
    }

    // ── ActivateSkill tests ──

    #[test]
    fn validate_rejects_empty_skill_name() {
        let a = HookAction::ActivateSkill {
            skill: String::new(),
            reason: "r".into(),
        };
        assert!(a.validate().is_err());
    }

    #[test]
    fn validate_accepts_nonempty() {
        let a = HookAction::ActivateSkill {
            skill: "docx".into(),
            reason: "r".into(),
        };
        assert!(a.validate().is_ok());
    }

    #[test]
    fn merge_takes_first_activate_skill() {
        let mut combined = HookResult::default();
        let first = HookResult::with_activate_skill("docx".into(), "r1".into());
        merge_result(&mut combined, first);
        let second = HookResult::with_activate_skill("pdf".into(), "r2".into());
        merge_result(&mut combined, second);
        assert_eq!(
            combined.activate_skill,
            Some(("docx".to_string(), "r1".to_string()))
        );
    }

    #[test]
    fn merge_activate_skill_from_none() {
        let mut combined = HookResult::default();
        merge_result(
            &mut combined,
            HookResult::with_activate_skill("pdf".into(), "r".into()),
        );
        assert_eq!(
            combined.activate_skill.as_ref().map(|(s, _)| s.clone()),
            Some("pdf".into())
        );
    }
}
