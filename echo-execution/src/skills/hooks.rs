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
    /** 不经过 LLM，直接激活一个技能。
    匹配成功时，hook 引擎会调用 `ReactAgent::activate_skill(skill)`。
    `reason` 会作为系统提示展示给模型，说明激活原因。 */
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
    /** 校验该 hook 动作，对不安全/无效的配置返回错误。

    在 hook 注册期间调用，用于尽早发现配置错误。 */
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

/** 汇总所有来源（技能与用户配置）的 hook 注册表。

手动实现 Clone，以保证类型擦除的运行时回调在克隆快照中仍然保留。 */
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

/** 一个已注册的编程式 hook：一个具名闭包，加上它订阅的事件列表
（空列表表示订阅所有事件）。 */
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
    /// 克隆注册表快照，同时保留所有类型擦除的运行时回调（沙箱、HTTP 客户端、执行器等）。
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

    /** 按名称注册（或替换）一个编程式 hook。`events` 限定该执行器关注的事件范围；
    空切片表示订阅所有事件。编程式 hook 会在所有数据驱动来源之后按名称顺序运行，
    合并与短路语义与数据驱动规则相同。 */
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

    /** 按确定的优先级顺序排列 hook 来源：用户配置最先，其次是按名称排序的插件，
    最后是按名称排序的技能。 */
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

        /* 编程式 hook 在所有数据驱动来源之后按名称顺序运行，采用相同的合并/短路
        语义。空的事件订阅表示关注所有事件。 */
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

// ── 动作执行 ───────────────────────────────────────────────────

/// 执行单个 hook 动作时所需的运行时依赖集合（借用自注册表）。
struct HookActionRuntime<'a> {
    /// 插件数据目录；仅插件来源的 hook 才会设置，用于注入插件相关环境变量。
    plugin_data_dir: Option<&'a str>,
    /// 可选的沙箱执行器；存在时命令 hook 在沙箱中运行。
    sandbox: Option<&'a Arc<dyn SandboxExecutor>>,
    /// 可选的共享 HTTP 客户端，供 HTTP hook 复用连接。
    http_client: Option<&'a reqwest::Client>,
    /// 可选的 MCP 工具执行回调，供 McpTool hook 使用。
    mcp_executor: Option<&'a McpExecutorFn>,
    /// 可选的子 agent 执行回调，供 Subagent hook 使用。
    subagent_executor: Option<&'a SubagentExecutorFn>,
}

/// 根据动作类型分发并执行单个 hook 动作，返回该动作产生的 [`HookResult`]。
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
            /* 权限决策会在所有匹配来源之间进行归约。
            尤其是较早出现的 allow/ask 不能掩盖之后出现的 deny。
            deny 仍会在上面设置 `block`，因此会在观察到最高优先级决策后停止执行；
            调用方仍可通过 `continue` 显式请求停止传播。 */
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

// -- 命令 hook 执行 --

/// 执行命令类型的 hook：优先走沙箱，否则直接派生本地进程，并解析其输出。
async fn execute_command_hook(
    command: &str,
    shell: Option<&str>,
    timeout_secs: u64,
    source_dir: &str,
    plugin_data_dir: Option<&str>,
    context: &HookContext,
    sandbox: Option<&Arc<dyn SandboxExecutor>>,
) -> HookResult {
    // 构建通过 stdin 传入的 JSON 上下文（为兼容性附带 hook_event_name）
    let mut stdin_value = serde_json::to_value(context).unwrap_or_default();
    stdin_value["hook_event_name"] = json!(context.event.as_str());
    let stdin_json = stdin_value;

    let timeout = Duration::from_secs(timeout_secs);

    // -- 沙箱执行路径 --
    if let Some(manager) = sandbox {
        let (program, args) = build_hook_shell_command(command, shell);
        let mut sandbox_cmd = SandboxCommand::program(&program, args).with_timeout(timeout);

        if !source_dir.is_empty() && Path::new(source_dir).exists() {
            sandbox_cmd = sandbox_cmd.with_working_dir(source_dir);
        }

        // 使用最小化环境变量
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

        // 通过 stdin 传入 JSON
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

    // -- 回退路径：直接执行进程（无沙箱）--
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

    // 保留用户可信的本地环境，并追加稳定的 hook 字段。
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

// -- HTTP hook 执行 --

/// 执行 HTTP 类型的 hook：以 JSON 形式发送上下文，并将成功响应体按 hook 输出解析。
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

// ── 输出解析 ─────────────────────────────────────────────────────

/** 解析 hook 命令的 JSON 输出。

Hook 可以返回 JSON 来控制执行：
```json
{
  "decision": "block",
  "reason": "Unsafe command detected",
  "continue": false,
  "permission_decision": "allow" | "deny" | "ask",
  "permission_suggestions": ["Allow", "Deny"],
  "continue_reason": "Check if tests pass",
  "injected_context": "Remember to use bun",
  "metadata": {}
}
```
*/
fn parse_hook_output(stdout: &str, stderr: &str, exit_code: i32) -> HookResult {
    let mut result = HookResult::default();

    /* 退出码语义（与 Claude Code 约定保持一致）：
     exit 0：通过（无阻止意图）
     exit 1：不阻止（hook 产生了输出但没有显式阻止）
     exit 2：阻止（显式阻止信号）
     其他非零值：告警，不阻止 */
    match exit_code {
        0 | 1 => {} // 不做隐式阻止
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
            // 其他非零值：记录告警，不阻止
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

    // 尝试解析 JSON 输出
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return result;
    }

    if let Ok(json) = serde_json::from_str::<Value>(trimmed) {
        let hook_specific = json.get("hookSpecificOutput").unwrap_or(&json);
        if let Some(message) = json.get("systemMessage").and_then(Value::as_str) {
            result.messages.push(truncate_hook_message(message));
        }

        // 解析 decision 字段
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

        /* 可移植的 Codex/Claude hook 输出字段。嵌入式应用保留下方原生的扁平字段，
        同时接受这些嵌套字段以便复用插件。 */
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

        // 解析 permission_decision 字段
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

        /* 解析标准的、作用于单次调用的权限模式覆盖。Hook 是用户安装的本地扩展；
        共享/托管部署仍可通过 PermissionService::bypass_disabled 禁用 BypassPermissions。 */
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

        // 解析生命周期相关字段
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

        // 解析 retry 字段（用于 PermissionDenied hook）
        if json.get("retry") == Some(&Value::Bool(true)) {
            result.retry = true;
        }

        if let Some(meta) = json.get("metadata")
            && !meta.is_null()
        {
            result.metadata = Some(meta.clone());
        }
    } else if exit_code == 0 {
        // 退出码为 0 且 stdout 不是 JSON：视为注入的上下文
        result.injected_context = Some(truncate_hook_message(trimmed));
    }

    /* 退出码 2 是带外（out-of-band）阻止信号。结构化的 stdout 可以追加上下文，
    但不得降级这一进程级决策。 */
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

/// 为插件来源的 hook 注入插件根目录和插件数据目录相关的环境变量。
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

/// 构造一个仅携带（截断后）说明消息的 [`HookResult`]，用于报告执行问题。
fn hook_execution_message(message: String) -> HookResult {
    let mut result = HookResult::default();
    result.messages.push(truncate_hook_message(&message));
    result
}

/// 将 hook 消息截断到最多 10000 个字符，超出部分以 `...` 结尾。
fn truncate_hook_message(message: &str) -> String {
    // 单条 hook 消息允许的最大字符数
    const MAX_CHARS: usize = 10_000;
    let mut text = message.chars().take(MAX_CHARS).collect::<String>();
    if message.chars().count() > MAX_CHARS {
        text.push_str("...");
    }
    text
}

// ── Shell 命令构建 ──────────────────────────────────────────────

/// 根据 shell 类型和平台构建执行 hook 命令所用的程序名与参数列表。
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

/// 通过执行 `<cmd> --version` 检测某个命令是否可用。
fn which_exists(cmd: &str) -> bool {
    std::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

// ── 结果合并 ─────────────────────────────────────────────────────

/// 将单个 hook 的结果合并进累积结果，按各字段的归约规则处理。
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
    /* 外部命令/HTTP/程序化 hook 可能同时返回权限决策和 `continue: false`。
    权限安全性会在所有匹配来源之间归约，因此 allow/ask/require-approval
    不能借助该停止位掩盖之后的 deny。不带权限决策的结果仍遵循普通的停止传播约定。 */
    if incoming.stop_propagation && !carries_permission_decision {
        combined.stop_propagation = true;
    }

    // 权限决策优先级：deny > ask > require_approval > allow
    if let Some(new_decision) = incoming.permission_decision {
        let should_replace = match (&combined.permission_decision, &new_decision) {
            // 已有 deny 时保持不变
            (Some(PermissionDecision::Deny { .. }), _) => false,
            // 新决策为 deny —— 总是采用
            (_, PermissionDecision::Deny { .. }) => true,
            // 已有 ask 时保持不变（ask > allow）
            (Some(PermissionDecision::Ask { .. }), _) => false,
            // 新决策为 ask —— 覆盖 allow
            (_, PermissionDecision::Ask { .. }) => true,
            // 已有 RequireApproval 时保持不变
            (Some(PermissionDecision::RequireApproval), _) => false,
            // 新决策为 RequireApproval —— 覆盖 allow
            (_, PermissionDecision::RequireApproval) => true,
            // 两者都是 allow —— 任取其一即可
            (Some(PermissionDecision::Allow), PermissionDecision::Allow) => false,
            // 尚无决策 —— 采用新决策
            (None, _) => true,
        };
        if should_replace {
            combined.permission_decision = Some(new_decision);
        }
    }

    // permission_mode_override：最后一个非 None 值生效
    if incoming.permission_mode_override.is_some() {
        combined.permission_mode_override = incoming.permission_mode_override;
    }

    // continue_reason：非 None 值覆盖 None
    if incoming.continue_reason.is_some() {
        combined.continue_reason = incoming.continue_reason;
    }

    // injected_context：以换行符拼接
    if let Some(ctx) = incoming.injected_context {
        combined.injected_context = Some(match combined.injected_context.take() {
            Some(existing) => format!("{}\n{}", existing, ctx),
            None => ctx,
        });
    }

    // retry：或语义（任一为 true 则合并结果为 true）
    if incoming.retry {
        combined.retry = true;
    }

    /* activate_skill：先到先得（保留第一个非 None 的激活请求；
    忽略其他 hook 后续返回的 activate_skill，以避免多个技能激活相互冲突）。 */
    if combined.activate_skill.is_none() {
        combined.activate_skill = incoming.activate_skill;
    }

    // metadata：深度合并
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

// ── 测试 ──────────────────────────────────────────────────────────────

// 单元测试：hook 事件、匹配、解析、合并与注册表执行。
#[cfg(test)]
mod tests {
    use super::*;

    // -- HookEvent 测试 --

    /// 验证 `is_tool_event` 对工具类事件返回 true、对其他事件返回 false。
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

    /// 验证 `supports_matcher` 对各事件的判断结果。
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

    /// 验证 `HookEvent` 的序列化与反序列化往返一致。
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

    // -- HookContext 测试 --

    /// 验证 PreToolUse 上下文构造携带工具名与输入。
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

    /// 验证 SessionStart 上下文的构造。
    #[test]
    fn test_hook_context_for_session_start() {
        let ctx = HookContext::for_session_start("startup", "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::SessionStart);
        assert_eq!(ctx.matcher.as_deref(), Some("startup"));
        assert!(ctx.tool_name.is_none());
    }

    /// 验证 Stop 上下文的构造。
    #[test]
    fn test_hook_context_for_stop() {
        let ctx = HookContext::for_stop(None, "sess-1", "agent", false);
        assert_eq!(ctx.event, HookEvent::Stop);
        assert!(ctx.matcher.is_none());
        assert!(ctx.tool_name.is_none());
    }

    /// 验证 Notification 上下文的构造。
    #[test]
    fn test_hook_context_for_notification() {
        let ctx = HookContext::for_notification("permission_prompt", "sess-1", "agent");
        assert_eq!(ctx.event, HookEvent::Notification);
        assert_eq!(ctx.matcher.as_deref(), Some("permission_prompt"));
    }

    /// 验证 `HookContext` 能正确序列化为 JSON。
    #[test]
    fn test_hook_context_serialization() {
        let ctx =
            HookContext::for_pre_tool_use("Bash", &json!({"command": "ls"}), "sess-1", "agent");
        let json_str = serde_json::to_string(&ctx).unwrap();
        assert!(json_str.contains("\"event\":\"PreToolUse\""));
        assert!(json_str.contains("\"tool_name\":\"Bash\""));
        // 值为 None 的字段应被省略
        assert!(!json_str.contains("\"tool_output\""));
        assert!(!json_str.contains("\"matcher\""));
    }

    // -- matches_hook 测试 --

    /// 验证通配符 matcher 匹配任意工具。
    #[test]
    fn test_matches_hook_wildcard() {
        let ctx = HookContext::for_pre_tool_use("Bash", &json!({}), "", "");
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
    }

    /// 验证工具名精确匹配。
    #[test]
    fn test_matches_hook_tool_name_exact() {
        let ctx = HookContext::for_pre_tool_use("Bash", &json!({}), "", "");
        assert!(matches_hook("Bash", &ctx));
        assert!(!matches_hook("Read", &ctx));
    }

    /// 验证以 `|` 分隔的多个工具名匹配。
    #[test]
    fn test_matches_hook_tool_name_pipe_separated() {
        let ctx = HookContext::for_pre_tool_use("Edit", &json!({}), "", "");
        assert!(matches_hook("Edit|Write", &ctx));
        assert!(matches_hook("Write|Edit", &ctx));
        assert!(!matches_hook("Bash|Read", &ctx));
    }

    /// 验证工具名前缀（glob）匹配。
    #[test]
    fn test_matches_hook_tool_name_prefix() {
        let ctx = HookContext::for_pre_tool_use("Bash(git:*)", &json!({}), "", "");
        assert!(matches_hook("Bash", &ctx));
    }

    /// 验证生命周期事件的 matcher 匹配。
    #[test]
    fn test_matches_hook_lifecycle_matcher() {
        let ctx = HookContext::for_session_start("startup", "", "");
        assert!(matches_hook("startup", &ctx));
        assert!(!matches_hook("resume", &ctx));
    }

    /// 验证生命周期事件以 `|` 分隔的多值匹配。
    #[test]
    fn test_matches_hook_lifecycle_pipe_separated() {
        let ctx = HookContext::for_session_start("resume", "", "");
        assert!(matches_hook("startup|resume", &ctx));
    }

    /// 验证生命周期事件的 glob 匹配。
    #[test]
    fn test_matches_hook_lifecycle_glob() {
        let ctx = HookContext::for_notification("permission_prompt", "", "");
        assert!(matches_hook("permission*", &ctx));
    }

    /// 验证不支持 matcher 的事件总是匹配。
    #[test]
    fn test_matches_hook_no_matcher_event() {
        let ctx = HookContext::for_stop(None, "", "", false);
        // Stop 不支持 matcher，因此只有 "*" 和 "" 能匹配
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
        assert!(!matches_hook("something", &ctx));
    }

    /// 验证 SubagentStart 事件按子代理名匹配。
    #[test]
    fn test_matches_hook_subagent_start() {
        let ctx = HookContext::for_subagent_start("coder", "sync", "implement X", "", "");
        assert!(matches_hook("coder", &ctx));
        assert!(matches_hook("*", &ctx));
        assert!(matches_hook("", &ctx));
        assert!(!matches_hook("planner", &ctx));
        // 以 `|` 分隔的多个候选
        assert!(matches_hook("planner|coder|reviewer", &ctx));
    }

    /// 验证 SubagentStop 事件按子代理名匹配。
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

    /// 验证 TaskCreated 事件的匹配。
    #[test]
    fn test_matches_hook_task_created() {
        let ctx = HookContext::for_task_created("t-1", "build API", "", "");
        assert!(matches_hook("build API", &ctx));
        assert!(matches_hook("*", &ctx));
        assert!(!matches_hook("deploy", &ctx));
    }

    /// 验证 TaskCompleted 事件的匹配。
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

    /// 验证子代理事件的 glob 匹配。
    #[test]
    fn test_matches_hook_subagent_glob() {
        let ctx = HookContext::for_subagent_start("code-reviewer", "sync", "review", "", "");
        assert!(matches_hook("code*", &ctx));
        assert!(!matches_hook("test*", &ctx));
    }

    // -- HooksDefinition 测试 --

    /// 验证默认的 `HooksDefinition` 为空。
    #[test]
    fn test_hooks_definition_empty() {
        let def = HooksDefinition::default();
        assert!(def.is_empty());
    }

    /// 验证 `add_rules` 能向定义中添加规则。
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

    /// 验证插件 hook 以独立的插件来源注册。
    #[test]
    fn test_register_plugin_hooks_distinct_source() {
        /* P0-2：插件 hook 必须注册在 HookSource::Plugin 下，而不能被并入
        HookSource::Skill。验证该来源标识在 list_sources 中独立出现。 */
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
        // 应显示为 plugin:data-analysis-pack，绝不能是 skill:plugin:...
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

    /// 验证插件 hook 注册时丢弃无效动作并保留有效动作。
    #[test]
    fn test_register_plugin_hooks_drops_invalid_actions() {
        /* P0-2：无效动作被丢弃，有效的同级动作保留，
        且插件仍注册在插件来源下。 */
        let mut registry = HookRegistry::new();
        let mut def = HooksDefinition::default();
        def.add_rules(
            HookEvent::PostToolUse,
            vec![HookRule {
                matcher: "*".into(),
                hooks: vec![
                    // 有效
                    HookAction::Prompt {
                        prompt: "keep".into(),
                    },
                    // 无效：命令为空
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

    /// 验证注册空的插件 hook 定义不产生任何效果。
    #[test]
    fn test_register_plugin_hooks_empty_noop() {
        let mut registry = HookRegistry::new();
        registry.register_plugin_hooks("empty-plugin", "/p", "/data/p", HooksDefinition::default());
        assert!(registry.is_empty(), "empty definition must not register");
    }

    /// 验证两个 `HooksDefinition` 的合并结果。
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

    /// 验证从 YAML 反序列化 hooks 定义。
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

    /// 验证反序列化包含 HTTP 动作的 hooks 定义。
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

    /// 验证反序列化包含 MCP 工具动作的 hooks 定义。
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

    // -- HookResult 测试 --

    /// 验证 allow 结果的构造。
    #[test]
    fn test_hook_result_allow() {
        let result = HookResult::allow();
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_allowed());
    }

    /// 验证 deny 结果的构造。
    #[test]
    fn test_hook_result_deny() {
        let result = HookResult::deny("test reason".to_string());
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_denied());
        assert!(result.block);
    }

    /// 验证 ask 结果的构造。
    #[test]
    fn test_hook_result_ask() {
        let result = HookResult::ask(vec!["Option A".to_string()]);
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().requires_approval());
    }

    /// 验证 `should_continue` 的判断逻辑。
    #[test]
    fn test_hook_result_should_continue() {
        let mut result = HookResult::default();
        assert!(!result.should_continue());
        result.continue_reason = Some("Check tests".to_string());
        assert!(result.should_continue());
    }

    // -- parse_hook_output 测试 --

    /// 验证空输出解析为默认结果。
    #[test]
    fn test_parse_hook_output_empty() {
        let result = parse_hook_output("", "", 0);
        assert!(!result.block);
    }

    /// 验证解析 block 决策输出。
    #[test]
    fn test_parse_hook_output_block() {
        let result = parse_hook_output(r#"{"decision": "block", "reason": "unsafe"}"#, "", 0);
        assert!(result.block);
        assert_eq!(result.block_reason, Some("unsafe".into()));
    }

    /// 验证解析 allow 决策输出。
    #[test]
    fn test_parse_hook_output_allow() {
        let result = parse_hook_output(r#"{"decision": "allow"}"#, "", 1);
        assert!(!result.block);
    }

    /// 验证退出码 1 不会阻止执行。
    #[test]
    fn test_parse_hook_output_nonzero_exit_code_1_no_block() {
        // exit 1 = 不阻止（hook 产生了输出但无阻止意图）
        let result = parse_hook_output("", "", 1);
        assert!(!result.block);
    }

    /// 验证退出码 2 会阻止执行。
    #[test]
    fn test_parse_hook_output_exit_code_2_block() {
        // exit 2 = 显式阻止信号
        let result = parse_hook_output("", "", 2);
        assert!(result.block);
        assert_eq!(
            result.block_reason,
            Some("Hook exited with code 2 (explicit block)".to_string())
        );
    }

    /// 验证退出码 2 时使用 stderr 作为阻止原因。
    #[test]
    fn test_parse_hook_output_exit_code_2_uses_stderr_reason() {
        let result = parse_hook_output("", "Blocked by policy", 2);
        assert!(result.block);
        assert_eq!(result.block_reason.as_deref(), Some("Blocked by policy"));
    }

    /// 验证退出码 2 的阻止不能被 stdout 输出降级。
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

    /// 验证其他退出码不会阻止执行。
    #[test]
    fn test_parse_hook_output_exit_code_other_no_block() {
        // 其他非零值 = 告警，不阻止
        let result = parse_hook_output("", "", 3);
        assert!(!result.block);
        let result = parse_hook_output("", "", 127);
        assert!(!result.block);
    }

    /// 验证解析 retry 字段。
    #[test]
    fn test_parse_hook_output_retry_field() {
        let result = parse_hook_output(r#"{"retry": true}"#, "", 0);
        assert!(result.retry);
    }

    /// 验证 retry 默认为 false。
    #[test]
    fn test_parse_hook_output_retry_false_by_default() {
        let result = parse_hook_output("", "", 0);
        assert!(!result.retry);
    }

    /// 验证解析 updated_input 字段。
    #[test]
    fn test_parse_hook_output_updated_input() {
        let result = parse_hook_output(r#"{"updatedInput": {"command": "safe-command"}}"#, "", 0);
        assert!(!result.block);
        assert_eq!(
            result.updated_input,
            Some(json!({"command": "safe-command"}))
        );
    }

    /// 验证能解析可移植格式的嵌套字段（hookSpecificOutput 等）。
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

    /// 验证能解析 PermissionRequest 事件的决策输出。
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

    /// 验证解析 continue_reason 字段。
    #[test]
    fn test_parse_hook_output_continue_reason() {
        let result =
            parse_hook_output(r#"{"continue_reason": "Run tests before stopping"}"#, "", 0);
        assert_eq!(
            result.continue_reason,
            Some("Run tests before stopping".to_string())
        );
    }

    /// 验证解析 injected_context 字段。
    #[test]
    fn test_parse_hook_output_injected_context() {
        let result = parse_hook_output(r#"{"injected_context": "Remember to use bun"}"#, "", 0);
        assert_eq!(
            result.injected_context,
            Some("Remember to use bun".to_string())
        );
    }

    /// 验证非 JSON 输出被视为注入上下文。
    #[test]
    fn test_parse_hook_output_non_json_as_context() {
        let result = parse_hook_output("Remember: use bun, not npm", "", 0);
        assert_eq!(
            result.injected_context,
            Some("Remember: use bun, not npm".to_string())
        );
    }

    /// 验证解析 permission_decision 为 allow。
    #[test]
    fn test_parse_hook_output_permission_decision_allow() {
        let result = parse_hook_output(r#"{"permission_decision": "allow"}"#, "", 0);
        assert!(result.has_permission_decision());
        assert!(result.permission_decision.unwrap().is_allowed());
    }

    /// 验证解析 permission_decision 为 deny。
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

    /// 验证解析权限模式覆盖字段。
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

    /// 验证忽略旧版的 permission_mode 字段。
    #[test]
    fn test_parse_hook_output_ignores_legacy_permission_mode() {
        let result = parse_hook_output(r#"{"permission_mode": "auto"}"#, "", 0);
        assert_eq!(result.permission_mode_override, None);
    }

    // -- merge_result 测试 --

    /// 验证合并权限决策时 deny 优先级最高。
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

    /// 验证权限 hook 在 allow/ask 短路前先跨来源归并决策。
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

    /// 验证 continue=false 的权限输出不能掩盖后续的 deny。
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

    /// 验证命令型权限 hook 的停止不能掩盖后续来源的 deny。
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

    /// 验证合并 continue_reason。
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

        // 后出现的非 None 值覆盖之前的值
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

    /// 验证合并时 injected_context 被拼接。
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

    /// 验证合并时 metadata 深度合并。
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

    /// 验证权限模式覆盖在合并时以最后一个为准。
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

    /// 验证 retry 合并采用“或”语义。
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

        // 一旦为 true，即使后续传入 false 也保持 true
        merge_result(
            &mut combined,
            HookResult {
                retry: false,
                ..HookResult::default()
            },
        );
        assert!(combined.retry);
    }

    // -- HookRegistry 测试 --

    /// 验证新建的注册表为空。
    #[test]
    fn test_hook_registry_empty() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());
    }

    /// 验证注册 skill hooks。
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

    /// 验证注册用户配置 hooks。
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

    /// 验证所有 hook 来源都会过滤无效动作，并支持 dry-run。
    #[test]
    fn all_hook_sources_filter_invalid_actions_and_support_dry_run() {
        /// 构造包含有效与无效动作的测试用 hooks 定义。
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

    /// 验证注销 skill hooks。
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

    /// 验证无匹配规则时返回默认结果。
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

    /// 验证 prompt 型 hook 匹配时注入消息。
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

    /// 验证命令型 hook 的执行。
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

    /// 验证插件命令 hook 能收到可移植的插件根目录与数据目录环境变量。
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

    /// 验证子代理 hook 通过已注册的执行器分发。
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

    /// 验证生命周期事件 hooks 的执行。
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

        // 不匹配的 matcher
        let ctx = HookContext::for_session_start("resume", "sess-1", "agent");
        let result = registry.run_lifecycle_hooks(&ctx).await;
        assert!(result.messages.is_empty());
    }

    /// 验证 Stop hook 返回 continue 时的行为。
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

    /// 验证用户配置 hooks 的优先级。
    #[tokio::test]
    async fn test_hook_registry_user_config_priority() {
        let mut registry = HookRegistry::new();

        // 先注册技能 hook
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

        // 注册用户配置
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
        // 用户配置先执行，然后才是技能
        assert_eq!(result.messages, vec!["from-user", "from-skill"]);
    }

    /// 验证 hooks 按 skill 名的确定顺序执行。
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

    /// 验证命令型 hook 能收到 session id。
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

    /// 验证工具名的 glob 匹配。
    #[test]
    fn test_matches_tool_name_glob() {
        assert!(matches_tool_name("Bash(*)", "Bash(git:*)"));
        assert!(matches_tool_name("Bash", "Bash(git:*)"));
        assert!(matches_tool_name("*", "Read"));
        assert!(!matches_tool_name("Bash", "Read"));
        assert!(!matches_tool_name("Write", "Bash(git:*)"));
    }

    // -- HookAction 序列化/反序列化测试 --

    /// 验证 HTTP 动作的反序列化。
    #[test]
    fn test_hook_action_http_deserialize() {
        let json = r#"{"type":"http","url":"https://example.com","timeout":5}"#;
        let action: HookAction = serde_json::from_str(json).unwrap();
        assert!(
            matches!(action, HookAction::Http { url, timeout: 5, .. } if url == "https://example.com")
        );
    }

    /// 验证 MCP 工具动作的反序列化。
    #[test]
    fn test_hook_action_mcp_tool_deserialize() {
        let json = r#"{"type":"mcp_tool","server":"slack","tool":"send_message","arguments":{"channel":"test"}}"#;
        let action: HookAction = serde_json::from_str(json).unwrap();
        assert!(
            matches!(action, HookAction::McpTool { server, tool, .. } if server == "slack" && tool == "send_message")
        );
    }

    /// 验证子代理动作的反序列化。
    #[test]
    fn test_hook_action_subagent_deserialize() -> Result<(), String> {
        let json = r#"{"type":"subagent","name":"reviewer","task":"Review the change"}"#;
        let action: HookAction = serde_json::from_str(json).map_err(|error| error.to_string())?;
        assert!(
            matches!(action, HookAction::Subagent { name, task, .. } if name == "reviewer" && task.as_deref() == Some("Review the change"))
        );
        Ok(())
    }

    // -- HookAction 校验测试 --

    /// 验证合法命令动作通过校验。
    #[test]
    fn test_hook_action_validate_command_ok() {
        let action = HookAction::Command {
            command: "echo hello".into(),
            shell: None,
            timeout: 10,
        };
        assert!(action.validate().is_ok());
    }

    /// 验证空命令动作校验失败。
    #[test]
    fn test_hook_action_validate_command_empty() {
        let action = HookAction::Command {
            command: "".into(),
            shell: None,
            timeout: 10,
        };
        assert!(action.validate().is_err());
    }

    /// 验证命令动作超时超过上限时校验失败。
    #[test]
    fn test_hook_action_validate_command_timeout_exceeds_max() {
        let action = HookAction::Command {
            command: "echo hi".into(),
            shell: None,
            timeout: 99999,
        };
        assert!(action.validate().is_err());
    }

    /// 验证空 prompt 动作校验失败。
    #[test]
    fn test_hook_action_validate_prompt_empty() {
        let action = HookAction::Prompt { prompt: "".into() };
        assert!(action.validate().is_err());
    }

    /// 验证非法权限决策校验失败。
    #[test]
    fn test_hook_action_validate_permission_invalid() {
        let action = HookAction::Permission {
            decision: "maybe".into(),
            reason: None,
            suggestions: vec![],
        };
        assert!(action.validate().is_err());
    }

    /// 验证合法权限决策通过校验。
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

    /// 验证 HTTP 动作 URL 为空时校验失败。
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

    /// 验证 HTTP 动作超时超过上限时校验失败。
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

    /// 验证 HTTP hook 允许本地明文地址、拒绝远程明文地址。
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

    /// 验证 MCP hook 允许用户配置的工具名。
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

    /// 验证 MCP 动作服务器名为空时校验失败。
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

    /// 验证 MCP 动作超时超过上限时校验失败。
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

    // ── ActivateSkill 测试 ──

    /// 验证 ActivateSkill 动作的 skill 名为空时校验失败。
    #[test]
    fn validate_rejects_empty_skill_name() {
        let a = HookAction::ActivateSkill {
            skill: String::new(),
            reason: "r".into(),
        };
        assert!(a.validate().is_err());
    }

    /// 验证 ActivateSkill 动作的 skill 名非空时通过校验。
    #[test]
    fn validate_accepts_nonempty() {
        let a = HookAction::ActivateSkill {
            skill: "docx".into(),
            reason: "r".into(),
        };
        assert!(a.validate().is_ok());
    }

    /// 验证合并时保留第一个 activate_skill。
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

    /// 验证从无到有合并 activate_skill。
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
