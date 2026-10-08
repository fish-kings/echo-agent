//! 子 Agent 核心类型 —— 定义、执行模式与结果。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::usage::LlmUsageStats;
use echo_core::agent::ExecutionUsage;

// ── 执行模式 ────────────────────────────────────────────────────────────

/// 子 Agent 相对于父级的执行方式。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ExecutionMode {
    /** Sync/delegate（同步委派）：父级会阻塞，直到子 Agent 返回。
     * 对应现有的 `AgentDispatchTool` 行为。
     */
    #[default]
    Sync,
    /** Fork（分叉）：独立运行，可选超时。它以已注册的角色提示词起步，
     * 只有在显式上下文传递策略要求时，才会接收经过过滤的历史。
     */
    Fork,
    /// Teammate（队友）：并行独立的 Agent，通过消息传递进行协调。
    Teammate,
    /// Team（团队）意图，被编译进权威的、带版本的任务 DAG 运行时。
    Team,
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Sync => write!(f, "sync"),
            ExecutionMode::Fork => write!(f, "fork"),
            ExecutionMode::Teammate => write!(f, "teammate"),
            ExecutionMode::Team => write!(f, "team"),
        }
    }
}

/// 为一次派发建立的隔离边界的产品中立名称。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservedIsolation(String);

impl ObservedIsolation {
    /// 从任意字符串构造，超长会按 `MAX_KIND_CHARS` 截断，空值回退为默认值。
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        let value = value.trim();
        if value.is_empty() {
            Self::default()
        } else {
            Self(value.chars().take(MAX_KIND_CHARS).collect())
        }
    }

    /// 以字符串切片形式借出内部值。
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ObservedIsolation {
    fn default() -> Self {
        Self("unknown".to_string())
    }
}

// ── 子 Agent 来源类型 ─────────────────────────────────────────────────────────────

/// 子 Agent 定义的来源方式。
#[derive(Debug, Clone, Default)]
pub enum SubagentKind {
    /// 由库或应用硬编码提供的 Agent。
    #[default]
    BuiltIn,
    /// 从 `.md` 定义文件加载（类似于 skills）。
    Custom {
        /// 定义文件的路径。
        path: PathBuf,
    },
    /// 从插件或远程注册表加载。
    Plugin {
        /// 来源标识（例如插件名或注册表 URL）。
        source: String,
    },
}

/// 由子 Agent 定义声明的变更权限。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentAccessMode {
    /// 只读，不允许产生变更。
    ReadOnly,
    /// 允许写入 / 产生变更。
    Write,
}

// ── 子 Agent 定义 ───────────────────────────────────────────────────────

/** 子 Agent 的声明式规格说明。
 *
 * 描述子 Agent “是什么样子”（模型、工具、提示词、约束），而不规定
 * “如何构建它”——那是工厂（factory）的职责。
 */
#[derive(Debug, Clone)]
pub struct SubagentDefinition {
    /// 用于发现和派发的唯一名称。
    pub name: String,
    /// 人类可读的描述（在工具描述中暴露给 LLM）。
    pub description: String,
    /// 该 Agent 的来源方式。
    pub kind: SubagentKind,
    /** 类型化的变更边界。标签（tags）仅是发现用的元数据，绝不能被
     * 反解码回这个权限本身。
     */
    pub access_mode: SubagentAccessMode,
    /// 派发时使用的默认执行模式。
    pub execution_mode: ExecutionMode,
    /// 模型覆盖项（`None` = 继承自父级）。
    pub model: Option<String>,
    /** 该子 Agent 的推理深度覆盖项，使用
     * `echo_core::llm::ThinkingConfig::parse_spec` 的语法
     * （`low`/`medium`/`high`/`disabled`/预算数字；`auto`/空 = 模型默认值）。
     * `None` = 继承父级生成配置的推理设置。用于那些有意使用较低
     * 推理深度的廉价、长时间运行角色。
     */
    pub thinking: Option<String>,
    /// System prompt 覆盖项（`None` = 继承或自动生成）。
    pub system_prompt: Option<String>,
    /// 按名称限制可用工具（`None` = 继承父级全部工具）。
    pub tool_filter: Option<Vec<String>>,
    /** 最大 Agent 迭代次数。`None` 会保留具体工厂自身的有界默认值；
     * `ReactAgentBuilder` 会拒绝显式传入的零值。
     */
    pub max_iterations: Option<usize>,
    /// 该子 Agent 的 token 限制（`None` = 使用默认值）。
    pub token_limit: Option<usize>,
    /** 从父级继承多少条最近消息。
     * `None` = 不继承，`Some(0)` = 继承全部，`Some(n)` = 最近 n 条消息。
     */
    pub inherit_history: Option<usize>,
    /// 是否继承父级的记忆存储。
    pub inherit_memory: bool,
    /// 超时时间（秒），`0` 表示不设超时。
    pub timeout_secs: u64,
    /// 该子 Agent 自身是否可以再向下派发更多子 Agent。
    pub can_delegate: bool,
    /// 用于发现 / 过滤的标签。
    pub tags: Vec<String>,
    /** 该子 Agent 是否使用轻量级（基础设施共享）模式。
     *
     * 为 `true` 时，子 Agent 会共享父级的 LLM 客户端、ToolManager
     * 和 GuardManager，而不是创建新实例。
     */
    pub lightweight: bool,
    /// 由注入的 provider 解析出的、可选的产品所属隔离种类。
    pub isolation: Option<String>,
    /// 由权威任务运行时编译的声明式 Team 意图。
    pub team: Option<super::team::TeamSpec>,
    /** 为 `true` 时，该角色偏好后台派发（第二阶段会进行异步调度）。
     * 第一阶段仅存储来自 frontmatter 的该标志位。
     */
    pub is_background: bool,
}

impl SubagentDefinition {
    /** 创建一个最小化的 Sync 模式内置定义。
     *
     * # 参数
     * * `name` - 用于发现和派发的唯一名称。
     * * `description` - 暴露给 LLM 的人类可读描述。
     */
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            kind: SubagentKind::BuiltIn,
            access_mode: SubagentAccessMode::Write,
            execution_mode: ExecutionMode::Sync,
            model: None,
            thinking: None,
            system_prompt: None,
            tool_filter: None,
            max_iterations: None,
            token_limit: None,
            inherit_history: None,
            inherit_memory: false,
            timeout_secs: 0,
            can_delegate: false,
            tags: Vec::new(),
            lightweight: false,
            isolation: None,
            team: None,
            is_background: false,
        }
    }

    /** 便捷方法：为向后兼容的注册场景创建一个 Sync 模式定义。
     *
     * # 参数
     * * `name` - 用于发现和派发的唯一名称。
     *
     * # 说明
     * 会自动生成形如 "Subagent {name}" 的描述文本。
     */
    pub fn simple_sync(name: impl Into<String>) -> Self {
        let name = name.into();
        Self::new(name.clone(), format!("Subagent {}", name))
    }
}

// ── 子 Agent 结果 ───────────────────────────────────────────────────────────

/// 当不存在结构化结果时，面向父级的摘要文本的默认字符预算。
const DEFAULT_SUMMARY_CHARS: usize = 1200;
/// 当前的子 Agent 结果契约版本号。
pub const SUBAGENT_RESULT_CONTRACT_VERSION: u32 = 2;
/// 结果列表（如 remaining_work）允许保留的最大条目数。
const MAX_RESULT_ITEMS: usize = 64;
/// 单条明细文本允许的最大字符数。
const MAX_DETAIL_CHARS: usize = 500;
/// 路径字符串允许的最大字符数。
const MAX_PATH_CHARS: usize = 2048;
/// kind 类字段允许的最大字符数。
const MAX_KIND_CHARS: usize = 80;
/// evidence 的 attributes 字段序列化后允许的最大字符数。
const MAX_ATTRIBUTES_CHARS: usize = 4_000;

/// 由运行时权威判定的、某次子 Agent 派发的终态状态。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    /// 正在运行中。
    Running,
    /// 已成功完成。
    Completed,
    /// 已失败。
    #[default]
    Failed,
    /// 已被取消。
    Cancelled,
    /// 已超时。
    TimedOut,
}

impl SubagentStatus {
    /// 返回该状态对应的稳定字符串表示。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
        }
    }
}

impl std::str::FromStr for SubagentStatus {
    type Err = String;

    /// 解析运行时投影所使用的稳定序列化拼写。
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "timed_out" => Ok(Self::TimedOut),
            _ => Err(format!("unknown Subagent status: {value}")),
        }
    }
}

/// 子 Agent 返回的单个产物（artifact）的可用性与完整性事实。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentArtifact {
    /// 产物路径。
    pub path: String,
    /// 产物种类（由产品层定义）。
    #[serde(default)]
    pub kind: String,
    /// 产物大小（字节），未知时为 `None`。
    #[serde(default)]
    pub bytes: Option<u64>,
    /// 产物内容的 SHA-256 哈希，未知时为 `None`。
    #[serde(default)]
    pub sha256: Option<String>,
    /// 产生该产物的执行 id，未知时为 `None`。
    #[serde(default)]
    pub producer_execution_id: Option<String>,
    /// 该产物当前是否可用（已被运行时确认存在）。
    #[serde(default)]
    pub available: bool,
}

/// 证据（evidence）是来自观察到的运行时行为，还是来自子 Agent 的输出。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentEvidenceSource {
    /// 由运行时观察到（更可信）。
    Observed,
    /// 由子 Agent 自行报告。
    #[default]
    Reported,
}

/// 在一次子 Agent 派发过程中产生或被观察到的、产品中立的证据。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentEvidence {
    /// 证据种类（由产品层定义）。
    pub kind: String,
    /// 被观察 / 报告的对象。
    pub subject: String,
    /// 可选的、有界长度的结果描述。
    #[serde(default)]
    pub outcome: Option<String>,
    /// 有界长度的证据明细。
    #[serde(default)]
    pub details: String,
    /// 该证据的来源（观察 or 报告）。
    #[serde(default)]
    pub source: SubagentEvidenceSource,
    /// 附加属性，结构由产品层自行约定。
    #[serde(default)]
    pub attributes: serde_json::Value,
}

/// 从结构化子 Agent 证据中派生出的验证检查项。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentVerification {
    /// 该验证检查的名称。
    pub check: String,
    /// 该验证检查的结果状态。
    pub status: SubagentVerificationStatus,
    /// 有界长度的验证明细。
    pub details: String,
    /// 该验证结果的来源（观察 or 报告）。
    pub source: SubagentEvidenceSource,
}

/// 由子 Agent 观察到或报告的验证检查的结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentVerificationStatus {
    /// 通过。
    Passed,
    /// 未通过。
    Failed,
    /// 未运行。
    NotRun,
}

/// 由观察到的子 Agent 工具证据所引用的文件。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentTouchedFiles {
    /// 被读取的文件路径列表。
    pub read: Vec<String>,
    /// 被写入的文件路径列表。
    pub written: Vec<String>,
}

/// 面向父级的、可序列化的子 Agent 派发结果契约。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentOutcome {
    /// 显式 JSON 契约为 `1`；旧版文本回退为 `0`。
    pub contract_version: u32,
    /// 由运行时权威判定的终态状态。模型自行提供的状态一律被忽略。
    pub status: SubagentStatus,
    /// 面向父级的摘要文本。
    pub summary: String,
    /// 该次派发产生的产物列表。
    #[serde(default)]
    pub artifacts: Vec<SubagentArtifact>,
    /// 该次派发产生的证据列表。
    #[serde(default)]
    pub evidence: Vec<SubagentEvidence>,
    /// 从通用证据流中派生出的验证视图。
    #[serde(default)]
    pub verification: Vec<SubagentVerification>,
    /// 尚未完成的剩余工作项。
    #[serde(default)]
    pub remaining_work: Vec<String>,
    /// 从观察到的工具证据中派生出的读写路径。
    #[serde(default)]
    pub touched_files: SubagentTouchedFiles,
}

impl Default for SubagentOutcome {
    fn default() -> Self {
        Self {
            contract_version: 0,
            status: SubagentStatus::Failed,
            summary: String::new(),
            artifacts: Vec::new(),
            evidence: Vec::new(),
            verification: Vec::new(),
            remaining_work: Vec::new(),
            touched_files: SubagentTouchedFiles::default(),
        }
    }
}

impl SubagentOutcome {
    /// 构造一个终态结果：裁剪摘要与剩余工作项长度，其余字段留空。
    pub fn terminal(
        status: SubagentStatus,
        summary: impl Into<String>,
        remaining_work: Vec<String>,
    ) -> Self {
        let summary = bounded_text(summary.into().trim(), DEFAULT_SUMMARY_CHARS);
        let remaining_work = bounded_unique(remaining_work, MAX_RESULT_ITEMS, MAX_DETAIL_CHARS);
        Self {
            contract_version: SUBAGENT_RESULT_CONTRACT_VERSION,
            status,
            summary,
            artifacts: Vec::new(),
            evidence: Vec::new(),
            verification: Vec::new(),
            remaining_work,
            touched_files: SubagentTouchedFiles::default(),
        }
    }

    /// 判断该结果是否为已成功完成状态。
    pub fn is_completed(&self) -> bool {
        self.status == SubagentStatus::Completed
    }

    /// 根据证据刷新通用的验证视图与文件访问视图。
    pub fn refresh_derived_views(&mut self) {
        self.verification = self
            .evidence
            .iter()
            .filter_map(project_verification_evidence)
            .collect();
        let mut touched = SubagentTouchedFiles::default();
        for evidence in &self.evidence {
            let Some((write, path)) = project_tool_file_access(evidence) else {
                continue;
            };
            let target = if write {
                &mut touched.written
            } else {
                &mut touched.read
            };
            if !target.iter().any(|existing| existing == &path) {
                target.push(path);
            }
        }
        self.touched_files = touched;
    }

    /// 借出派生出的验证视图，而不重新构建它。
    pub fn verification(&self) -> &[SubagentVerification] {
        &self.verification
    }

    /// 借出派生出的读写文件视图，而不重新构建它。
    pub fn touched_files(&self) -> &SubagentTouchedFiles {
        &self.touched_files
    }
}

/// 从一条证据中投影出一个验证检查项（若该证据可解释为验证检查的话）。
fn project_verification_evidence(evidence: &SubagentEvidence) -> Option<SubagentVerification> {
    let check = if evidence.kind == "verification" {
        Some(evidence.subject.clone())
    } else if evidence.kind == "tool_result" {
        evidence
            .attributes
            .get("args")
            .and_then(|args| verification_check_from_tool(&evidence.subject, args))
    } else {
        None
    }?;
    Some(SubagentVerification {
        check,
        status: match evidence.outcome.as_deref() {
            Some("passed" | "succeeded") => SubagentVerificationStatus::Passed,
            Some("failed") => SubagentVerificationStatus::Failed,
            _ => SubagentVerificationStatus::NotRun,
        },
        details: evidence.details.clone(),
        source: evidence.source,
    })
}

/// 若工具名属于命令执行类工具，则从其参数中提取被执行的命令文本。
fn verification_check_from_tool(name: &str, args: &serde_json::Value) -> Option<String> {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    if !matches!(
        normalized.as_str(),
        "shell" | "bash" | "terminal" | "run_code" | "execute_command"
    ) {
        return None;
    }
    ["command", "cmd", "code", "script"]
        .iter()
        .find_map(|key| args.get(*key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// 从一条证据中投影出被读写的文件路径（若该证据可解释为文件访问的话）。
fn project_tool_file_access(evidence: &SubagentEvidence) -> Option<(bool, String)> {
    let normalized = evidence.subject.to_ascii_lowercase().replace('-', "_");
    let write = normalized.contains("write")
        || normalized.contains("edit")
        || normalized.contains("delete")
        || normalized.contains("patch");
    let read = normalized.contains("read")
        || normalized.contains("search")
        || normalized.contains("glob")
        || normalized.contains("grep");
    if !write && !read {
        return None;
    }
    let args = evidence.attributes.get("args")?;
    ["path", "file_path", "target", "directory"]
        .iter()
        .find_map(|key| args.get(*key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|path| (write, path.to_string()))
}

/** 渲染供 [`parse_subagent_outcome`] 解析的、面向模型的权威结果契约。
 * 产品层可以在此区块之前追加可选章节，但最终的 `## Result` 信封部分
 * 始终由框架自身负责。
 */
pub fn render_result_contract() -> String {
    format!(
        "## Result\nEnd the response with exactly one fenced JSON object using this shape:\n\
```json\n\
{{\"contract_version\":{SUBAGENT_RESULT_CONTRACT_VERSION},\"status\":\"completed\",\
\"summary\":\"at most 1200 characters\",\"artifacts\":[{{\"path\":\"actual path\",\
\"kind\":\"product-defined kind\"}}],\"evidence\":[{{\"kind\":\"product-defined kind\",\
\"subject\":\"what was observed\",\"outcome\":\"optional bounded outcome\",\
\"details\":\"bounded evidence\",\"source\":\"reported\",\"attributes\":{{}}}}],\"remaining_work\":[]}}\n\
```\n\
Runtime owns terminal status and observed evidence. Report only real artifacts and evidence; put incomplete or blocked work in remaining_work."
    )
}

/** 解析嵌入在子 Agent 响应中的、围栏包裹或独立存在的 JSON 对象。
 * 产品层可以检查其中命名的可选字段，而终态的 `## Result` 对象仍由
 * [`parse_subagent_outcome`] 负责解析。
 */
pub fn parse_json_objects(raw: &str) -> Vec<serde_json::Value> {
    let mut objects = Vec::new();
    for block in raw.split("```json").skip(1) {
        if let Some(json) = block.split("```").next()
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(json.trim())
            && value.is_object()
        {
            objects.push(value);
        }
    }
    let trimmed = raw.trim();
    if trimmed.starts_with('{')
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
        && value.is_object()
    {
        objects.push(value);
    }
    objects
}

/// 从 `## Result` JSON 块中反序列化出的、模型自行报告的原始结果。
#[derive(Debug, Deserialize)]
struct ReportedSubagentOutcome {
    /// 模型自行报告的契约版本号。
    #[serde(default)]
    contract_version: u32,
    /// 模型自行报告的摘要文本。
    summary: String,
    /// 模型自行报告的产物列表。
    #[serde(default)]
    artifacts: Vec<SubagentArtifact>,
    /// 模型自行报告的证据列表。
    #[serde(default)]
    evidence: Vec<SubagentEvidence>,
    /// 模型自行报告的剩余工作项。
    #[serde(default)]
    remaining_work: Vec<String>,
}

/** 将子 Agent 的原始输出拆分为面向父级的摘要和产物路径列表。
 *
 * 查找 `## Summary` 和 `## Artifacts` 这两个 markdown 标题。如果没有
 * Summary 标题，则回退为对全文做 UTF-8 安全截断（对多字节字符绝不
 * panic）。
 */
pub fn split_subagent_output(raw: &str) -> (String, Vec<String>) {
    let summary = extract_markdown_section(raw, "Summary")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| raw.chars().take(DEFAULT_SUMMARY_CHARS).collect());
    let artifacts = extract_markdown_section(raw, "Artifacts")
        .map(|section| {
            section
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    let item = trimmed
                        .strip_prefix("- ")
                        .or_else(|| trimmed.strip_prefix("* "))
                        .unwrap_or(trimmed)
                        .trim();
                    if item.is_empty() {
                        None
                    } else {
                        Some(item.to_string())
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    (summary, artifacts)
}

/** 解析显式的 `## Result` JSON 契约，失败时回退为有界长度的摘要。
 *
 * 终态状态由调用方提供。模型无法通过返回不同的 JSON 值，把一次失败、
 * 被取消或超时的执行伪装成 `completed`。
 */
pub fn parse_subagent_outcome(
    raw: &str,
    status: SubagentStatus,
    _execution_id: Option<&str>,
    _working_dir: Option<&Path>,
) -> SubagentOutcome {
    let reported = extract_markdown_section(raw, "Result")
        .and_then(extract_fenced_json)
        .and_then(|json| serde_json::from_str::<ReportedSubagentOutcome>(json).ok())
        .filter(|result| result.contract_version == SUBAGENT_RESULT_CONTRACT_VERSION);

    let mut outcome = if let Some(reported) = reported {
        SubagentOutcome {
            contract_version: SUBAGENT_RESULT_CONTRACT_VERSION,
            status,
            summary: reported
                .summary
                .trim()
                .chars()
                .take(DEFAULT_SUMMARY_CHARS)
                .collect(),
            artifacts: reported.artifacts,
            evidence: reported
                .evidence
                .into_iter()
                .map(|mut evidence| {
                    // 只有运行时观察到的工具事件才能产生“已观察”类型的证据。
                    evidence.source = SubagentEvidenceSource::Reported;
                    evidence
                })
                .collect(),
            verification: Vec::new(),
            remaining_work: bounded_unique(
                reported.remaining_work,
                MAX_RESULT_ITEMS,
                MAX_DETAIL_CHARS,
            ),
            touched_files: SubagentTouchedFiles::default(),
        }
    } else {
        let (summary, artifact_paths) = split_subagent_output(raw);
        SubagentOutcome {
            contract_version: 0,
            status,
            summary,
            artifacts: artifact_paths
                .into_iter()
                .map(|path| SubagentArtifact {
                    path,
                    kind: String::new(),
                    bytes: None,
                    sha256: None,
                    producer_execution_id: None,
                    available: false,
                })
                .collect(),
            evidence: Vec::new(),
            verification: Vec::new(),
            remaining_work: Vec::new(),
            touched_files: SubagentTouchedFiles::default(),
        }
    };

    normalize_outcome(&mut outcome);
    outcome.refresh_derived_views();
    for artifact in &mut outcome.artifacts {
        artifact.bytes = None;
        artifact.sha256 = None;
        artifact.producer_execution_id = None;
        artifact.available = false;
    }
    outcome
}

/// 从一段文本中提取 ` ```json ` 围栏代码块内的内容。
fn extract_fenced_json(section: &str) -> Option<&str> {
    let marker = "```json";
    let start = section.find(marker)?.saturating_add(marker.len());
    let after = section.get(start..)?;
    let end = after.find("```")?;
    after.get(..end).map(str::trim)
}

/// 对字符串列表去重、裁剪单条长度并限制总条目数。
fn bounded_unique(values: Vec<String>, max_items: usize, max_chars: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    values
        .into_iter()
        .map(|value| bounded_text(value.trim(), max_chars))
        .filter(|value| !value.is_empty() && seen.insert(value.clone()))
        .take(max_items)
        .collect()
}

/// 按字符数（而非字节数）截断文本，保证 UTF-8 安全。
fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

/// 只保留列表中最新的 `max_items` 项，丢弃更早的多余项。
fn keep_latest<T>(values: &mut Vec<T>, max_items: usize) {
    let excess = values.len().saturating_sub(max_items);
    if excess > 0 {
        values.drain(..excess);
    }
}

/// 对结果的各个字段做统一的长度裁剪与空值过滤，使其满足契约边界。
pub(crate) fn normalize_outcome(outcome: &mut SubagentOutcome) {
    outcome.summary = bounded_text(outcome.summary.trim(), DEFAULT_SUMMARY_CHARS);
    keep_latest(&mut outcome.artifacts, MAX_RESULT_ITEMS);
    for artifact in &mut outcome.artifacts {
        artifact.path = bounded_text(artifact.path.trim(), MAX_PATH_CHARS);
        artifact.kind = bounded_text(artifact.kind.trim(), MAX_KIND_CHARS);
    }
    outcome
        .artifacts
        .retain(|artifact| !artifact.path.is_empty());

    keep_latest(&mut outcome.evidence, MAX_RESULT_ITEMS);
    for evidence in &mut outcome.evidence {
        evidence.kind = bounded_text(evidence.kind.trim(), MAX_KIND_CHARS);
        evidence.subject = bounded_text(evidence.subject.trim(), MAX_PATH_CHARS);
        evidence.outcome = evidence
            .outcome
            .take()
            .map(|outcome| bounded_text(outcome.trim(), MAX_DETAIL_CHARS))
            .filter(|outcome| !outcome.is_empty());
        evidence.details = bounded_text(evidence.details.trim(), MAX_DETAIL_CHARS);
        if evidence.attributes.to_string().chars().count() > MAX_ATTRIBUTES_CHARS {
            evidence.attributes = serde_json::json!({ "truncated": true });
        }
    }
    outcome
        .evidence
        .retain(|evidence| !evidence.kind.is_empty() && !evidence.subject.is_empty());

    outcome.remaining_work = bounded_unique(
        std::mem::take(&mut outcome.remaining_work),
        MAX_RESULT_ITEMS,
        MAX_DETAIL_CHARS,
    );
}

/// 提取指定 `## <heading>` 标题下、到下一个同级或更高级标题之前的正文内容。
fn extract_markdown_section<'a>(raw: &'a str, heading: &str) -> Option<&'a str> {
    let needle = format!("## {heading}");
    let start = raw.find(&needle)?;
    let after = raw.get(start.saturating_add(needle.len())..)?;
    let body_start = after.find('\n').map(|i| i.saturating_add(1)).unwrap_or(0);
    let body = after.get(body_start..)?;
    let end = body
        .find("\n## ")
        .or_else(|| body.find("\n# "))
        .unwrap_or(body.len());
    body.get(..end)
}

/// 一次子 Agent 执行返回的结果。
#[derive(Debug, Clone)]
pub struct SubagentResult {
    /// 产生该结果的 Agent 名称。
    pub agent_name: String,
    /// 最终输出文本（供 UI / 存储使用的完整明细）。
    pub output: String,
    /// 由运行时权威判定的结构化终态结果。
    pub outcome: SubagentOutcome,
    /// 执行耗时。
    pub duration: Duration,
    /// 使用的迭代次数。
    pub iterations: usize,
    /// token 用量（如果可获取）。
    pub tokens_used: Option<usize>,
    /// 输出是否因为 token 限制而被截断。
    pub was_truncated: bool,
    /// 实际使用的执行模式。
    pub mode: ExecutionMode,
    /// 在模型执行前实际建立的隔离边界。
    pub isolation_observed: ObservedIsolation,
    /** 本次派发中所有调用累计的 LLM 用量。
     * 当该 Agent 没有产生任何 `LlmUsage` 事件时为 `None`（例如在首次
     * LLM 调用之前就被取消，或 provider 从未返回用量信息）。
     */
    pub llm_usage: Option<LlmUsageStats>,
}

impl SubagentResult {
    /// 返回用于持久化或上报的稳定用量事实。
    pub fn usage(&self) -> ExecutionUsage {
        let duration_ms = u64::try_from(self.duration.as_millis()).unwrap_or(u64::MAX);
        let iterations = u64::try_from(self.iterations).unwrap_or(u64::MAX);
        ExecutionUsage {
            duration_ms: Some(duration_ms),
            tokens_used: self
                .llm_usage
                .as_ref()
                .map(|usage| usage.prompt_tokens.saturating_add(usage.completion_tokens)),
            iterations: Some(iterations),
        }
    }

    /** 创建一个同步（阻塞）子 Agent 执行的结果。
     *
     * # 参数
     * * `agent_name` - 产生该结果的 Agent 名称。
     * * `output` - 该 Agent 的最终输出文本。
     * * `duration` - 执行耗时。
     */
    pub fn sync_result(agent_name: &str, output: String, duration: Duration) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            output,
            outcome: SubagentOutcome {
                status: SubagentStatus::Completed,
                ..SubagentOutcome::default()
            },
            duration,
            iterations: 1,
            tokens_used: None,
            was_truncated: false,
            mode: ExecutionMode::Sync,
            isolation_observed: ObservedIsolation::default(),
            llm_usage: None,
        }
        .with_structured(None, None)
    }

    /** 创建一个 fork（非阻塞）子 Agent 执行的结果。
     *
     * # 参数
     * * `agent_name` - 产生该结果的 Agent 名称。
     * * `output` - 该 Agent 的最终输出文本。
     * * `duration` - 执行耗时。
     * * `iterations` - 该 Agent 使用的迭代次数。
     */
    pub fn fork_result(
        agent_name: &str,
        output: String,
        duration: Duration,
        iterations: usize,
    ) -> Self {
        Self {
            agent_name: agent_name.to_string(),
            output,
            outcome: SubagentOutcome {
                status: SubagentStatus::Completed,
                ..SubagentOutcome::default()
            },
            duration,
            iterations,
            tokens_used: None,
            was_truncated: false,
            mode: ExecutionMode::Fork,
            isolation_observed: ObservedIsolation::default(),
            llm_usage: None,
        }
        .with_structured(None, None)
    }

    /// 根据显式结果契约（或回退文本）填充结构化的终态结果。
    pub fn with_structured(
        mut self,
        execution_id: Option<&str>,
        working_dir: Option<&Path>,
    ) -> Self {
        let status = self.outcome.status;
        self.outcome = parse_subagent_outcome(&self.output, status, execution_id, working_dir);
        self
    }

    /// 创建一个被取消的子 Agent 执行结果。
    pub fn cancelled(
        agent_name: impl Into<String>,
        output: impl Into<String>,
        mode: ExecutionMode,
    ) -> Self {
        let output = output.into();
        Self {
            agent_name: agent_name.into(),
            outcome: SubagentOutcome::terminal(
                SubagentStatus::Cancelled,
                output.clone(),
                vec![output.clone()],
            ),
            output,
            duration: Duration::ZERO,
            iterations: 0,
            tokens_used: None,
            was_truncated: false,
            mode,
            isolation_observed: ObservedIsolation::default(),
            llm_usage: None,
        }
    }
}

// ── 已注册子 Agent（视图类型） ──────────────────────────────────────────

/// 一个已注册子 Agent 的只读快照。
#[derive(Debug, Clone)]
pub struct RegisteredSubagent {
    /// 子 Agent 定义。
    pub definition: SubagentDefinition,
    /// Agent 实例当前是否可用（通过工厂或预先构建）。
    pub has_instance: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::subagent::{TeamConfig, TeamSpec, TeamStrategy};

    #[test]
    fn test_execution_mode_default() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Sync);
    }

    #[test]
    fn test_execution_mode_display() {
        assert_eq!(ExecutionMode::Sync.to_string(), "sync");
        assert_eq!(ExecutionMode::Fork.to_string(), "fork");
        assert_eq!(ExecutionMode::Teammate.to_string(), "teammate");
        assert_eq!(ExecutionMode::Team.to_string(), "team");

        let spec = TeamSpec {
            strategy: TeamStrategy::ManagerSubagent,
            manager: "planner".to_string(),
            subagents: vec!["researcher".to_string()],
            config: TeamConfig::default(),
        };
        let definition = SubagentDefinition {
            team: Some(spec),
            execution_mode: ExecutionMode::Team,
            ..SubagentDefinition::new("review-team", "Review Team")
        };
        assert!(definition.team.is_some());
    }

    #[test]
    fn split_output_prefers_summary_heading() {
        let raw = "## Summary\n短结论\n\n## Artifacts\n- src/a.rs\n- docs/b.md\n\n## Notes\n细节";
        let (summary, artifacts) = split_subagent_output(raw);
        assert_eq!(summary, "短结论");
        assert_eq!(
            artifacts,
            vec!["src/a.rs".to_string(), "docs/b.md".to_string()]
        );
    }

    #[test]
    fn split_output_truncates_utf8_safely_without_heading() {
        let raw: String = "中文"
            .chars()
            .chain(std::iter::repeat_n('x', 2000))
            .collect();
        let (summary, _) = split_subagent_output(&raw);
        assert_eq!(summary.chars().count(), DEFAULT_SUMMARY_CHARS);
        // 不能 panic，且必须以这段多字节前缀开头。
        assert!(summary.starts_with("中文"));
    }

    #[test]
    fn test_subagent_definition_new() {
        let def = SubagentDefinition::new("researcher", "Researches topics");
        assert_eq!(def.name, "researcher");
        assert_eq!(def.execution_mode, ExecutionMode::Sync);
        assert!(matches!(def.kind, SubagentKind::BuiltIn));
        assert!(def.inherit_history.is_none());
    }

    #[test]
    fn test_simple_sync() {
        let def = SubagentDefinition::simple_sync("subagent");
        assert_eq!(def.name, "subagent");
        assert!(def.description.contains("subagent"));
    }

    #[test]
    fn test_subagent_result_sync() {
        let mut result = SubagentResult::sync_result("a", "ok".into(), Duration::from_millis(100));
        result.iterations = 3;
        result.llm_usage = Some(LlmUsageStats {
            prompt_tokens: 7,
            completion_tokens: 5,
            ..LlmUsageStats::default()
        });
        assert_eq!(result.mode, ExecutionMode::Sync);
        assert_eq!(result.output, "ok");
        assert_eq!(result.outcome.status, SubagentStatus::Completed);
        assert_eq!(
            result.usage(),
            ExecutionUsage {
                duration_ms: Some(100),
                tokens_used: Some(12),
                iterations: Some(3),
            }
        );
    }

    #[test]
    fn structured_result_does_not_attest_model_reported_artifact() -> Result<(), String> {
        let dir = tempfile::tempdir().map_err(|error| error.to_string())?;
        let artifact_path = dir.path().join("report.txt");
        std::fs::write(&artifact_path, "result").map_err(|error| error.to_string())?;
        let raw = "## Result\n```json\n{\"contract_version\":2,\"status\":\"completed\",\"summary\":\"done\",\"artifacts\":[{\"path\":\"report.txt\",\"kind\":\"report\"}],\"evidence\":[{\"kind\":\"verification\",\"subject\":\"cargo test\",\"outcome\":\"passed\",\"source\":\"observed\"}],\"remaining_work\":[]}\n```";
        let outcome = parse_subagent_outcome(
            raw,
            SubagentStatus::TimedOut,
            Some("task-1:1"),
            Some(dir.path()),
        );
        assert_eq!(outcome.status, SubagentStatus::TimedOut);
        assert_eq!(outcome.contract_version, 2);
        let artifact = outcome
            .artifacts
            .first()
            .ok_or_else(|| "artifact missing".to_string())?;
        assert!(!artifact.available);
        assert!(artifact.sha256.is_none());
        assert!(artifact.producer_execution_id.is_none());
        assert!(matches!(
            outcome.evidence.first(),
            Some(SubagentEvidence {
                source: SubagentEvidenceSource::Reported,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn structured_outcome_exposes_generic_evidence_views() -> Result<(), String> {
        let raw = "## Result\n```json\n{\"contract_version\":2,\"status\":\"completed\",\"summary\":\"done\",\"artifacts\":[],\"evidence\":[{\"kind\":\"verification\",\"subject\":\"cargo check\",\"outcome\":\"passed\",\"details\":\"ok\"},{\"kind\":\"tool_result\",\"subject\":\"write_file\",\"outcome\":\"succeeded\",\"attributes\":{\"args\":{\"path\":\"src/lib.rs\"}}}],\"remaining_work\":[]}\n```";
        let outcome = parse_subagent_outcome(raw, SubagentStatus::Completed, None, None);
        assert_eq!(outcome.verification().len(), 1);
        assert_eq!(outcome.touched_files().written, vec!["src/lib.rs"]);
        Ok(())
    }

    #[test]
    fn structured_result_bounds_utf8_evidence() -> Result<(), String> {
        let long = "中".repeat(600);
        let remaining_work: Vec<String> = (0..70).map(|index| format!("{index}-{long}")).collect();
        let payload = serde_json::json!({
            "contract_version": 2,
            "status": "completed",
            "summary": long.repeat(3),
            "artifacts": [],
            "evidence": [],
            "remaining_work": remaining_work,
        });
        let raw = format!("## Result\n```json\n{payload}\n```");
        let outcome = parse_subagent_outcome(&raw, SubagentStatus::Completed, None, None);

        assert_eq!(outcome.summary.chars().count(), DEFAULT_SUMMARY_CHARS);
        assert_eq!(outcome.remaining_work.len(), MAX_RESULT_ITEMS);
        assert!(
            outcome
                .remaining_work
                .iter()
                .all(|item| item.chars().count() <= MAX_DETAIL_CHARS)
        );
        Ok(())
    }

    #[test]
    fn rendered_result_contract_round_trips_through_parser() {
        let rendered = render_result_contract();
        let outcome = parse_subagent_outcome(
            &rendered,
            SubagentStatus::Completed,
            Some("round-trip:1"),
            None,
        );

        assert_eq!(outcome.contract_version, SUBAGENT_RESULT_CONTRACT_VERSION);
        assert_eq!(outcome.status, SubagentStatus::Completed);
        assert_eq!(outcome.summary, "at most 1200 characters");
        assert!(matches!(
            outcome.evidence.first(),
            Some(SubagentEvidence {
                source: SubagentEvidenceSource::Reported,
                ..
            })
        ));
    }
}
