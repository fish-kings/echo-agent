/*! 面向 skill 脚本的轻量级依赖探测器。

从 [`SkillDescriptor`] 的 frontmatter 中读取 `metadata.requires-binaries` 和
`metadata.requires-python-packages`，生成结构化的 [`ProbeReport`]。系统二进制
通过 `which` 探测；Python 包仅作文档记录（运行时由 uv + PEP 723 处理）。

从不自动安装任何东西 —— 仅探测并给出建议。
*/

use crate::skills::external::types::SkillDescriptor;

/// 外部依赖的种类。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepKind {
    /// 系统二进制程序（例如 `soffice`、`pdftoppm`）。
    Binary,
    /// Python 包（由 uv + PEP 723 处理；仅作文档记录）。
    PythonPkg,
    /// Node.js 模块。
    NodeModule,
}

/// skill 声明的单个依赖项。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillDependency {
    /// 依赖的种类。
    pub kind: DepKind,
    /// 依赖的名称（二进制名或包名）。
    pub name: String,
    /// `true` 表示缺失该依赖时 skill 无法使用；`false` 表示仅是某项可选功能会降级。
    pub required: bool,
    /// 可读的安装提示（例如 "brew install --cask libreoffice"）。
    pub install_hint: String,
}

/// 探测某个 skill 声明的全部依赖后得到的结果。
#[derive(Debug, Clone, Default)]
pub struct ProbeReport {
    /// 系统中已找到的依赖名称。
    pub satisfied: Vec<String>,
    /// 缺失的必需依赖（skill 无法工作）。
    pub missing_required: Vec<SkillDependency>,
    /// 缺失的可选依赖（功能会降级）。
    pub missing_optional: Vec<SkillDependency>,
}

impl ProbeReport {
    /// 当所有必需依赖都已满足时返回 `true`。
    pub fn is_ok(&self) -> bool {
        self.missing_required.is_empty()
    }
}

/** 将 metadata 中逗号分隔或类似 JSON 数组的字符串值解析为独立条目列表。

frontmatter 的 `metadata:` 值存储为 `HashMap<String, String>`，因此像
`[soffice, pdftoppm]` 这样的 YAML 列表会被读成字符串 `"[soffice, pdftoppm]"`。
*/
fn parse_metadata_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| {
            s.trim()
                .trim_matches(|c: char| c == '[' || c == ']' || c == '"' || c == '\'')
        })
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/** 从 [`SkillDescriptor`] 的 frontmatter metadata 中提取声明的依赖。

读取 metadata map 中的 `requires-binaries` 和 `requires-python-packages`
两个键（均以逗号分隔的字符串形式存储）。
*/
pub fn extract_dependencies(desc: &SkillDescriptor) -> Vec<SkillDependency> {
    let mut deps = Vec::new();

    if let Some(bins_raw) = desc.metadata.get("requires-binaries") {
        for name in parse_metadata_list(bins_raw) {
            deps.push(SkillDependency {
                kind: DepKind::Binary,
                install_hint: install_hint_for_binary(&name),
                name,
                required: true,
            });
        }
    }

    if let Some(pkgs_raw) = desc.metadata.get("requires-python-packages") {
        for name in parse_metadata_list(pkgs_raw) {
            deps.push(SkillDependency {
                kind: DepKind::PythonPkg,
                install_hint: format!("uv run 自动处理: {}", name),
                name,
                required: false, // 由 uv + PEP 723 处理
            });
        }
    }

    deps
}

/** 探测单个二进制是否在 PATH 上（通过 `which` 子进程，不引入 which crate）。

失败时（例如没有 which 命令）返回 false —— 偏保守处理，把声明的二进制当作
缺失显示，这样用户至少能看到提示，而不是静默通过。
*/
fn binary_available(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/** 探测一个 skill 声明的全部必需二进制，返回**缺失**的二进制名称列表。

供 SkillsHub scan 调用，结果存入 SkillHubEntry.missing_dependencies，
前端据此显示 ⚠️ 提示。不探测 Python 包（由 uv + PEP 723 在运行时处理）。
*/
pub fn missing_binary_names(desc: &SkillDescriptor) -> Vec<String> {
    extract_dependencies(desc)
        .into_iter()
        .filter(|d| d.required && matches!(d.kind, DepKind::Binary) && !binary_available(&d.name))
        .map(|d| d.name)
        .collect()
}

/// 为常见的二进制程序返回可读的安装提示。
fn install_hint_for_binary(bin: &str) -> String {
    match bin {
        "soffice" => "brew install --cask libreoffice".into(),
        "pdftoppm" => "brew install poppler".into(),
        "ffmpeg" => "brew install ffmpeg".into(),
        "sqlite3" => "pre-installed on macOS".into(),
        "git" => "pre-installed on macOS".into(),
        _ => format!("请自行安装: {}", bin),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::external::types::SkillSandboxPolicy;
    use std::collections::HashMap;

    /// 构造一个仅指定 metadata、其余字段取默认值的测试用 SkillDescriptor。
    fn desc_with_metadata(md: HashMap<String, String>) -> SkillDescriptor {
        SkillDescriptor {
            source: None,
            name: "test-skill".into(),
            description: "test".into(),
            license: None,
            compatibility: None,
            metadata: md,
            allowed_tools: vec![],
            shell: None,
            paths: vec![],
            triggers: vec![],
            hooks: None,
            sandbox: Some(SkillSandboxPolicy::default()),
            depends_on: vec![],
            location: std::path::PathBuf::from("/tmp/test-skill"),
        }
    }

    // metadata 为空时应提取出空依赖列表。
    #[test]
    fn extract_empty_for_no_metadata() {
        let desc = desc_with_metadata(HashMap::new());
        let deps = extract_dependencies(&desc);
        assert!(deps.is_empty());
    }

    // 应从 frontmatter 的 requires-binaries 中正确解析出二进制依赖。
    #[test]
    fn extract_binaries_from_frontmatter() {
        let mut md = HashMap::new();
        md.insert("requires-binaries".into(), "soffice, pdftoppm".into());
        let desc = desc_with_metadata(md);
        let deps = extract_dependencies(&desc);
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0].name, "soffice");
        assert_eq!(deps[0].kind, DepKind::Binary);
        assert!(deps[0].required);
        assert_eq!(deps[1].name, "pdftoppm");
    }

    // Python 包依赖应被标记为可选（非必需）。
    #[test]
    fn extract_python_packages_as_optional() {
        let mut md = HashMap::new();
        md.insert("requires-python-packages".into(), "defusedxml".into());
        let desc = desc_with_metadata(md);
        let deps = extract_dependencies(&desc);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].name, "defusedxml");
        assert_eq!(deps[0].kind, DepKind::PythonPkg);
        assert!(
            !deps[0].required,
            "python packages are handled by uv + PEP 723; documentation-only"
        );
    }

    // ProbeReport 的默认值（无缺失依赖）应判定为 is_ok。
    #[test]
    fn probe_report_default_is_ok() {
        let report = ProbeReport::default();
        assert!(report.is_ok());
    }
}
