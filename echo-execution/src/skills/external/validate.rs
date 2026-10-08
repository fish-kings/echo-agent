/*!
技能目录校验 —— echo-agent 中对应 `skills-ref validate` 的实现。

[agentskills.io 规范](https://agentskills.io/specification)
自带一个参考校验器（`skills-ref validate`）。本模块以进程内 API 的形式
提供同样的校验能力，使得嵌入方应用和 CI 无需依赖外部工具即可强制执行规范：

- frontmatter 只能使用官方顶层字段；
- `allowed-tools` 必须是空格分隔的字符串（而非 YAML 序列）；
- `name` 需符合规范定义的模式，并且与技能目录名一致；
- `description` 必须存在，且不超过 1024 个字符的限制；
- `metadata` 必须是字符串键到字符串值的映射；
- 技能文件只能包含官方格式；Hooks 属于应用/运行时配置，
  而不是 Agent Skills 文件格式的一部分。

违规项（violations）会导致目录校验硬性失败；警告项（warnings）仅作提示，不会阻断校验。
*/

use std::path::{Path, PathBuf};

use serde_yaml_ng::Value;

use super::types::SkillDocument;

/// agentskills.io 官方规定的顶层 frontmatter 字段。
pub const OFFICIAL_FRONTMATTER_FIELDS: &[&str] = &[
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
];

/// `description` 字段允许的最大长度（以 Unicode 标量值计数，规范规定为 1024）。
const MAX_DESCRIPTION_CHARS: usize = 1024;

/// 建议的 `SKILL.md` 正文行数上限（规范建议值，并非硬性限制）。
const RECOMMENDED_MAX_BODY_LINES: usize = 500;

/// 单个技能的校验结果。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillValidationReport {
    /// 被校验的技能目录（或文件）路径。
    pub path: PathBuf,
    /// 规范违规项 —— 只要存在任意一项，目录校验就判定为失败。
    pub violations: Vec<String>,
    /// 仅作提示、不会阻断校验的发现项。
    pub warnings: Vec<String>,
}

impl SkillValidationReport {
    /// 判断该技能是否通过了规范校验。
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    /// 追加一条违规记录。
    fn violation(&mut self, message: impl Into<String>) {
        self.violations.push(message.into());
    }

    /// 追加一条警告记录。
    fn warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }
}

/**
依据官方规范校验一份 `SKILL.md` 文档。

`dir_name` 是规范要求 `name` 字段必须匹配的父目录名；
若是内存中的文档、无需做该项检查，可传入空字符串跳过。
*/
pub fn validate_skill_markdown(content: &str, dir_name: &str) -> SkillValidationReport {
    let mut report = SkillValidationReport::default();

    let yaml = match extract_frontmatter_yaml(content) {
        Ok(yaml) => yaml,
        Err(error) => {
            report.violation(error);
            return report;
        }
    };

    /*
    完整的类型化解析：强制校验必填字段、description 非空、metadata 为字符串类型，
    以及未知顶层字段（deny_unknown_fields）
    */
    let parsed = SkillDocument::parse(content);
    if let Err(error) = &parsed {
        report.violation(format!("SKILL.md does not parse: {error}"));
    }

    let root: Value = match serde_yaml_ng::from_str(yaml) {
        Ok(value) => value,
        Err(error) => {
            report.violation(format!("frontmatter YAML error: {error}"));
            return report;
        }
    };

    let Some(mapping) = root.as_mapping() else {
        report.violation("frontmatter must be a YAML mapping");
        return report;
    };

    let descriptor = parsed.as_ref().map(SkillDocument::descriptor).ok();
    for (key, value) in mapping.iter() {
        let Some(field) = key.as_str() else {
            report.violation("frontmatter contains a non-string field name");
            continue;
        };
        if !OFFICIAL_FRONTMATTER_FIELDS.contains(&field) {
            report.violation(format!(
                "non-official top-level field '{field}'; use only standard fields — routing \
                 intent belongs in description and hooks belong to application configuration"
            ));
            continue;
        }
        validate_field_value(&mut report, field, value);
    }

    if let Some(Value::String(description)) = mapping.get(Value::String("description".into()))
        && description.chars().count() > MAX_DESCRIPTION_CHARS
    {
        report.violation(format!(
            "description is {} chars, exceeding the {MAX_DESCRIPTION_CHARS}-char limit",
            description.chars().count()
        ));
    }
    if let Some(Value::String(compatibility)) = mapping.get(Value::String("compatibility".into()))
        && compatibility.chars().count() > 500
    {
        report.violation("compatibility exceeds the 500-character limit".to_string());
    }

    if let Some(descriptor) = descriptor {
        for warning in descriptor.validate_name() {
            report.violation(format!("name: {warning}"));
        }
        if !dir_name.is_empty() && descriptor.name != dir_name {
            report.violation(format!(
                "name '{}' must match parent directory '{}' per spec",
                descriptor.name, dir_name
            ));
        }
        let description_chars = descriptor.description.chars().count();
        if description_chars > MAX_DESCRIPTION_CHARS {
            report.violation(format!(
                "description is {description_chars} chars, exceeding the {MAX_DESCRIPTION_CHARS}-char limit"
            ));
        }
        if let Some(compatibility) = &descriptor.compatibility
            && compatibility.chars().count() > 500
        {
            report.violation("compatibility exceeds the 500-character limit".to_string());
        }
    }

    let trimmed_start = content;
    let body = trimmed_start
        .find("\n---")
        .and_then(|idx| trimmed_start.get(idx + 4..))
        .unwrap_or("");
    let body_lines = body.lines().count();
    if body_lines > RECOMMENDED_MAX_BODY_LINES {
        report.warning(format!(
            "SKILL.md is {body_lines} lines; spec recommends keeping it under {RECOMMENDED_MAX_BODY_LINES}"
        ));
    }

    report
}

/// 校验一个技能目录及其必需的 `SKILL.md` 文件。
pub fn validate_skill_dir(dir: &Path) -> SkillValidationReport {
    let mut report = SkillValidationReport {
        path: dir.to_path_buf(),
        ..SkillValidationReport::default()
    };

    let skill_file = dir.join("SKILL.md");
    if dir.join("hooks.json").is_file() {
        report.violation(
            "hooks.json is not an official Agent Skills file; configure Hooks in the host or plugin"
                .to_string(),
        );
    }
    let Some(dir_name) = dir.file_name().and_then(|name| name.to_str()) else {
        report.violation("skill directory name is not valid UTF-8");
        return report;
    };
    let content = match std::fs::read_to_string(&skill_file) {
        Ok(content) => content,
        Err(error) => {
            report.violation(format!("cannot read SKILL.md: {error}"));
            return report;
        }
    };

    let markdown_report = validate_skill_markdown(&content, dir_name);
    report.violations.extend(markdown_report.violations);
    report.warnings.extend(markdown_report.warnings);

    report
}

/// 需要原始 YAML 值才能完成的字段级格式检查。
fn validate_field_value(report: &mut SkillValidationReport, field: &str, value: &Value) {
    match field {
        "allowed-tools" => {
            // 官方格式是空格分隔的字符串。YAML 序列是旧式布局，必须迁移。
            if value.is_sequence() {
                report.violation(
                    "allowed-tools must be a space-separated string, not a list \
                     (e.g. `allowed-tools: shell read_file apply_patch`)"
                        .to_string(),
                );
            } else if !value.is_string() {
                report.violation("allowed-tools must be a string".to_string());
            }
        }
        "metadata" => {
            let Some(mapping) = value.as_mapping() else {
                report.violation("metadata must be a mapping".to_string());
                return;
            };
            for (key, entry) in mapping.iter() {
                let Some(key) = key.as_str() else {
                    report.violation("metadata contains a non-string key".to_string());
                    continue;
                };
                if !matches!(entry, Value::String(_)) {
                    report.violation(format!(
                        "metadata.{key} must be a string value (spec maps strings to strings; \
                         no vendor namespaces)"
                    ));
                }
            }
        }
        _ => {}
    }
}

/// 提取 frontmatter 分隔符之间的原始 YAML 文本。
fn extract_frontmatter_yaml(content: &str) -> Result<&str, String> {
    if !content.starts_with("---") {
        return Err("SKILL.md must begin with YAML frontmatter (---)".to_string());
    }
    let trimmed = content;
    let after_open = trimmed
        .get(3..)
        .unwrap_or("")
        .trim_start_matches('\r')
        .trim_start_matches('\n');
    let close_idx = after_open
        .find("\n---")
        .ok_or_else(|| "SKILL.md frontmatter missing closing ---".to_string())?;
    Ok(after_open.get(..close_idx).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一份符合官方规范布局的合法 SKILL.md 文本，供测试使用。
    fn valid_skill() -> String {
        "---\nname: demo-skill\ndescription: Demonstrates the official layout.\nmetadata:\n  category: demo\nallowed-tools: shell read_file\n---\n# Demo\n\nBody.\n".to_string()
    }

    /// 验证符合官方布局规范的 SKILL.md 能够通过校验。
    #[test]
    fn official_layout_passes() {
        let report = validate_skill_markdown(&valid_skill(), "demo-skill");
        assert!(report.is_valid(), "violations: {:?}", report.violations);
    }

    /// 验证使用旧式顶层字段（如 `triggers`）会校验失败，并给出迁移提示。
    #[test]
    fn legacy_top_level_fields_fail_with_migration_hint() {
        let content = "---\nname: demo-skill\ndescription: Legacy.\ntriggers:\n  - demo\n---\nBody";
        let report = validate_skill_markdown(content, "demo-skill");
        assert!(!report.is_valid());
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("non-official top-level field 'triggers'"))
        );
    }

    /// 验证 `allowed-tools` 若写成 YAML 列表形式会校验失败。
    #[test]
    fn list_shaped_allowed_tools_fails() {
        let content = "---\nname: demo-skill\ndescription: Legacy tools.\nallowed-tools: [shell, read_file]\n---\nBody";
        let report = validate_skill_markdown(content, "demo-skill");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("space-separated"))
        );
    }

    /// 验证 `name` 字段与所在目录名不一致时会校验失败。
    #[test]
    fn name_directory_mismatch_fails() {
        let report = validate_skill_markdown(&valid_skill(), "other-dir");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("must match parent directory"))
        );
    }

    /// 验证 `description` 超过 1024 字符限制时会校验失败。
    #[test]
    fn overlong_description_fails() {
        let long: String = "x".repeat(1025);
        let content = format!("---\nname: demo-skill\ndescription: {long}\n---\nBody");
        let report = validate_skill_markdown(&content, "demo-skill");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("1024-char limit"))
        );
    }

    /// 验证 frontmatter 中包含 `hooks` 块时会校验失败（Hooks 不属于官方文件格式）。
    #[test]
    fn frontmatter_hooks_block_fails() {
        let content = "---\nname: demo-skill\ndescription: Hooked.\nhooks:\n  Stop:\n    - hooks:\n        - type: command\n          command: echo done\n---\nBody";
        let report = validate_skill_markdown(content, "demo-skill");
        assert!(!report.is_valid());
    }

    /// 验证 `metadata` 中出现厂商命名空间式的嵌套结构（值非字符串）时会校验失败。
    #[test]
    fn vendor_namespace_in_metadata_fails() {
        let content = "---\nname: demo-skill\ndescription: Namespaced.\nmetadata:\n  echo-agent:\n    triggers: [demo]\n---\nBody";
        let report = validate_skill_markdown(content, "demo-skill");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("metadata.echo-agent must be a string value"))
        );
    }

    /// 验证 `metadata` 中出现非标量（嵌套对象）值时会校验失败。
    #[test]
    fn non_scalar_foreign_metadata_fails() {
        let content = "---\nname: demo-skill\ndescription: Nested.\nmetadata:\n  vendor:\n    nested: true\n---\nBody";
        let report = validate_skill_markdown(content, "demo-skill");
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains("metadata.vendor"))
        );
    }

    /**
    验证 `validate_skill_dir` 能正确处理合法技能目录、存在非官方
    `hooks.json` 文件，以及缺失 `SKILL.md` 文件这几种场景。
    */
    #[test]
    fn skill_dir_validates_required_skill_file() -> std::result::Result<(), String> {
        let root = std::env::temp_dir().join(format!(
            "echo_skill_validate_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let dir = root.join("demo-skill");
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        /// 测试结束时自动清理临时目录的守卫。
        struct Guard(std::path::PathBuf);
        impl Drop for Guard {
            /// 删除守卫持有的临时目录。
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _guard = Guard(root.clone());

        std::fs::write(dir.join("SKILL.md"), valid_skill()).map_err(|error| error.to_string())?;
        let ok = validate_skill_dir(&dir);
        assert!(ok.is_valid(), "violations: {:?}", ok.violations);

        std::fs::write(dir.join("hooks.json"), "{}\n").map_err(|error| error.to_string())?;
        let sidecar = validate_skill_dir(&dir);
        assert!(
            sidecar
                .violations
                .iter()
                .any(|violation| violation.contains("hooks.json"))
        );
        std::fs::remove_file(dir.join("hooks.json")).map_err(|error| error.to_string())?;
        std::fs::remove_file(dir.join("SKILL.md")).map_err(|error| error.to_string())?;
        let broken = validate_skill_dir(&dir);
        assert!(broken.violations.iter().any(|v| v.contains("SKILL.md")));
        Ok(())
    }
}
