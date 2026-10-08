/*!
 * BibTeX 生成工具。
 *
 * 根据论文元数据（arxiv、Semantic Scholar 搜索结果）生成 BibTeX 条目。
 */

use echo_core::error::{Result, ToolError};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult};
use futures::future::BoxFuture;
use serde_json::Value;
use std::collections::HashMap;

/// 工具名称
const TOOL_NAME: &str = "bibtex_generate";

/// 根据论文元数据生成 BibTeX 条目的工具。
pub struct BibtexGenerateTool;

impl Tool for BibtexGenerateTool {
    /// 工具名称。
    fn name(&self) -> &str {
        TOOL_NAME
    }

    /// 所需权限：写入（可选写出 .bib 文件）。
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Write]
    }

    /// 工具描述。
    fn description(&self) -> &str {
        "Generate BibTeX entries from paper metadata. Accepts paper objects from arxiv_search or semantic_scholar_search results. Outputs standard .bib format. Example: bibtex_generate(papers=[{title: 'Attention Is All You Need', authors: ['Vaswani et al.'], year: 2017, arxiv_id: '1706.03762'}])"
    }

    /// 参数的 JSON Schema。
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "description": "Array of paper objects. Each should have: title, authors (array), year, and optionally arxiv_id, doi, url, venue",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string" },
                            "authors": { "type": "array", "items": { "type": "string" } },
                            "year": { "type": "integer" },
                            "arxiv_id": { "type": "string" },
                            "doi": { "type": "string" },
                            "url": { "type": "string" },
                            "venue": { "type": "string" },
                            "abstract": { "type": "string" }
                        },
                        "required": ["title"]
                    }
                },
                "output_file": {
                    "type": "string",
                    "description": "Optional file path to write the .bib file. If not provided, returns the BibTeX content as text."
                }
            },
            "required": ["papers"]
        })
    }

    /// 为每篇论文生成 BibTeX 条目，可选写入文件，并以 JSON 返回结果。
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let papers_val = parameters
                .get("papers")
                .ok_or_else(|| ToolError::MissingParameter("papers".to_string()))?;

            let papers = papers_val
                .as_array()
                .ok_or_else(|| ToolError::InvalidParameter {
                    name: "papers".to_string(),
                    message: "Must be an array of paper objects".to_string(),
                })?;

            if papers.is_empty() {
                return Err(ToolError::InvalidParameter {
                    name: "papers".to_string(),
                    message: "At least one paper is required".to_string(),
                }
                .into());
            }

            let output_file = parameters.get("output_file").and_then(|v| v.as_str());

            let mut bibtex_entries = Vec::new();
            let mut cite_key_counts: HashMap<String, u32> = HashMap::new();

            for paper in papers {
                let entry = generate_bibtex_entry(paper, &mut cite_key_counts)?;
                bibtex_entries.push(entry);
            }

            let bibtex_content = bibtex_entries.join("\n\n");

            // 可选：写入文件
            if let Some(path) = output_file {
                tokio::fs::write(path, &bibtex_content).await.map_err(|e| {
                    ToolError::ExecutionFailed {
                        tool: TOOL_NAME.to_string(),
                        message: format!("Failed to write .bib file: {}", e),
                    }
                })?;
            }

            let result = serde_json::json!({
                "entries_count": bibtex_entries.len(),
                "bibtex": bibtex_content,
                "output_file": output_file,
            });

            Ok(ToolResult::success_json(result))
        })
    }
}

/**
 * 根据论文 JSON 对象生成单条 BibTeX 条目。
 * `cite_key_counts` 记录每个基础引用键已被使用的次数，用于消歧。
 */
fn generate_bibtex_entry(
    paper: &Value,
    cite_key_counts: &mut HashMap<String, u32>,
) -> Result<String> {
    let title =
        paper
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParameter {
                name: "papers".to_string(),
                message: "Each paper must have a 'title' field".to_string(),
            })?;

    let authors: Vec<String> = paper
        .get("authors")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|a| a.as_str().unwrap_or("").to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let year = paper
        .get("year")
        .and_then(|v| v.as_u64())
        .map(|y| y.to_string())
        .unwrap_or_else(|| "XXXX".to_string());

    let arxiv_id = paper.get("arxiv_id").and_then(|v| v.as_str()).unwrap_or("");
    let doi = paper.get("doi").and_then(|v| v.as_str()).unwrap_or("");
    let url = paper.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let venue = paper.get("venue").and_then(|v| v.as_str()).unwrap_or("");

    // 生成引用键：第一作者姓氏 + 年份，并做消歧处理
    let base_cite_key = if let Some(first_author) = authors.first() {
        let last_name = first_author
            .split_whitespace()
            .last()
            .unwrap_or(first_author)
            .to_lowercase()
            .replace(|c: char| !c.is_alphanumeric(), "");
        format!("{}{}", last_name, year)
    } else {
        format!("unknown{}", year)
    };

    // 消歧：smith2023、smith2023a、smith2023b……
    let count = cite_key_counts.entry(base_cite_key.clone()).or_insert(0);
    let cite_key = if *count == 0 {
        *count += 1;
        base_cite_key
    } else {
        let suffix = char::from(b'a' + (*count - 1) as u8);
        *count += 1;
        format!("{}{}", base_cite_key, suffix)
    };

    // 转义 BibTeX 特殊字符
    let escaped_title = escape_bibtex(title);
    let author_str = if authors.is_empty() {
        "Unknown".to_string()
    } else {
        authors.join(" and ")
    };

    let mut entry = format!(
        "@article{{{},\n  title = {{{}}},\n  author = {{{}}},\n  year = {{{}}}",
        cite_key, escaped_title, author_str, year
    );

    if !arxiv_id.is_empty() {
        entry.push_str(&format!(
            ",\n  eprint = {{{}}},\n  archivePrefix = {{arXiv}}",
            arxiv_id
        ));
        // 提取 primaryClass：旧格式 "cs.AI/1234567" -> "cs"，新格式 "2301.12345" -> 跳过
        let primary_class = extract_arxiv_primary_class(arxiv_id, paper);
        if let Some(class) = primary_class {
            entry.push_str(&format!(",\n  primaryClass = {{{}}}", class));
        }
    }

    if !doi.is_empty() {
        entry.push_str(&format!(",\n  doi = {{{}}}", doi));
    }

    if !url.is_empty() {
        entry.push_str(&format!(",\n  url = {{{}}}", url));
    }

    if !venue.is_empty() {
        entry.push_str(&format!(",\n  journal = {{{}}}", escape_bibtex(venue)));
    }

    entry.push_str("\n}");

    Ok(entry)
}

/**
 * 从 arxiv ID 或论文元数据中提取主分类。
 * 旧格式："cs.AI/1234567" -> Some("cs")
 * 新格式："2301.12345" -> 若有 fields_of_study 则使用之，否则为 None
 */
fn extract_arxiv_primary_class(arxiv_id: &str, paper: &Value) -> Option<String> {
    // 旧格式包含斜杠："cs.AI/1234567"
    if let Some(slash_pos) = arxiv_id.find('/') {
        let prefix = &arxiv_id[..slash_pos];
        // "cs.AI" -> "cs"
        return prefix.split('.').next().map(|s| s.to_string());
    }
    // 新格式 "2301.12345"——尝试使用论文元数据中的 fields_of_study
    if let Some(fields) = paper.get("fields_of_study").and_then(|v| v.as_array())
        && let Some(first) = fields.first().and_then(|f| f.as_str())
    {
        // 将常见领域名称映射为 arxiv 分类
        let mapped = match first {
            "Computer Science" => "cs",
            "Mathematics" => "math",
            "Physics" => "physics",
            "Statistics" => "stat",
            "Electrical Engineering" => "eess",
            "Economics" => "econ",
            "Quantitative Biology" => "q-bio",
            "Quantitative Finance" => "q-fin",
            _ => return Some(first.to_lowercase().replace(' ', "-")),
        };
        return Some(mapped.to_string());
    }
    None
}

/// 转义 BibTeX 特殊字符。
fn escape_bibtex(s: &str) -> String {
    s.replace('&', r"\&")
        .replace('%', r"\%")
        .replace('#', r"\#")
        .replace('_', r"\_")
}
