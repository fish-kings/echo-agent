//! 可复用的开放学术元数据与文献管理器 API 客户端。
//!
//! 这些客户端返回一个精简的、归一化的「作品」契约，便于应用按照自己的
//! 存储模型持久化记录。各服务商特有的响应字段仍可通过专门的 Zotero 和
//! Europe PMC 类型获取。

use std::collections::BTreeMap;
use std::time::Duration;

use echo_core::error::{ReactError, Result, ToolError};
use reqwest::{Client, Method, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESULTS: usize = 100;
const MAX_ZOTERO_ITEMS: usize = 10_000;

/// 归一化后的学术作品记录，跨不同数据源（OpenAlex、Crossref、Europe PMC、
/// Zotero 等）使用统一字段表示。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScholarlyWork {
    /// 来源服务商标识（如 "openalex"、"crossref"、"europe_pmc"、"zotero"）。
    pub provider: String,
    /// 该作品在来源服务商体系内的唯一标识。
    pub provider_id: String,
    /// 标题。
    pub title: String,
    /// 作者姓名列表。
    #[serde(default)]
    pub authors: Vec<String>,
    /// 摘要文本（若可获取）。
    pub abstract_text: Option<String>,
    /// 归一化后的 DOI（若有）。
    pub doi: Option<String>,
    /// PubMed ID（若有）。
    pub pmid: Option<String>,
    /// PubMed Central ID（若有）。
    pub pmcid: Option<String>,
    /// arXiv ID（若有）。
    pub arxiv_id: Option<String>,
    /// OpenAlex ID（若有）。
    pub openalex_id: Option<String>,
    /// 发表年份（若可解析）。
    pub year: Option<i32>,
    /// 发表载体（期刊/会议名称等）。
    pub venue: Option<String>,
    /// 指向该作品的 URL。
    pub url: Option<String>,
    /// 关键词/标签列表。
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// 一页学术搜索结果。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScholarlySearchPage {
    /// 服务商上报的匹配总数（若提供）。
    pub total: Option<u64>,
    /// 本页返回的作品列表。
    #[serde(default)]
    pub works: Vec<ScholarlyWork>,
}

/// OpenAlex API 客户端。
#[derive(Debug, Clone)]
pub struct OpenAlexClient {
    /// 底层 HTTP 客户端。
    client: Client,
    /// API 基础 URL。
    base_url: String,
    /// 可选的 `mailto` 参数，用于加入 OpenAlex 的礼貌池以获得更高限额。
    mailto: Option<String>,
}

impl OpenAlexClient {
    /// 使用 OpenAlex 官方默认地址创建客户端。
    pub fn new(mailto: Option<String>) -> Result<Self> {
        Self::with_base_url("https://api.openalex.org", mailto)
    }

    /// 使用自定义基础 URL 创建客户端（便于测试时指向本地 mock 服务）。
    pub fn with_base_url(base_url: impl Into<String>, mailto: Option<String>) -> Result<Self> {
        Ok(Self {
            client: build_client()?,
            base_url: clean_base_url(base_url),
            mailto: clean_optional(mailto),
        })
    }

    /// 按关键字搜索作品，返回归一化的一页结果。
    pub async fn search(&self, query: &str, limit: usize) -> Result<ScholarlySearchPage> {
        let query = required_query(query)?;
        let mut request = self.client.get(format!("{}/works", self.base_url)).query(&[
            ("search", query),
            ("per-page", &limit.clamp(1, MAX_RESULTS).to_string()),
        ]);
        if let Some(mailto) = self.mailto.as_deref() {
            request = request.query(&[("mailto", mailto)]);
        }
        let value = json_response(
            request
                .send()
                .await
                .map_err(|error| http_error("OpenAlex search", error))?,
            "OpenAlex search",
        )
        .await?;
        parse_openalex_page(&value)
    }

    /// 按 DOI 精确查找单篇作品；若未找到返回 `None`。
    pub async fn resolve_doi(&self, doi: &str) -> Result<Option<ScholarlyWork>> {
        let doi = normalize_doi(doi).ok_or_else(|| invalid("DOI cannot be empty"))?;
        let mut request = self.client.get(format!("{}/works", self.base_url)).query(&[
            ("filter", format!("doi:{doi}")),
            ("per-page", "1".to_string()),
        ]);
        if let Some(mailto) = self.mailto.as_deref() {
            request = request.query(&[("mailto", mailto)]);
        }
        let value = json_response(
            request
                .send()
                .await
                .map_err(|error| http_error("OpenAlex DOI lookup", error))?,
            "OpenAlex DOI lookup",
        )
        .await?;
        Ok(parse_openalex_page(&value)?.works.into_iter().next())
    }
}

/// Crossref API 客户端。
#[derive(Debug, Clone)]
pub struct CrossrefClient {
    /// 底层 HTTP 客户端。
    client: Client,
    /// API 基础 URL。
    base_url: String,
    /// 可选的 `mailto` 参数，用于加入 Crossref 的礼貌池以获得更高限额。
    mailto: Option<String>,
}

impl CrossrefClient {
    /// 使用 Crossref 官方默认地址创建客户端。
    pub fn new(mailto: Option<String>) -> Result<Self> {
        Self::with_base_url("https://api.crossref.org", mailto)
    }

    /// 使用自定义基础 URL 创建客户端（便于测试时指向本地 mock 服务）。
    pub fn with_base_url(base_url: impl Into<String>, mailto: Option<String>) -> Result<Self> {
        Ok(Self {
            client: build_client()?,
            base_url: clean_base_url(base_url),
            mailto: clean_optional(mailto),
        })
    }

    /// 按关键字搜索作品，返回归一化的一页结果。
    pub async fn search(&self, query: &str, limit: usize) -> Result<ScholarlySearchPage> {
        let query = required_query(query)?;
        let mut request = self.client.get(format!("{}/works", self.base_url)).query(&[
            ("query", query),
            ("rows", &limit.clamp(1, MAX_RESULTS).to_string()),
        ]);
        if let Some(mailto) = self.mailto.as_deref() {
            request = request.query(&[("mailto", mailto)]);
        }
        let value = json_response(
            request
                .send()
                .await
                .map_err(|error| http_error("Crossref search", error))?,
            "Crossref search",
        )
        .await?;
        parse_crossref_page(&value)
    }

    /// 按 DOI 精确查找单篇作品；若返回 404 则视为未找到，返回 `None`。
    pub async fn resolve_doi(&self, doi: &str) -> Result<Option<ScholarlyWork>> {
        let doi = normalize_doi(doi).ok_or_else(|| invalid("DOI cannot be empty"))?;
        let url = format!("{}/works/{}", self.base_url, urlencoding::encode(&doi));
        let mut request = self.client.get(url);
        if let Some(mailto) = self.mailto.as_deref() {
            request = request.query(&[("mailto", mailto)]);
        }
        let response = request
            .send()
            .await
            .map_err(|error| http_error("Crossref DOI lookup", error))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let value = json_response(response, "Crossref DOI lookup").await?;
        value.get("message").map(parse_crossref_work).transpose()
    }
}

/// Europe PMC 中的一条引用/被引关系链接。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EuropePmcLink {
    /// 关联作品的标识。
    pub id: String,
    /// 关联作品的来源库（如 MED、PMC）。
    pub source: Option<String>,
    /// 关联作品的标题。
    pub title: Option<String>,
    /// 关联作品的作者字符串。
    pub authors: Option<String>,
    /// 关联作品的发表年份。
    pub year: Option<i32>,
}

/// Europe PMC 文本挖掘得到的术语条目。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EuropePmcTerm {
    /// 术语名称。
    pub name: String,
    /// 术语的语义类型分类。
    pub semantic_type: Option<String>,
    /// 该术语在文中出现的频次。
    pub frequency: Option<u64>,
}

/// Europe PMC API 客户端。
#[derive(Debug, Clone)]
pub struct EuropePmcClient {
    /// 底层 HTTP 客户端。
    client: Client,
    /// API 基础 URL。
    base_url: String,
}

impl EuropePmcClient {
    /// 使用 Europe PMC 官方默认地址创建客户端。
    pub fn new() -> Result<Self> {
        Self::with_base_url("https://www.ebi.ac.uk/europepmc/webservices/rest")
    }

    /// 使用自定义基础 URL 创建客户端（便于测试时指向本地 mock 服务）。
    pub fn with_base_url(base_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: build_client()?,
            base_url: clean_base_url(base_url),
        })
    }

    /// 按关键字搜索作品，返回归一化的一页结果。
    pub async fn search(&self, query: &str, limit: usize) -> Result<ScholarlySearchPage> {
        let query = required_query(query)?;
        let response = self
            .client
            .get(format!("{}/search", self.base_url))
            .query(&[
                ("query", query),
                ("format", "json"),
                ("pageSize", &limit.clamp(1, MAX_RESULTS).to_string()),
            ])
            .send()
            .await
            .map_err(|error| http_error("Europe PMC search", error))?;
        let value = json_response(response, "Europe PMC search").await?;
        parse_europe_pmc_page(&value)
    }

    /// 获取引用了该作品的文献列表（谁引用了它）。
    pub async fn citations(&self, source: &str, id: &str) -> Result<Vec<EuropePmcLink>> {
        self.links(source, id, "citations").await
    }

    /// 获取该作品所引用的文献列表（它引用了谁）。
    pub async fn references(&self, source: &str, id: &str) -> Result<Vec<EuropePmcLink>> {
        self.links(source, id, "references").await
    }

    /// 获取该作品的文本挖掘术语（按语义类型分组的实体标注）。
    pub async fn text_mined_terms(&self, source: &str, id: &str) -> Result<Vec<EuropePmcTerm>> {
        let response = self
            .client
            .get(format!(
                "{}/{}/{}/textMinedTerms",
                self.base_url,
                encode_path(source)?,
                encode_path(id)?
            ))
            .query(&[("format", "json")])
            .send()
            .await
            .map_err(|error| http_error("Europe PMC text-mined terms", error))?;
        let value = json_response(response, "Europe PMC text-mined terms").await?;
        Ok(parse_europe_pmc_terms(&value))
    }

    /// 获取指定 PMC ID 的全文 XML。
    pub async fn full_text_xml(&self, pmcid: &str) -> Result<String> {
        let response = self
            .client
            .get(format!(
                "{}/{}/fullTextXML",
                self.base_url,
                encode_path(pmcid)?
            ))
            .send()
            .await
            .map_err(|error| http_error("Europe PMC full text", error))?;
        text_response(response, "Europe PMC full text").await
    }

    /// `citations`/`references` 的共用实现：按给定关系类型拉取关联作品链接。
    async fn links(&self, source: &str, id: &str, relation: &str) -> Result<Vec<EuropePmcLink>> {
        let response = self
            .client
            .get(format!(
                "{}/{}/{}/{}",
                self.base_url,
                encode_path(source)?,
                encode_path(id)?,
                relation
            ))
            .query(&[("format", "json"), ("pageSize", "100")])
            .send()
            .await
            .map_err(|error| http_error("Europe PMC links", error))?;
        let value = json_response(response, "Europe PMC links").await?;
        Ok(parse_europe_pmc_links(&value))
    }
}

/// Zotero 图书馆的所属类型：个人图书馆或群组图书馆。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoteroLibraryKind {
    /// 个人图书馆。
    User,
    /// 群组图书馆。
    Group,
}

impl ZoteroLibraryKind {
    /// 返回该图书馆类型在 Zotero API 路径中对应的段（"users" 或 "groups"）。
    fn path_segment(self) -> &'static str {
        match self {
            Self::User => "users",
            Self::Group => "groups",
        }
    }
}

/// Zotero 条目的创建者（作者、编者等）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZoteroCreator {
    /// 创建者角色类型（如 "author"）。
    pub creator_type: String,
    /// 名。
    pub first_name: Option<String>,
    /// 姓。
    pub last_name: Option<String>,
    /// 单字段全名（与姓/名二选一使用）。
    pub name: Option<String>,
}

/// Zotero 条目的标签。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZoteroTag {
    /// 标签文本。
    pub tag: String,
}

/// Zotero 条目的数据字段。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZoteroItemData {
    /// 条目类型（如 "journalArticle"）。
    pub item_type: String,
    /// 标题。
    pub title: String,
    /// 创建者列表。
    #[serde(default)]
    pub creators: Vec<ZoteroCreator>,
    /// 摘要备注。
    pub abstract_note: Option<String>,
    /// 发表载体标题（期刊/出版物名称）。
    pub publication_title: Option<String>,
    /// 发表日期（原始字符串形式）。
    pub date: Option<String>,
    /// DOI。
    #[serde(rename = "DOI")]
    pub doi: Option<String>,
    /// URL。
    pub url: Option<String>,
    /// 标签列表。
    #[serde(default)]
    pub tags: Vec<ZoteroTag>,
    /// 未被显式建模的其余字段，原样保留透传。
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// 一条完整的 Zotero 条目记录，包含服务端元数据与条目数据。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ZoteroItem {
    /// 条目唯一键。
    pub key: String,
    /// 条目版本号，用于乐观并发控制。
    pub version: u64,
    /// 条目的具体数据字段。
    pub data: ZoteroItemData,
}

/// Zotero API 客户端。
#[derive(Debug, Clone)]
pub struct ZoteroClient {
    /// 底层 HTTP 客户端。
    client: Client,
    /// API 基础 URL。
    base_url: String,
    /// 目标图书馆类型（个人/群组）。
    library_kind: ZoteroLibraryKind,
    /// 目标图书馆 ID。
    library_id: String,
    /// Zotero API 密钥。
    api_key: String,
}

impl ZoteroClient {
    /// 使用 Zotero 官方默认地址创建客户端。
    pub fn new(
        library_kind: ZoteroLibraryKind,
        library_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        Self::with_base_url("https://api.zotero.org", library_kind, library_id, api_key)
    }

    /// 使用自定义基础 URL 创建客户端（便于测试时指向本地 mock 服务）；
    /// 若图书馆 ID 或 API 密钥为空则返回错误。
    pub fn with_base_url(
        base_url: impl Into<String>,
        library_kind: ZoteroLibraryKind,
        library_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let library_id = library_id.into().trim().to_string();
        let api_key = api_key.into().trim().to_string();
        if library_id.is_empty() || api_key.is_empty() {
            return Err(invalid("Zotero library ID and API key are required"));
        }
        Ok(Self {
            client: build_client()?,
            base_url: clean_base_url(base_url),
            library_kind,
            library_id,
            api_key,
        })
    }

    /// 分页拉取图书馆中的条目，直到达到 `limit`、服务端返回空页或已取完全部条目。
    pub async fn list_items(&self, limit: usize) -> Result<Vec<ZoteroItem>> {
        let target = limit.clamp(1, MAX_ZOTERO_ITEMS);
        let mut items = Vec::new();
        let mut start = 0usize;
        loop {
            let page_size = target.saturating_sub(items.len()).min(MAX_RESULTS);
            if page_size == 0 {
                break;
            }
            let response = self
                .request(Method::GET, format!("{}/items", self.library_path()))
                .query(&[
                    ("format", "json"),
                    ("limit", &page_size.to_string()),
                    ("start", &start.to_string()),
                ])
                .send()
                .await
                .map_err(|error| http_error("Zotero item list", error))?;
            let total = response
                .headers()
                .get("Total-Results")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<usize>().ok());
            let value = json_response(response, "Zotero item list").await?;
            let mut page: Vec<ZoteroItem> = serde_json::from_value(value)
                .map_err(|error| json_error("Zotero item list", error))?;
            let returned = page.len();
            items.append(&mut page);
            start = start.saturating_add(returned);
            if returned == 0 || items.len() >= target || total.is_some_and(|total| start >= total) {
                break;
            }
        }
        Ok(items)
    }

    /// 在图书馆中批量创建条目，返回 Zotero API 的原始响应。
    pub async fn create_items(&self, items: &[ZoteroItemData]) -> Result<Value> {
        if items.is_empty() {
            return Err(invalid("at least one Zotero item is required"));
        }
        let response = self
            .request(Method::POST, format!("{}/items", self.library_path()))
            .json(items)
            .send()
            .await
            .map_err(|error| http_error("Zotero item creation", error))?;
        json_response(response, "Zotero item creation").await
    }

    /// 拼接该图书馆的基础 API 路径（`base_url/{users|groups}/{library_id}`）。
    fn library_path(&self) -> String {
        format!(
            "{}/{}/{library_id}",
            self.base_url,
            self.library_kind.path_segment(),
            library_id = self.library_id
        )
    }

    /// 构建携带 Zotero 认证头的请求构建器。
    fn request(&self, method: Method, url: String) -> reqwest::RequestBuilder {
        self.client
            .request(method, url)
            .header("Zotero-API-Version", "3")
            .header("Zotero-API-Key", &self.api_key)
    }
}

/// 将一条 Zotero 条目转换为归一化的 `ScholarlyWork`。
pub fn scholarly_work_from_zotero(item: &ZoteroItem) -> ScholarlyWork {
    ScholarlyWork {
        provider: "zotero".to_string(),
        provider_id: item.key.clone(),
        title: item.data.title.clone(),
        authors: item
            .data
            .creators
            .iter()
            .filter_map(zotero_creator_name)
            .collect(),
        abstract_text: clean_optional(item.data.abstract_note.clone()),
        doi: item.data.doi.as_deref().and_then(normalize_doi),
        year: item.data.date.as_deref().and_then(first_year),
        venue: clean_optional(item.data.publication_title.clone()),
        url: clean_optional(item.data.url.clone()),
        keywords: item
            .data
            .tags
            .iter()
            .filter_map(|tag| clean_optional(Some(tag.tag.clone())))
            .collect(),
        ..ScholarlyWork::default()
    }
}

/// 将一条归一化的 `ScholarlyWork` 转换为可提交给 Zotero 的条目数据。
pub fn scholarly_work_to_zotero(work: &ScholarlyWork) -> ZoteroItemData {
    ZoteroItemData {
        item_type: "journalArticle".to_string(),
        title: work.title.clone(),
        creators: work
            .authors
            .iter()
            .filter_map(|name| {
                clean_optional(Some(name.clone())).map(|name| ZoteroCreator {
                    creator_type: "author".to_string(),
                    name: Some(name),
                    ..ZoteroCreator::default()
                })
            })
            .collect(),
        abstract_note: work.abstract_text.clone(),
        publication_title: work.venue.clone(),
        date: work.year.map(|year| year.to_string()),
        doi: work.doi.clone(),
        url: work.url.clone(),
        tags: work
            .keywords
            .iter()
            .map(|tag| ZoteroTag { tag: tag.clone() })
            .collect(),
        extra: BTreeMap::new(),
    }
}

/// 解析 OpenAlex 搜索响应为归一化的一页结果。
fn parse_openalex_page(value: &Value) -> Result<ScholarlySearchPage> {
    let works = value
        .get("results")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(parse_openalex_work).collect())
        .transpose()?
        .unwrap_or_default();
    Ok(ScholarlySearchPage {
        total: value
            .get("meta")
            .and_then(|meta| meta.get("count"))
            .and_then(Value::as_u64),
        works,
    })
}

/// 解析单条 OpenAlex 作品记录为归一化的 `ScholarlyWork`。
fn parse_openalex_work(value: &Value) -> Result<ScholarlyWork> {
    let provider_id = string_at(value, &["id"]).unwrap_or_default();
    let openalex_id = strip_url_identifier(&provider_id, "https://openalex.org/");
    let ids = value.get("ids").unwrap_or(&Value::Null);
    let primary_location = value.get("primary_location").unwrap_or(&Value::Null);
    Ok(ScholarlyWork {
        provider: "openalex".to_string(),
        provider_id,
        title: string_at(value, &["title"]).unwrap_or_default(),
        authors: value
            .get("authorships")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| string_at(item, &["author", "display_name"]))
                    .collect()
            })
            .unwrap_or_default(),
        abstract_text: inverted_abstract(value.get("abstract_inverted_index")),
        doi: string_at(value, &["doi"]).and_then(|doi| normalize_doi(&doi)),
        pmid: string_at(ids, &["pmid"])
            .and_then(|id| strip_url_identifier(&id, "https://pubmed.ncbi.nlm.nih.gov/")),
        pmcid: string_at(ids, &["pmcid"])
            .and_then(|id| strip_url_identifier(&id, "https://www.ncbi.nlm.nih.gov/pmc/articles/")),
        openalex_id,
        year: value
            .get("publication_year")
            .and_then(Value::as_i64)
            .and_then(|year| i32::try_from(year).ok()),
        venue: string_at(primary_location, &["source", "display_name"]),
        url: string_at(primary_location, &["landing_page_url"])
            .or_else(|| string_at(value, &["doi"])),
        keywords: value
            .get("keywords")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| string_at(item, &["display_name"]))
                    .collect()
            })
            .unwrap_or_default(),
        ..ScholarlyWork::default()
    })
}

/// 解析 Crossref 搜索响应为归一化的一页结果。
fn parse_crossref_page(value: &Value) -> Result<ScholarlySearchPage> {
    let message = value.get("message").unwrap_or(&Value::Null);
    let works = message
        .get("items")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(parse_crossref_work).collect())
        .transpose()?
        .unwrap_or_default();
    Ok(ScholarlySearchPage {
        total: message.get("total-results").and_then(Value::as_u64),
        works,
    })
}

/// 解析单条 Crossref 作品记录为归一化的 `ScholarlyWork`。
fn parse_crossref_work(value: &Value) -> Result<ScholarlyWork> {
    let doi = string_at(value, &["DOI"]).and_then(|value| normalize_doi(&value));
    let provider_id = doi
        .clone()
        .or_else(|| string_at(value, &["URL"]))
        .unwrap_or_default();
    Ok(ScholarlyWork {
        provider: "crossref".to_string(),
        provider_id,
        title: first_string(value.get("title")).unwrap_or_default(),
        authors: value
            .get("author")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(crossref_author_name).collect())
            .unwrap_or_default(),
        abstract_text: string_at(value, &["abstract"]),
        doi,
        year: crossref_year(value),
        venue: first_string(value.get("container-title")),
        url: string_at(value, &["URL"]),
        keywords: value
            .get("subject")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        ..ScholarlyWork::default()
    })
}

/// 解析 Europe PMC 搜索响应为归一化的一页结果。
fn parse_europe_pmc_page(value: &Value) -> Result<ScholarlySearchPage> {
    let works = value
        .get("resultList")
        .and_then(|list| list.get("result"))
        .and_then(Value::as_array)
        .map(|items| items.iter().map(parse_europe_pmc_work).collect())
        .transpose()?
        .unwrap_or_default();
    Ok(ScholarlySearchPage {
        total: value.get("hitCount").and_then(Value::as_u64),
        works,
    })
}

/// 解析单条 Europe PMC 作品记录为归一化的 `ScholarlyWork`。
fn parse_europe_pmc_work(value: &Value) -> Result<ScholarlyWork> {
    let source = string_at(value, &["source"]).unwrap_or_default();
    let id = string_at(value, &["id"]).unwrap_or_default();
    Ok(ScholarlyWork {
        provider: "europe_pmc".to_string(),
        provider_id: if source.is_empty() {
            id.clone()
        } else {
            format!("{source}:{id}")
        },
        title: string_at(value, &["title"]).unwrap_or_default(),
        authors: string_at(value, &["authorString"])
            .map(|authors| {
                authors
                    .split(',')
                    .filter_map(|name| clean_optional(Some(name.to_string())))
                    .collect()
            })
            .unwrap_or_default(),
        abstract_text: string_at(value, &["abstractText"]),
        doi: string_at(value, &["doi"]).and_then(|doi| normalize_doi(&doi)),
        pmid: (source.eq_ignore_ascii_case("MED") || id.chars().all(|ch| ch.is_ascii_digit()))
            .then_some(id.clone()),
        pmcid: string_at(value, &["pmcid"]),
        year: string_at(value, &["pubYear"]).and_then(|year| year.parse::<i32>().ok()),
        venue: string_at(value, &["journalTitle"]),
        url: (!id.is_empty()).then(|| format!("https://europepmc.org/article/{source}/{id}")),
        ..ScholarlyWork::default()
    })
}

/// 解析 Europe PMC 的引用/参考文献链接列表响应。
fn parse_europe_pmc_links(value: &Value) -> Vec<EuropePmcLink> {
    ["citationList", "referenceList"]
        .iter()
        .find_map(|key| {
            value
                .get(key)
                .and_then(|list| list.get("citation").or_else(|| list.get("reference")))
                .and_then(Value::as_array)
        })
        .map(|items| {
            items
                .iter()
                .map(|item| EuropePmcLink {
                    id: string_at(item, &["id"])
                        .or_else(|| string_at(item, &["citedById"]))
                        .unwrap_or_default(),
                    source: string_at(item, &["source"]),
                    title: string_at(item, &["title"]),
                    authors: string_at(item, &["authorString"]),
                    year: string_at(item, &["pubYear"]).and_then(|year| year.parse::<i32>().ok()),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 解析 Europe PMC 的文本挖掘术语响应（按语义类型分组后展平）。
fn parse_europe_pmc_terms(value: &Value) -> Vec<EuropePmcTerm> {
    value
        .get("semanticTypeList")
        .and_then(|list| list.get("semanticType"))
        .and_then(Value::as_array)
        .map(|groups| {
            groups
                .iter()
                .flat_map(|group| {
                    let semantic_type = string_at(group, &["name"]);
                    group
                        .get("tmSummaryList")
                        .and_then(|list| list.get("tmSummary"))
                        .and_then(Value::as_array)
                        .map(|terms| {
                            terms
                                .iter()
                                .filter_map(|term| {
                                    string_at(term, &["term"]).map(|name| EuropePmcTerm {
                                        name,
                                        semantic_type: semantic_type.clone(),
                                        frequency: term.get("count").and_then(Value::as_u64),
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 构建带有默认超时、User-Agent 与 SSRF 防护重定向策略的共享 HTTP 客户端。
fn build_client() -> Result<Client> {
    Client::builder()
        .timeout(DEFAULT_TIMEOUT)
        .user_agent("echo-agent/0.2 scholarly-research")
        .redirect(crate::security::ssrf_safe_redirect_policy())
        .build()
        .map_err(|error| http_error("research client setup", error))
}

/// 读取响应体并在状态码非成功时返回错误，否则解析为 JSON `Value`。
async fn json_response(response: Response, operation: &str) -> Result<Value> {
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| http_error(operation, error))?;
    if !status.is_success() {
        return Err(invalid(format!(
            "{operation} failed with HTTP {status}: {}",
            body.chars().take(500).collect::<String>()
        )));
    }
    serde_json::from_str(&body).map_err(|error| json_error(operation, error))
}

/// 读取响应体并在状态码非成功时返回错误，否则返回原始文本。
async fn text_response(response: Response, operation: &str) -> Result<String> {
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| http_error(operation, error))?;
    if !status.is_success() {
        return Err(invalid(format!(
            "{operation} failed with HTTP {status}: {}",
            body.chars().take(500).collect::<String>()
        )));
    }
    Ok(body)
}

/// 构建一个「参数不合法」类型的错误。
fn invalid(message: impl Into<String>) -> ReactError {
    ToolError::InvalidParameter {
        name: "research_api".to_string(),
        message: message.into(),
    }
    .into()
}

/// 将底层 HTTP 请求错误包装为工具执行失败错误。
fn http_error(operation: &str, error: reqwest::Error) -> ReactError {
    ToolError::ExecutionFailed {
        tool: "research_api".to_string(),
        message: format!("{operation} request failed: {error}"),
    }
    .into()
}

/// 将 JSON 反序列化错误包装为工具执行失败错误。
fn json_error(operation: &str, error: serde_json::Error) -> ReactError {
    ToolError::ExecutionFailed {
        tool: "research_api".to_string(),
        message: format!("{operation} returned invalid JSON: {error}"),
    }
    .into()
}

/// 校验查询字符串去除首尾空白后非空，返回修剪后的引用。
fn required_query(query: &str) -> Result<&str> {
    let query = query.trim();
    if query.is_empty() {
        return Err(invalid("query cannot be empty"));
    }
    Ok(query)
}

/// 修剪基础 URL 首尾空白，并去掉末尾多余的斜杠。
fn clean_base_url(value: impl Into<String>) -> String {
    value.into().trim().trim_end_matches('/').to_string()
}

/// 修剪可选字符串的首尾空白，若结果为空则归一化为 `None`。
fn clean_optional(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let value = value.trim().to_string();
        (!value.is_empty()).then_some(value)
    })
}

/// 归一化 DOI：去除首尾空白、转小写，并剥离常见的 URL/协议前缀。
fn normalize_doi(value: &str) -> Option<String> {
    let mut doi = value.trim().to_lowercase();
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"] {
        if doi.starts_with(prefix) {
            doi = doi.chars().skip(prefix.chars().count()).collect();
            break;
        }
    }
    clean_optional(Some(doi))
}

/// 剥离给定前缀（若存在）及首尾的 `/`，用于从 URL 形式的标识符中提取原始 ID。
fn strip_url_identifier(value: &str, prefix: &str) -> Option<String> {
    let stripped = value
        .strip_prefix(prefix)
        .unwrap_or(value)
        .trim_matches('/')
        .to_string();
    clean_optional(Some(stripped))
}

/// 校验路径标识符非空后进行 URL 编码。
fn encode_path(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(invalid("path identifier cannot be empty"));
    }
    Ok(urlencoding::encode(value).to_string())
}

/// 按给定的键路径依次深入 JSON 值，取出字符串并做清理（返回 `None` 表示缺失或为空）。
fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_str()
        .and_then(|text| clean_optional(Some(text.to_string())))
}

/// 取 JSON 数组中的第一个字符串元素并做清理（用于 Crossref 的数组型字段）。
fn first_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .and_then(|text| clean_optional(Some(text.to_string())))
}

/// 从 Crossref 作者对象中拼接出「名 姓」形式的全名。
fn crossref_author_name(value: &Value) -> Option<String> {
    let given = string_at(value, &["given"]).unwrap_or_default();
    let family = string_at(value, &["family"]).unwrap_or_default();
    clean_optional(Some(format!("{given} {family}")))
}

/// 依次尝试 `published-print`/`published-online`/`issued` 字段，提取发表年份。
fn crossref_year(value: &Value) -> Option<i32> {
    ["published-print", "published-online", "issued"]
        .iter()
        .find_map(|key| {
            value
                .get(key)
                .and_then(|date| date.get("date-parts"))
                .and_then(Value::as_array)
                .and_then(|parts| parts.first())
                .and_then(Value::as_array)
                .and_then(|part| part.first())
                .and_then(Value::as_i64)
                .and_then(|year| i32::try_from(year).ok())
        })
}

/// 从日期字符串中提取前 4 位数字作为年份（用于 Zotero 的自由格式日期字段）。
fn first_year(value: &str) -> Option<i32> {
    let digits: String = value.chars().filter(char::is_ascii_digit).take(4).collect();
    (digits.chars().count() == 4)
        .then(|| digits.parse::<i32>().ok())
        .flatten()
}

/// 取创建者的单字段全名，若缺失则回退为「名 姓」拼接。
fn zotero_creator_name(value: &ZoteroCreator) -> Option<String> {
    value
        .name
        .clone()
        .and_then(|name| clean_optional(Some(name)))
        .or_else(|| {
            clean_optional(Some(format!(
                "{} {}",
                value.first_name.as_deref().unwrap_or(""),
                value.last_name.as_deref().unwrap_or("")
            )))
        })
}

/// 将 OpenAlex 的倒排索引摘要（词 -> 出现位置列表）还原为按位置排序的普通文本。
fn inverted_abstract(value: Option<&Value>) -> Option<String> {
    let index = value?.as_object()?;
    let mut words = Vec::new();
    for (word, positions) in index {
        for position in positions
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
        {
            words.push((position, word.clone()));
        }
    }
    words.sort_by_key(|(position, _)| *position);
    clean_optional(Some(
        words
            .into_iter()
            .map(|(_, word)| word)
            .collect::<Vec<_>>()
            .join(" "),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_provider_contracts() -> Result<()> {
        let openalex = parse_openalex_page(&json!({
            "meta": {"count": 1},
            "results": [{
                "id": "https://openalex.org/W1",
                "title": "A study",
                "doi": "https://doi.org/10.1000/test",
                "publication_year": 2025,
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "abstract_inverted_index": {"hello": [0], "world": [1]},
                "primary_location": {"source": {"display_name": "Journal"}, "landing_page_url": "https://example.test"}
            }]
        }))?;
        assert_eq!(
            openalex
                .works
                .first()
                .and_then(|work| work.openalex_id.as_deref()),
            Some("W1")
        );
        assert_eq!(
            openalex
                .works
                .first()
                .and_then(|work| work.abstract_text.as_deref()),
            Some("hello world")
        );

        let crossref = parse_crossref_page(&json!({
            "message": {"total-results": 1, "items": [{
                "DOI": "10.1000/test", "title": ["A study"],
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "published-online": {"date-parts": [[2025, 1, 1]]}
            }]}
        }))?;
        assert_eq!(
            crossref.works.first().and_then(|work| work.year),
            Some(2025)
        );

        let europe = parse_europe_pmc_page(&json!({
            "hitCount": 1,
            "resultList": {"result": [{"source": "MED", "id": "123", "title": "Trial", "pubYear": "2024"}]}
        }))?;
        assert_eq!(
            europe.works.first().and_then(|work| work.pmid.as_deref()),
            Some("123")
        );
        Ok(())
    }

    #[test]
    fn zotero_conversion_preserves_citation_fields() -> Result<()> {
        let item: ZoteroItem = serde_json::from_value(json!({
            "key": "ABC", "version": 1,
            "data": {
                "itemType": "journalArticle", "title": "Paper", "DOI": "10.1/test",
                "date": "2025-02-03", "creators": [{"creatorType": "author", "name": "A. Author"}],
                "tags": [{"tag": "evidence"}]
            }
        }))?;
        let work = scholarly_work_from_zotero(&item);
        assert_eq!(work.provider_id, "ABC");
        assert_eq!(work.year, Some(2025));
        assert_eq!(
            scholarly_work_to_zotero(&work).doi.as_deref(),
            Some("10.1/test")
        );
        Ok(())
    }

    #[tokio::test]
    #[ignore = "opt-in live smoke test; set ECHO_AGENT_PROVIDER_SMOKE=1"]
    async fn live_open_scholarly_providers_return_results() -> Result<()> {
        if std::env::var("ECHO_AGENT_PROVIDER_SMOKE").as_deref() != Ok("1") {
            return Err(invalid(
                "set ECHO_AGENT_PROVIDER_SMOKE=1 before running ignored provider smoke tests",
            ));
        }
        let mailto = std::env::var("OPENALEX_MAILTO").ok();
        let openalex = OpenAlexClient::new(mailto.clone())?
            .search("systematic review", 1)
            .await?;
        let crossref = CrossrefClient::new(mailto)?
            .search("systematic review", 1)
            .await?;
        let europe_pmc = EuropePmcClient::new()?
            .search("systematic review", 1)
            .await?;
        if openalex.works.is_empty() || crossref.works.is_empty() || europe_pmc.works.is_empty() {
            return Err(invalid(
                "one or more scholarly providers returned an empty smoke-test page",
            ));
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires ZOTERO_API_KEY and ZOTERO_LIBRARY_ID"]
    async fn live_zotero_credentials_can_list_items() -> Result<()> {
        let api_key = std::env::var("ZOTERO_API_KEY")
            .map_err(|_| invalid("ZOTERO_API_KEY is required for the Zotero smoke test"))?;
        let library_id = std::env::var("ZOTERO_LIBRARY_ID")
            .map_err(|_| invalid("ZOTERO_LIBRARY_ID is required for the Zotero smoke test"))?;
        let library_kind = match std::env::var("ZOTERO_LIBRARY_KIND")
            .unwrap_or_else(|_| "user".to_string())
            .as_str()
        {
            "user" => ZoteroLibraryKind::User,
            "group" => ZoteroLibraryKind::Group,
            value => {
                return Err(invalid(format!(
                    "ZOTERO_LIBRARY_KIND must be user or group, got {value}"
                )));
            }
        };
        ZoteroClient::new(library_kind, library_id, api_key)?
            .list_items(1)
            .await?;
        Ok(())
    }
}
