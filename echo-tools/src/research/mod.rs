/*!
 * 科研与论文检索工具。
 *
 * 提供学术论文搜索（arxiv、Semantic Scholar、PubMed）、
 * 临床试验搜索、PDF 下载与解析，以及 BibTeX 生成。
 */

/// arXiv 论文搜索工具。
pub mod arxiv;
/// BibTeX 引用生成工具。
pub mod bibtex;
/// 学术数据源（Crossref、OpenAlex、Europe PMC、Zotero）客户端。
pub mod clients;
/// 临床试验搜索工具。
pub mod clinical_trials;
/// PDF 下载与解析工具。
pub mod pdf_fetch;
/// PubMed 文献搜索工具。
pub mod pubmed;
/// Semantic Scholar 论文搜索工具。
pub mod semantic_scholar;

pub use arxiv::ArxivSearchTool;
pub use bibtex::BibtexGenerateTool;
pub use clients::{
    CrossrefClient, EuropePmcClient, EuropePmcLink, EuropePmcTerm, OpenAlexClient,
    ScholarlySearchPage, ScholarlyWork, ZoteroClient, ZoteroCreator, ZoteroItem, ZoteroItemData,
    ZoteroLibraryKind, ZoteroTag, scholarly_work_from_zotero, scholarly_work_to_zotero,
};
pub use clinical_trials::ClinicalTrialsSearchTool;
pub use pdf_fetch::PdfFetchTool;
pub use pubmed::PubMedSearchTool;
pub use semantic_scholar::SemanticScholarSearchTool;
