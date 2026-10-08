/*!
数据库 SQL 工具

基于 sqlx 提供跨数据库的只读查询能力：
- sql_query：执行只读 SQL 查询
- list_tables：列出数据库中的所有表
- describe_table：查看表结构
*/

use base64::Engine;
use futures::{StreamExt, future::BoxFuture};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sqlx::any::{AnyPoolOptions, AnyTypeInfoKind};
use sqlx::{Column, Row};

use echo_core::error::{Result, ToolError};
use echo_core::tools::artifact::{
    ToolOutputArtifactIdentity, ToolOutputArtifactRef, persist_tool_output,
};
use echo_core::tools::pagination::{PageInfo, PageRequest};
use echo_core::tools::permission::ToolPermission;
use echo_core::tools::{Tool, ToolParameters, ToolResult, ToolRiskLevel};

/// SQL 查询默认每页行数
const DEFAULT_SQL_PAGE_ROWS: usize = 50;
/// SQL 查询每页最大行数
const MAX_SQL_PAGE_ROWS: usize = 100;
/// 结果溢出到 artifact 时，内联列名保留的最大字符数
const MAX_INLINE_COLUMN_CHARS: usize = 80;
/// 结果溢出到 artifact 时，内联单元格保留的最大字符数
const MAX_INLINE_CELL_CHARS: usize = 512;

/**
确保 sqlx `any` 驱动已安装。

sqlx 0.8 的 `any` 模式要求在第一次建立 `AnyPool` 连接前于运行时安装驱动，
否则连接时会以 "No drivers installed" panic。这里在首次使用时惰性地
安装所有已编译进来的驱动，并用 OnceLock 保护，使重复调用既廉价又安全。

（早期版本会隐式调用 `AnyConnection::install_default_drivers` 自动安装，
sqlx 0.8 移除了这一行为；显式调用 `install_default_drivers()` 是官方文档给出的替代方式。）
*/
fn ensure_drivers_installed() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        // install_default_drivers 会安装 sqlx 中已编译的所有驱动（取决于 Cargo.toml 中启用的 sqlite/postgres/mysql），且幂等。
        sqlx::any::install_default_drivers();
    });
}

// ── SQL 查询（只读） ─────────────────────────────────────────────────────────

/// 只读 SQL 查询工具，支持分页与大结果溢出到 artifact
pub struct SqlQueryTool;

impl Tool for SqlQueryTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "sql_query"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 返回工具风险等级
    fn risk_level(&self) -> ToolRiskLevel {
        ToolRiskLevel::ReadOnly
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "Execute read-only SQL queries (SELECT only). Supports SQLite, MySQL, PostgreSQL. \
         Connection URL format: sqlite://path.db, mysql://user:pass@host/db, postgresql://user:pass@host/db"
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "connection_url": {
                    "type": "string",
                    "description": "Database connection URL (sqlite:///path.db | mysql://user:pass@host/db | postgresql://user:pass@host/db)"
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to execute (only SELECT / SHOW / DESCRIBE / EXPLAIN allowed)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_SQL_PAGE_ROWS,
                    "description": "Maximum rows per page (default 50)"
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from page.next_cursor; reuse only with identical parameters"
                }
            },
            "required": ["connection_url", "query"]
        })
    }

    /// 结合工具上下文执行工具逻辑
    fn execute_with_context<'a>(
        &'a self,
        parameters: ToolParameters,
        ctx: &'a echo_core::tools::ToolContext,
    ) -> BoxFuture<'a, Result<ToolResult>> {
        Box::pin(async move {
            let conn_url = parameters
                .get("connection_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("connection_url".to_string()))?;

            // 校验连接 URL 的 scheme
            validate_db_url(conn_url)?;

            let query = parameters
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("query".to_string()))?;
            let page_request = match PageRequest::from_parameters(
                &parameters,
                DEFAULT_SQL_PAGE_ROWS,
                MAX_SQL_PAGE_ROWS,
            ) {
                Ok(request) => request,
                Err(error) => return Ok(ToolResult::invalid_arguments(error.to_string())),
            };

            if let Err(error) = validate_readonly_query(query) {
                return Ok(ToolResult::error(error));
            }

            match execute_readonly_page(conn_url, query, &page_request).await {
                Ok(data) => {
                    let full_data = serde_json::json!({
                        "columns": data.columns,
                        "rows": data.rows,
                        "page": &data.page_info,
                    });
                    sql_page_result(ctx, full_data, data.page_info)
                }
                Err(e) => Ok(ToolResult::error(format!("Query failed: {}", e))),
            }
        })
    }
}

/// 校验查询是否为只读语句：必须以允许的关键字开头，且不含任何写入类关键字
fn validate_readonly_query(query: &str) -> std::result::Result<(), String> {
    let trimmed = query.trim().to_uppercase();
    let allowed = trimmed.starts_with("SELECT")
        || trimmed.starts_with("SHOW")
        || trimmed.starts_with("DESCRIBE")
        || trimmed.starts_with("DESC ")
        || trimmed.starts_with("EXPLAIN")
        || trimmed.starts_with("WITH");
    if !allowed {
        return Err(format!(
            "Only read-only queries allowed (SELECT/SHOW/DESCRIBE/EXPLAIN), received: {query}"
        ));
    }
    const MUTATING_SQL: &[&str] = &[
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "REPLACE",
        "EXECUTE",
        "EXEC",
        "INTO OUTFILE",
        "INTO DUMPFILE",
        "LOAD_FILE",
        "DBLINK",
        "LO_IMPORT",
        "PG_TERMINATE_BACKEND",
        "PG_CANCEL_BACKEND",
        "PG_RELOAD_CONF",
        "COPY",
        "PRAGMA",
    ];
    if let Some(keyword) = MUTATING_SQL
        .iter()
        .find(|keyword| trimmed.contains(**keyword))
    {
        return Err(format!(
            "Query contains forbidden keyword: {keyword}. Only read-only queries allowed."
        ));
    }
    Ok(())
}

// ── 列出数据表 ───────────────────────────────────────────────────────────────────

/// 列出数据库中所有表的工具
pub struct ListTablesTool;

impl Tool for ListTablesTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "list_tables"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "List all tables in the database. Supports SQLite, MySQL, PostgreSQL."
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "connection_url": {
                    "type": "string",
                    "description": "Database connection URL"
                }
            },
            "required": ["connection_url"]
        })
    }

    /// 执行工具逻辑
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let conn_url = parameters
                .get("connection_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("connection_url".to_string()))?;

            // 校验连接 URL 的 scheme
            validate_db_url(conn_url)?;

            // 根据数据库类型选择合适的查询语句
            let query = if conn_url.starts_with("sqlite") {
                "SELECT name AS table_name FROM sqlite_master WHERE type='table' ORDER BY name"
            } else if conn_url.starts_with("mysql") {
                "SELECT TABLE_NAME AS table_name FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME"
            } else {
                // PostgreSQL 及其他数据库
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            };

            match execute_readonly_query(conn_url, query).await {
                Ok(data) => Ok(ToolResult::success_json(data.into_json())),
                Err(e) => Ok(ToolResult::error(format!("List tables failed: {}", e))),
            }
        })
    }
}

// ── 查看表结构 ───────────────────────────────────────────────────────────────

/// 查看指定表结构（列名、类型、是否可空）的工具
pub struct DescribeTableTool;

impl Tool for DescribeTableTool {
    /// 返回工具名称
    fn name(&self) -> &str {
        "describe_table"
    }

    /// 返回工具所需权限
    fn permissions(&self) -> Vec<ToolPermission> {
        vec![ToolPermission::Network]
    }

    /// 返回工具描述
    fn description(&self) -> &str {
        "View the structure of a specified table (column names, types, nullable). Supports SQLite, MySQL, PostgreSQL."
    }

    /// 返回参数的 JSON Schema
    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "connection_url": {
                    "type": "string",
                    "description": "Database connection URL"
                },
                "table_name": {
                    "type": "string",
                    "description": "Table name to view"
                }
            },
            "required": ["connection_url", "table_name"]
        })
    }

    /// 执行工具逻辑
    fn execute(&self, parameters: ToolParameters) -> BoxFuture<'_, Result<ToolResult>> {
        Box::pin(async move {
            let conn_url = parameters
                .get("connection_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("connection_url".to_string()))?;

            // 校验连接 URL 的 scheme
            validate_db_url(conn_url)?;

            let table_name = parameters
                .get("table_name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::MissingParameter("table_name".to_string()))?;

            // 校验表名：只允许字母数字、下划线和点（用于 schema.table）
            if !table_name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
            {
                return Ok(ToolResult::error(format!(
                    "Invalid table name '{}': only alphanumeric, underscore, and dot characters allowed",
                    table_name
                )));
            }

            // 根据数据库类型选择合适的查询语句
            let query = if conn_url.starts_with("sqlite") {
                format!("PRAGMA table_info('{}')", table_name.replace('\'', "''"))
            } else if conn_url.starts_with("mysql") {
                format!(
                    "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT \
                     FROM information_schema.COLUMNS \
                     WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{}' \
                     ORDER BY ORDINAL_POSITION",
                    table_name.replace('\'', "''")
                )
            } else {
                // PostgreSQL
                format!(
                    "SELECT column_name, data_type, is_nullable, column_default \
                     FROM information_schema.columns \
                     WHERE table_name = '{}' \
                     ORDER BY ordinal_position",
                    table_name.replace('\'', "''")
                )
            };

            match execute_readonly_query(conn_url, &query).await {
                Ok(data) => Ok(ToolResult::success_json(data.into_json())),
                Err(e) => Ok(ToolResult::error(format!("Describe table failed: {}", e))),
            }
        })
    }
}

// ── 辅助函数 ───────────────────────────────────────────────────────────────────

/**
通过解析校验数据库连接 URL 的 scheme。

只允许 `sqlite://`、`mysql://`、`postgresql://` 和 `postgres://`。
对于 SQLite，还会校验路径中不含可疑字符。
*/
fn validate_db_url(url_str: &str) -> Result<()> {
    let (scheme, path) = url_str
        .split_once("://")
        .ok_or_else(|| ToolError::InvalidParameter {
            name: "connection_url".to_string(),
            message: "Invalid URL format: missing '://'".to_string(),
        })?;

    match scheme {
        "sqlite" | "mysql" | "postgresql" | "postgres" => {}
        _ => {
            return Err(ToolError::InvalidParameter {
                name: "connection_url".to_string(),
                message: format!(
                    "Unsupported database scheme '{}'. Use sqlite://, mysql://, or postgresql://",
                    scheme
                ),
            }
            .into());
        }
    }

    // 对 SQLite 校验路径不含可疑字符
    if scheme == "sqlite" {
        // 去掉开头的斜杠以兼容相对路径（sqlite:///path 与 sqlite://:memory:）
        let path = path.strip_prefix('/').unwrap_or(path);
        if path.contains('\0') || path.contains('\n') || path.contains('\r') {
            return Err(ToolError::InvalidParameter {
                name: "connection_url".to_string(),
                message: "SQLite path contains invalid characters".to_string(),
            }
            .into());
        }
    }

    Ok(())
}

/**
只读查询返回的结构化行数据。

纵深防御：非 SQLite 查询会包裹在显式的 READ ONLY 事务中，
由数据库本身拒绝任何写入尝试（而不只依赖关键字过滤）。
*/
struct DbRows {
    /// 列名列表
    columns: Vec<String>,
    /// 行数据，每行按列顺序存放 JSON 值
    rows: Vec<Vec<serde_json::Value>>,
}

/// 单页查询结果
struct DbPage {
    /// 列名列表
    columns: Vec<String>,
    /// 当前页的行数据
    rows: Vec<Vec<serde_json::Value>>,
    /// 分页信息（下一页游标、是否截断等）
    page_info: PageInfo,
}

/// SQL 分页游标的序列化载荷
#[derive(serde::Serialize, serde::Deserialize)]
struct SqlCursor {
    /// 游标格式版本，目前固定为 1
    version: u8,
    /// 连接 URL 与查询语句的指纹，防止游标被用于其它查询
    fingerprint: String,
    /// 下一页起始行偏移
    offset: usize,
    /// 生成游标时的每页行数
    limit: usize,
}

/// 计算连接 URL 与查询语句的 SHA-256 指纹
fn sql_query_fingerprint(conn_url: &str, query: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(conn_url.as_bytes());
    hasher.update([0]);
    hasher.update(query.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// 解码并校验分页游标，返回起始偏移；无游标时返回 0
fn decode_sql_cursor(request: &PageRequest, conn_url: &str, query: &str) -> Result<usize> {
    let Some(cursor) = request.cursor.as_deref() else {
        return Ok(0);
    };
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|error| ToolError::InvalidParameter {
            name: "cursor".to_string(),
            message: format!("invalid SQL cursor encoding: {error}"),
        })?;
    let decoded: SqlCursor =
        serde_json::from_slice(&bytes).map_err(|error| ToolError::InvalidParameter {
            name: "cursor".to_string(),
            message: format!("invalid SQL cursor payload: {error}"),
        })?;
    if decoded.version != 1
        || decoded.limit != request.limit
        || decoded.fingerprint != sql_query_fingerprint(conn_url, query)
    {
        return Err(ToolError::InvalidParameter {
            name: "cursor".to_string(),
            message: "SQL cursor does not match this query and limit".to_string(),
        }
        .into());
    }
    Ok(decoded.offset)
}

/// 将分页游标编码为 URL 安全的 base64 字符串
fn encode_sql_cursor(cursor: &SqlCursor) -> Result<String> {
    let bytes = serde_json::to_vec(cursor).map_err(|error| ToolError::ExecutionFailed {
        tool: "sql_query".to_string(),
        message: format!("failed to encode SQL cursor: {error}"),
    })?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

/// 以只读方式执行查询并返回一页结果（非 SQLite 使用 READ ONLY 事务）
async fn execute_readonly_page(
    conn_url: &str,
    query: &str,
    request: &PageRequest,
) -> Result<DbPage> {
    ensure_drivers_installed();
    let offset = decode_sql_cursor(request, conn_url, query)?;
    let pool = AnyPoolOptions::new()
        .max_connections(1)
        .connect(conn_url)
        .await
        .map_err(|error| ToolError::ExecutionFailed {
            tool: "database".to_string(),
            message: format!("Database connection failed: {error}"),
        })?;
    let take = request.limit.saturating_add(1);
    let rows = if conn_url.starts_with("sqlite") {
        collect_bounded_rows(sqlx::query(query).fetch(&pool), offset, take).await?
    } else {
        let mut transaction = pool
            .begin()
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "database".to_string(),
                message: format!("Failed to begin transaction: {error}"),
            })?;
        sqlx::query("SET TRANSACTION READ ONLY")
            .execute(&mut *transaction)
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "database".to_string(),
                message: format!("Failed to set read-only mode: {error}"),
            })?;
        let rows =
            collect_bounded_rows(sqlx::query(query).fetch(&mut *transaction), offset, take).await?;
        transaction
            .rollback()
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "database".to_string(),
                message: format!("Failed to close read-only transaction: {error}"),
            })?;
        rows
    };
    build_db_page(rows, request, offset, conn_url, query)
}

/// 从行流中跳过 `offset` 行后最多收集 `take` 行
async fn collect_bounded_rows<S>(
    mut stream: S,
    offset: usize,
    take: usize,
) -> Result<Vec<sqlx::any::AnyRow>>
where
    S: futures::Stream<Item = std::result::Result<sqlx::any::AnyRow, sqlx::Error>> + Unpin,
{
    let mut rows = Vec::with_capacity(take);
    let mut seen = 0usize;
    while let Some(row) = stream.next().await {
        let row = row.map_err(|error| ToolError::ExecutionFailed {
            tool: "database".to_string(),
            message: format!("Query execution failed: {error}"),
        })?;
        if seen < offset {
            seen = seen.saturating_add(1);
            continue;
        }
        rows.push(row);
        if rows.len() >= take {
            break;
        }
    }
    Ok(rows)
}

/// 根据多取一行的结果判断是否还有下一页，并构建分页结果
fn build_db_page(
    mut rows: Vec<sqlx::any::AnyRow>,
    request: &PageRequest,
    offset: usize,
    conn_url: &str,
    query: &str,
) -> Result<DbPage> {
    let has_more = rows.len() > request.limit;
    if has_more {
        rows.truncate(request.limit);
    }
    let formatted = format_db_rows(&rows)?;
    let next_offset = offset.saturating_add(formatted.rows.len());
    let next_cursor = if has_more {
        Some(encode_sql_cursor(&SqlCursor {
            version: 1,
            fingerprint: sql_query_fingerprint(conn_url, query),
            offset: next_offset,
            limit: request.limit,
        })?)
    } else {
        None
    };
    Ok(DbPage {
        columns: formatted.columns,
        rows: formatted.rows,
        page_info: PageInfo {
            next_cursor,
            truncated: has_more,
            total_known: false,
            total: None,
            returned: next_offset.saturating_sub(offset),
        },
    })
}

impl DbRows {
    /// 转换为包含列、行和总行数的 JSON
    fn into_json(self) -> serde_json::Value {
        let total_rows = self.rows.len();
        serde_json::json!({
            "columns": self.columns,
            "rows": self.rows,
            "total_rows": total_rows,
        })
    }
}

/// 以只读方式执行查询并返回全部结果行
async fn execute_readonly_query(conn_url: &str, query: &str) -> Result<DbRows> {
    ensure_drivers_installed();
    let pool = AnyPoolOptions::new()
        .max_connections(1)
        .connect(conn_url)
        .await
        .map_err(|e| ToolError::ExecutionFailed {
            tool: "database".to_string(),
            message: format!("Database connection failed: {}", e),
        })?;

    if conn_url.starts_with("sqlite") {
        // SQLite 不支持 SET TRANSACTION READ ONLY，跳过。
        let rows =
            sqlx::query(query)
                .fetch_all(&pool)
                .await
                .map_err(|e| ToolError::ExecutionFailed {
                    tool: "database".to_string(),
                    message: format!("Query execution failed: {}", e),
                })?;
        format_db_rows(&rows)
    } else {
        /* PostgreSQL/MySQL：包裹在显式的 READ ONLY 事务中。
        pool.begin() 会开启显式事务，随后 SET TRANSACTION READ ONLY
        作用于该事务（而不仅是自动提交内部的隐式事务）。 */
        let mut tx = pool.begin().await.map_err(|e| ToolError::ExecutionFailed {
            tool: "database".to_string(),
            message: format!("Failed to begin transaction: {}", e),
        })?;
        sqlx::query("SET TRANSACTION READ ONLY")
            .execute(&mut *tx)
            .await
            .map_err(|e| ToolError::ExecutionFailed {
                tool: "database".to_string(),
                message: format!("Failed to set read-only mode: {}", e),
            })?;
        let rows = sqlx::query(query).fetch_all(&mut *tx).await.map_err(|e| {
            ToolError::ExecutionFailed {
                tool: "database".to_string(),
                message: format!("Query execution failed: {}", e),
            }
        })?;
        /* 只读事务：显式回滚（没有需要写入的内容）。之前的注释写的是
        "否则在 drop 时回滚"，但代码实际执行了 commit；这里让两者保持一致（P1-1）。
        对只读事务而言 commit 与 rollback 等价，但 rollback 的意图更明确。 */
        tx.rollback().await.ok();
        format_db_rows(&rows)
    }
}

/// 将 sqlx 查询结果行格式化为结构化 JSON。
fn format_db_rows(rows: &[sqlx::any::AnyRow]) -> Result<DbRows> {
    let columns: Vec<String> = rows
        .first()
        .map(|row| row.columns().iter().map(|c| c.name().to_string()).collect())
        .unwrap_or_default();

    let col_count = columns.len();
    let mut row_values: Vec<Vec<serde_json::Value>> = Vec::with_capacity(rows.len());

    for row in rows {
        let mut values: Vec<serde_json::Value> = Vec::with_capacity(col_count);
        for (index, column) in row.columns().iter().enumerate() {
            values.push(decode_db_cell(
                row,
                index,
                column.name(),
                column.type_info().kind(),
            )?);
        }
        row_values.push(values);
    }

    Ok(DbRows {
        columns,
        rows: row_values,
    })
}

/// 按列类型将单个单元格解码为 JSON 值（BLOB 编码为 base64 对象）
fn decode_db_cell(
    row: &sqlx::any::AnyRow,
    index: usize,
    column_name: &str,
    kind: AnyTypeInfoKind,
) -> Result<serde_json::Value> {
    let decode_error = |error: sqlx::Error| ToolError::ExecutionFailed {
        tool: "database".to_string(),
        message: format!(
            "Failed to decode column '{}' as {:?}: {error}",
            column_name, kind
        ),
    };
    let value = match kind {
        AnyTypeInfoKind::Null => serde_json::Value::Null,
        AnyTypeInfoKind::Bool => match row
            .try_get::<Option<bool>, _>(index)
            .map_err(decode_error)?
        {
            Some(value) => serde_json::Value::Bool(value),
            None => serde_json::Value::Null,
        },
        AnyTypeInfoKind::SmallInt => {
            serde_json::json!(row.try_get::<Option<i16>, _>(index).map_err(decode_error)?)
        }
        AnyTypeInfoKind::Integer => {
            serde_json::json!(row.try_get::<Option<i32>, _>(index).map_err(decode_error)?)
        }
        AnyTypeInfoKind::BigInt => {
            serde_json::json!(row.try_get::<Option<i64>, _>(index).map_err(decode_error)?)
        }
        AnyTypeInfoKind::Real => {
            serde_json::json!(row.try_get::<Option<f32>, _>(index).map_err(decode_error)?)
        }
        AnyTypeInfoKind::Double => {
            serde_json::json!(row.try_get::<Option<f64>, _>(index).map_err(decode_error)?)
        }
        AnyTypeInfoKind::Text => match row
            .try_get::<Option<String>, _>(index)
            .map_err(decode_error)?
        {
            Some(value) => serde_json::Value::String(value),
            None => serde_json::Value::Null,
        },
        AnyTypeInfoKind::Blob => match row
            .try_get::<Option<Vec<u8>>, _>(index)
            .map_err(decode_error)?
        {
            Some(bytes) => serde_json::json!({
                "type": "binary",
                "encoding": "base64",
                "data": base64::engine::general_purpose::STANDARD.encode(bytes),
            }),
            None => serde_json::Value::Null,
        },
    };
    Ok(value)
}

/// 构建 SQL 分页结果：若配置了 artifact 则持久化完整输出，并只内联截断后的预览
fn sql_page_result(
    ctx: &echo_core::tools::ToolContext,
    full_data: serde_json::Value,
    page_info: echo_core::tools::pagination::PageInfo,
) -> Result<ToolResult> {
    let full_output =
        serde_json::to_string(&full_data).map_err(|error| ToolError::ExecutionFailed {
            tool: "sql_query".to_string(),
            message: format!("Failed to serialize query page: {error}"),
        })?;
    let mut artifact_error = None;
    let artifact = match ctx.output_artifacts.clone() {
        Some(config) => match persist_tool_output(
            config,
            ToolOutputArtifactIdentity::from_context(ctx, "sql_query"),
            &full_output,
        ) {
            Ok(artifact) => artifact,
            Err(error) => {
                tracing::warn!(%error, "sql_query output artifact write failed");
                artifact_error = Some(error.to_string());
                None
            }
        },
        None => None,
    };

    let visible_data = if artifact.is_some() {
        project_sql_data(&full_data)
    } else {
        full_data
    };
    let mut result = ToolResult::success_json(visible_data);
    if let Some(artifact) = artifact {
        apply_sql_artifact(artifact, full_output.len(), &mut result);
    } else if let Some(error) = artifact_error {
        result
            .metadata
            .insert("artifact_status".to_string(), "write_failed".to_string());
        result.metadata.insert("artifact_error".to_string(), error);
    }
    page_info.apply_to(&mut result);
    Ok(result)
}

/// 生成内联预览数据：截断过长的列名与单元格文本
fn project_sql_data(data: &serde_json::Value) -> serde_json::Value {
    let columns = data
        .get("columns")
        .and_then(Value::as_array)
        .map(|columns| {
            columns
                .iter()
                .map(|column| match column.as_str() {
                    Some(column) => Value::String(char_preview(column, MAX_INLINE_COLUMN_CHARS)),
                    None => column.clone(),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let rows = data
        .get("rows")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    Value::Array(
                        row.as_array()
                            .map(|cells| {
                                cells
                                    .iter()
                                    .map(|cell| match cell.as_str() {
                                        Some(cell) => {
                                            Value::String(char_preview(cell, MAX_INLINE_CELL_CHARS))
                                        }
                                        None => cell.clone(),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::json!({
        "columns": columns,
        "rows": rows,
        "page": data.get("page").cloned().unwrap_or(Value::Null),
        "complete_page": "stored in artifact metadata",
    })
}

/// 按字符数截断字符串，超出时追加 "..."
fn char_preview(value: &str, max_chars: usize) -> String {
    let mut preview: String = value.chars().take(max_chars).collect();
    if value.chars().count() > max_chars {
        preview.push_str("...");
    }
    preview
}

/// 将 artifact 引用及溢出相关元数据写入工具结果
fn apply_sql_artifact(
    artifact: ToolOutputArtifactRef,
    original_bytes: usize,
    result: &mut ToolResult,
) {
    result.artifact = Some(artifact);
    result
        .metadata
        .insert("output_handling".to_string(), "spilled".to_string());
    result
        .metadata
        .insert("original_bytes".to_string(), original_bytes.to_string());
    result.metadata.insert(
        "returned_bytes".to_string(),
        result.output.len().to_string(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use echo_core::error::ReactError;

    /// SQLite 分页结果有界，且保留各类型单元格的值
    #[tokio::test]
    async fn sqlite_page_is_bounded_and_preserves_typed_cells() -> Result<()> {
        let directory = tempfile::tempdir().map_err(ReactError::from)?;
        let database = directory.path().join("typed.sqlite");
        let url = format!("sqlite://{}?mode=rwc", database.display());
        ensure_drivers_installed();
        let pool = AnyPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "database-test".to_string(),
                message: error.to_string(),
            })?;
        sqlx::query(
            "CREATE TABLE samples (id INTEGER, flag INTEGER, score REAL, label TEXT, payload BLOB, optional INTEGER)",
        )
        .execute(&pool)
        .await
        .map_err(|error| ToolError::ExecutionFailed {
            tool: "database-test".to_string(),
            message: error.to_string(),
        })?;
        for id in 1_i64..=5_i64 {
            sqlx::query(
                "INSERT INTO samples (id, flag, score, label, payload, optional) VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(id)
            .bind(id % 2 == 0)
            .bind(id as f64 + 0.5)
            .bind(format!("row-{id}"))
            .bind(vec![u8::try_from(id).unwrap_or_default(), 255])
            .bind((id != 1).then_some(id))
            .execute(&pool)
            .await
            .map_err(|error| ToolError::ExecutionFailed {
                tool: "database-test".to_string(),
                message: error.to_string(),
            })?;
        }
        pool.close().await;

        let first_request = PageRequest {
            limit: 2,
            cursor: None,
        };
        let first = execute_readonly_page(
            &url,
            "SELECT id, flag, score, label, payload, optional FROM samples ORDER BY id",
            &first_request,
        )
        .await?;
        assert_eq!(first.rows.len(), 2);
        assert_eq!(
            first.rows.first().and_then(|row| row.first()),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            first.rows.first().and_then(|row| row.get(5)),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(
            first
                .rows
                .first()
                .and_then(|row| row.get(4))
                .and_then(|value| value.get("encoding")),
            Some(&serde_json::json!("base64"))
        );
        let cursor = first.page_info.next_cursor.ok_or_else(|| {
            ReactError::Other("first SQL page did not return a cursor".to_string())
        })?;

        let second = execute_readonly_page(
            &url,
            "SELECT id, flag, score, label, payload, optional FROM samples ORDER BY id",
            &PageRequest {
                limit: 2,
                cursor: Some(cursor),
            },
        )
        .await?;
        assert_eq!(second.rows.len(), 2);
        assert_eq!(
            second.rows.first().and_then(|row| row.first()),
            Some(&serde_json::json!(3))
        );
        assert!(second.page_info.next_cursor.is_some());
        assert!(!second.page_info.total_known);
        Ok(())
    }

    /// 写入类 SQL 永远不会被视为只读查询
    #[test]
    fn writable_sql_never_qualifies_as_read_only() {
        use echo_core::tools::Tool;

        assert_eq!(SqlQueryTool.risk_level(), ToolRiskLevel::ReadOnly);
        assert!(validate_readonly_query("SELECT * FROM samples").is_ok());
        assert!(validate_readonly_query("UPDATE samples SET flag = true").is_err());
        assert!(
            validate_readonly_query(
                "WITH changed AS (DELETE FROM samples RETURNING id) SELECT * FROM changed"
            )
            .is_err()
        );
    }
}
