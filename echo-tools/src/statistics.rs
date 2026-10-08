/*!
 * 探索性统计摘要。
 *
 * 此处刻意不实现正式的统计推断。Agent 应编写可审查的 Python/R 脚本，
 * 使用 SciPy、statsmodels 或成熟的 R 包等库，再通过沙箱化的
 * `run_code` 路径执行这些脚本。本模块只提供在选择模型或检验方法之前
 * 有用的描述性统计值。
 */

use std::cmp::Ordering;

use polars::prelude::*;

use crate::data::{is_numeric, load_dataframe};
use crate::security::SecurityConfig;
use echo_core::error::{Result, ToolError};
use echo_core::tools::{ToolResult, ToolRunner};

/// 工具名称
const TOOL_NAME: &str = "exploratory_statistics";

/// 描述性统计工具，刻意不做任何推断性结论。
#[derive(Default, echo_macros::Tool)]
#[tool(
    name = "exploratory_statistics",
    description = "Compute exploratory descriptive statistics for numeric columns: valid/missing counts, mean, sample standard deviation, min, quartiles, max, skewness, and excess kurtosis. This tool never returns p-values, confidence intervals, or significance conclusions. For formal inference, write a reviewable Python/R script using SciPy, statsmodels, or established R packages and execute it with run_code."
)]
/** 派生宏使用这些字段生成参数类型和 Schema 类型；
零大小的 Tool 值本身并不直接读取它们。 */
#[allow(dead_code)]
pub struct ExploratoryStatisticsTool {
    /// 数据文件的绝对路径
    #[tool_param(description = "Absolute path to the data file (CSV, JSON, or Parquet)")]
    data_path: String,
    /// 要分析的列名（逗号分隔），缺省为全部数值列
    #[tool_param(
        description = "Column names to analyze, comma-separated (default: all numeric columns)"
    )]
    columns: Option<String>,
}

impl ToolRunner<ExploratoryStatisticsToolParams> for ExploratoryStatisticsTool {
    /// 加载数据并计算所选数值列的描述性统计量。
    async fn run(&self, params: ExploratoryStatisticsToolParams) -> Result<ToolResult> {
        let security = SecurityConfig::global();
        let path = security.validate_file(&params.data_path)?;
        let frame = load_dataframe(&path, None)?;
        let target_columns = selected_numeric_columns(&frame, params.columns.as_deref());
        if target_columns.is_empty() {
            return Ok(ToolResult::success(
                "No numeric columns found for exploratory statistics",
            ));
        }

        let mut summaries = Vec::new();
        for column_name in target_columns {
            let column =
                frame
                    .column(&column_name)
                    .map_err(|error| ToolError::ExecutionFailed {
                        tool: TOOL_NAME.to_string(),
                        message: format!("Column '{column_name}' not found: {error}"),
                    })?;
            if !is_numeric(column.dtype()) {
                summaries.push(serde_json::json!({
                    "name": column_name,
                    "error": "Non-numeric column"
                }));
                continue;
            }

            let casted =
                column
                    .cast(&DataType::Float64)
                    .map_err(|error| ToolError::ExecutionFailed {
                        tool: TOOL_NAME.to_string(),
                        message: format!("Cast column '{column_name}' failed: {error}"),
                    })?;
            let mut values: Vec<f64> = casted
                .f64()
                .map_err(|error| ToolError::ExecutionFailed {
                    tool: TOOL_NAME.to_string(),
                    message: format!("Convert column '{column_name}' to f64 failed: {error}"),
                })?
                .iter()
                .flatten()
                .filter(|value| value.is_finite())
                .collect();
            values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));

            let total_count = column.len();
            let valid_count = values.len();
            let missing_count = total_count.saturating_sub(valid_count);
            if valid_count == 0 {
                summaries.push(serde_json::json!({
                    "name": column_name,
                    "total_count": total_count,
                    "valid_count": 0,
                    "missing_or_non_finite_count": missing_count,
                    "error": "No finite numeric values"
                }));
                continue;
            }

            let mean = values.iter().sum::<f64>() / valid_count as f64;
            let sample_std_dev = sample_standard_deviation(&values, mean);
            let (skewness, excess_kurtosis) = standardized_moments(&values, mean);
            summaries.push(serde_json::json!({
                "name": column_name,
                "total_count": total_count,
                "valid_count": valid_count,
                "missing_or_non_finite_count": missing_count,
                "mean": mean,
                "sample_std_dev": sample_std_dev,
                "min": values.first().copied(),
                "p25": quantile(&values, 0.25),
                "median": quantile(&values, 0.5),
                "p75": quantile(&values, 0.75),
                "max": values.last().copied(),
                "skewness_moment": skewness,
                "excess_kurtosis_moment": excess_kurtosis,
            }));
        }

        Ok(ToolResult::success_json(serde_json::json!({
            "contract_version": 1,
            "analysis_kind": "exploratory_descriptive",
            "inference": false,
            "file": path.display().to_string(),
            "columns": summaries,
            "limitations": [
                "Moment skewness and kurtosis are exploratory estimators.",
                "No p-values, confidence intervals, causal claims, or significance decisions are produced.",
                "Use a persisted SciPy/statsmodels/R script through run_code for formal inference."
            ]
        })))
    }
}

/// 返回要分析的列：显式指定的列，或数据框中的全部数值列。
fn selected_numeric_columns(frame: &DataFrame, requested: Option<&str>) -> Vec<String> {
    match requested {
        Some(columns) => columns
            .split(',')
            .map(str::trim)
            .filter(|column| !column.is_empty())
            .map(|name| name.to_string())
            .collect(),
        None => frame
            .get_column_names()
            .into_iter()
            .filter(|name| {
                frame
                    .column(name)
                    .map(|column| is_numeric(column.dtype()))
                    .unwrap_or(false)
            })
            .map(|name| name.to_string())
            .collect(),
    }
}

/// 计算样本标准差（n - 1 为分母），少于两个值时返回 None。
fn sample_standard_deviation(values: &[f64], mean: f64) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let sum_of_squares = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    Some((sum_of_squares / values.len().saturating_sub(1) as f64).sqrt())
}

/// 计算偏度与超额峰度，少于三个值时返回 (None, None)。
fn standardized_moments(values: &[f64], mean: f64) -> (Option<f64>, Option<f64>) {
    if values.len() < 3 {
        return (None, None);
    }
    let count = values.len() as f64;
    let second_moment = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    if second_moment <= f64::EPSILON {
        return (Some(0.0), Some(0.0));
    }
    let third_moment = values
        .iter()
        .map(|value| (value - mean).powi(3))
        .sum::<f64>()
        / count;
    let fourth_moment = values
        .iter()
        .map(|value| (value - mean).powi(4))
        .sum::<f64>()
        / count;
    (
        Some(third_moment / second_moment.powf(1.5)),
        Some(fourth_moment / second_moment.powi(2) - 3.0),
    )
}

/// 对已排序的数据按线性插值计算分位数，空输入返回 None。
pub(crate) fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let bounded_probability = probability.clamp(0.0, 1.0);
    let max_index = values.len().saturating_sub(1);
    let position = bounded_probability * max_index as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let lower = values.get(lower_index).copied()?;
    let upper = values.get(upper_index).copied()?;
    let fraction = position - lower_index as f64;
    Some(lower + (upper - lower) * fraction)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use echo_core::error::ReactError;
    use echo_core::tools::{Tool, ToolParameters};

    use super::*;

    /// 探索性统计结果从不包含推断性结论。
    #[tokio::test]
    async fn exploratory_statistics_never_claims_inference() -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "echo-exploratory-statistics-{}-{timestamp}.csv",
            std::process::id()
        ));
        fs::write(&path, "value\n1\n2\n3\n4\n")?;
        let mut parameters = ToolParameters::new();
        parameters.insert(
            "data_path".to_string(),
            serde_json::Value::String(path.display().to_string()),
        );

        let result = ExploratoryStatisticsTool::default()
            .execute(parameters)
            .await?;
        assert!(result.success);
        let data = result
            .data
            .ok_or_else(|| ReactError::Other("missing structured output".to_string()))?;
        assert_eq!(
            data.get("inference").and_then(serde_json::Value::as_bool),
            Some(false)
        );
        let rendered = data.to_string();
        assert!(!rendered.contains("p_value"));
        assert!(!rendered.contains("significant"));
        assert!(!rendered.contains("confidence_interval"));
        fs::remove_file(path)?;
        Ok(())
    }

    /// 分位数使用线性插值且不会发生索引越界。
    #[test]
    fn quantiles_use_linear_interpolation_without_index_panics() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&values, 0.25), Some(1.75));
        assert_eq!(quantile(&values, 0.5), Some(2.5));
        assert_eq!(quantile(&values, 0.75), Some(3.25));
        assert_eq!(quantile(&[], 0.5), None);
    }
}
