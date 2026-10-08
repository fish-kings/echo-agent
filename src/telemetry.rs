/*!
OpenTelemetry 集成

提供 OTLP 导出配置和初始化函数，与现有的 `tracing` 基础设施集成。
同时提供 Metrics 基础设施（Counter / Histogram），用于记录 LLM 调用、
Token 使用量、工具执行等关键指标。

# 用法

```rust,no_run
use echo_agent::telemetry::{TelemetryConfig, init_telemetry, shutdown_telemetry};

# fn main() -> echo_agent::error::Result<()> {
init_telemetry(TelemetryConfig::default())?;

// ... Run Agent ...

shutdown_telemetry()?;
# Ok(())
# }
```
*/

use crate::error::{ReactError, Result};
use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, MeterProvider};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::TracerProvider;
use std::sync::{Mutex, OnceLock};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// ── 配置 ───────────────────────────────────────────────────────────────────

/// OpenTelemetry 配置
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// OTLP 端点（gRPC）
    pub otlp_endpoint: String,
    /// 服务名称
    pub service_name: String,
    /// 是否同时输出到控制台
    pub enable_console: bool,
}

impl Default for TelemetryConfig {
    /// 创建默认实例。
    fn default() -> Self {
        Self {
            otlp_endpoint: "http://localhost:4317".to_string(),
            service_name: "echo-agent".to_string(),
            enable_console: true,
        }
    }
}

// ── Metrics 指标 ──────────────────────────────────────────────────────────────

/// 全局 Metrics 实例（懒初始化）
static METRICS: OnceLock<Metrics> = OnceLock::new();
/// 全局遥测运行时状态（懒初始化）
static TELEMETRY_RUNTIME: OnceLock<TelemetryRuntime> = OnceLock::new();
/// 保护初始化过程的互斥锁，防止并发重复初始化
static TELEMETRY_INIT_LOCK: Mutex<()> = Mutex::new(());

/// 保存 tracer/meter provider 的运行时句柄，用于 shutdown 时统一关闭
struct TelemetryRuntime {
    /// OpenTelemetry 链路追踪 provider
    tracer_provider: TracerProvider,
    /// OpenTelemetry 指标 provider
    meter_provider: SdkMeterProvider,
}

/**
Agent 运行时指标

提供 LLM 调用、Token 使用量、工具执行等核心指标的记录接口。
所有指标都会通过 OTLP Metrics exporter 导出。
*/
#[derive(Debug, Clone)]
pub struct Metrics {
    /// LLM 调用计数器（标签：provider、model、status）
    llm_calls: Counter<u64>,
    /// LLM Token 使用量计数器（标签：provider、model、type=input|output）
    llm_tokens: Counter<u64>,
    /// LLM 调用延迟直方图（标签：provider、model）
    llm_latency: Histogram<f64>,
    /// 工具执行计数器（标签：tool、status）
    tool_executions: Counter<u64>,
    /// 工具执行延迟直方图（标签：tool）
    tool_latency: Histogram<f64>,
}

impl Metrics {
    /// 获取全局 Metrics 实例
    pub fn get() -> Option<&'static Metrics> {
        METRICS.get()
    }

    /// 记录一次 LLM 调用
    pub fn record_llm_call(provider: &str, model: &str, status: &str) {
        if let Some(m) = Self::get() {
            m.llm_calls.add(
                1,
                &[
                    KeyValue::new("provider", provider.to_string()),
                    KeyValue::new("model", model.to_string()),
                    KeyValue::new("status", status.to_string()),
                ],
            );
        }
    }

    /// 记录 LLM Token 使用量
    pub fn record_llm_tokens(provider: &str, model: &str, token_type: &str, count: u64) {
        if let Some(m) = Self::get() {
            m.llm_tokens.add(
                count,
                &[
                    KeyValue::new("provider", provider.to_string()),
                    KeyValue::new("model", model.to_string()),
                    KeyValue::new("type", token_type.to_string()),
                ],
            );
        }
    }

    /// 记录 LLM 调用延迟（毫秒）
    pub fn record_llm_latency(provider: &str, model: &str, duration_ms: f64) {
        if let Some(m) = Self::get() {
            m.llm_latency.record(
                duration_ms,
                &[
                    KeyValue::new("provider", provider.to_string()),
                    KeyValue::new("model", model.to_string()),
                ],
            );
        }
    }

    /// 记录一次工具执行
    pub fn record_tool_execution(tool: &str, status: &str) {
        if let Some(m) = Self::get() {
            m.tool_executions.add(
                1,
                &[
                    KeyValue::new("tool", tool.to_string()),
                    KeyValue::new("status", status.to_string()),
                ],
            );
        }
    }

    /// 记录工具执行延迟（毫秒）
    pub fn record_tool_latency(tool: &str, duration_ms: f64) {
        if let Some(m) = Self::get() {
            m.tool_latency
                .record(duration_ms, &[KeyValue::new("tool", tool.to_string())]);
        }
    }
}

// ── 初始化 / 关闭 ──────────────────────────────────────────────────────────

/**
初始化 OpenTelemetry 的 tracing 和 metrics。

向全局 subscriber 注册 OTLP exporter 和 tracing-opentelemetry layer，
并初始化 MeterProvider 和全局 Metrics。

如果 `enable_console` 为 true，还会额外注册一个 fmt layer。
*/
pub fn init_telemetry(config: TelemetryConfig) -> Result<()> {
    let _init_guard = TELEMETRY_INIT_LOCK.lock().map_err(|error| {
        ReactError::Other(format!("telemetry initialization lock poisoned: {error}"))
    })?;
    if TELEMETRY_RUNTIME.get().is_some() {
        return Err(ReactError::Other(
            "Telemetry is already initialized".to_string(),
        ));
    }
    // ── Tracing 追踪 ──
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build()
        .map_err(|e| ReactError::Other(format!("OTLP exporter error: {e}")))?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(opentelemetry_sdk::Resource::new(vec![KeyValue::new(
            "service.name",
            config.service_name.clone(),
        )]))
        .build();

    let tracer = provider.tracer("echo-agent");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let (metrics, meter_provider) = build_metrics(&config)?;

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if config.enable_console {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .without_time()
            .with_target(false);
        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .with(fmt_layer)
            .try_init()
            .map_err(|error| {
                ReactError::Other(format!("tracing subscriber init failed: {error}"))
            })?;
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(otel_layer)
            .try_init()
            .map_err(|error| {
                ReactError::Other(format!("tracing subscriber init failed: {error}"))
            })?;
    }

    // 只有在 exporter 和 subscriber 都准备就绪之后才发布全局实例。
    METRICS
        .set(metrics)
        .map_err(|_| ReactError::Other("Metrics already initialized".to_string()))?;
    TELEMETRY_RUNTIME
        .set(TelemetryRuntime {
            tracer_provider: provider,
            meter_provider,
        })
        .map_err(|_| ReactError::Other("Telemetry is already initialized".to_string()))?;
    tracing::info!(endpoint = %config.otlp_endpoint, "Metrics initialized");

    Ok(())
}

/// 初始化 Metrics（MeterProvider + 全局 Instruments）
fn build_metrics(config: &TelemetryConfig) -> Result<(Metrics, SdkMeterProvider)> {
    let metrics_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build()
        .map_err(|e| ReactError::Other(format!("OTLP metrics exporter error: {e}")))?;

    let reader =
        PeriodicReader::builder(metrics_exporter, opentelemetry_sdk::runtime::Tokio).build();

    let meter_provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(opentelemetry_sdk::Resource::new(vec![KeyValue::new(
            "service.name",
            config.service_name.clone(),
        )]))
        .build();

    let meter = meter_provider.meter("echo-agent");

    let metrics = Metrics {
        llm_calls: meter
            .u64_counter("llm.calls")
            .with_description("Number of LLM API calls")
            .build(),
        llm_tokens: meter
            .u64_counter("llm.tokens")
            .with_description("Number of tokens consumed")
            .with_unit("tokens")
            .build(),
        llm_latency: meter
            .f64_histogram("llm.latency")
            .with_description("LLM API call latency in milliseconds")
            .with_unit("ms")
            .build(),
        tool_executions: meter
            .u64_counter("tool.executions")
            .with_description("Number of tool executions")
            .build(),
        tool_latency: meter
            .f64_histogram("tool.latency")
            .with_description("Tool execution latency in milliseconds")
            .with_unit("ms")
            .build(),
    };

    Ok((metrics, meter_provider))
}

/// 关闭 OpenTelemetry，刷新未发送的 span 和 metrics
pub fn shutdown_telemetry() -> Result<()> {
    let runtime = TELEMETRY_RUNTIME
        .get()
        .ok_or_else(|| ReactError::Other("Telemetry is not initialized".to_string()))?;
    runtime
        .meter_provider
        .shutdown()
        .map_err(|error| ReactError::Other(format!("metrics shutdown failed: {error}")))?;
    runtime
        .tracer_provider
        .shutdown()
        .map_err(|error| ReactError::Other(format!("tracing shutdown failed: {error}")))?;
    Ok(())
}
