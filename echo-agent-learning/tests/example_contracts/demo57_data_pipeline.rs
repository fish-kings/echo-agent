/*! demo57：以代码为先、基于文件的数据分析流水线。

真实的流水线需要一个具备文件工具和 `run_code` 能力的 Agent。本示例
使用 `MockAgent` 在不实际写文件的情况下，演示其公开的配置与状态契约。

契约测试：`contract_demo57_data_pipeline`（需要 `testing` feature）。
*/

use echo_agent::error::Result;
use echo_agent::testing::MockAgent;
use echo_agent::workflow::pipelines::{
    DataPipelineConfig, DataPipelineLanguage, run_data_pipeline,
};
use echo_agent::workflow::shared_agent;

/// 契约测试：验证数据分析流水线在使用 mock agent 时的配置与状态契约保持稳定。
#[tokio::test]
async fn contract_demo57_data_pipeline() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info,echo_orchestration=debug")
        .try_init();

    println!("demo57: code-first data analysis pipeline\n");

    let config = DataPipelineConfig::new("data/sales_2024.csv")
        .with_objective("Identify revenue trends and seasonal patterns")
        .with_artifact_dir("analysis/revenue-trends")
        .with_language(DataPipelineLanguage::Python)
        .with_max_charts(5)
        .with_random_seed(42)
        .with_parameters(serde_json::json!({"group_by": "region"}));

    println!("dataset_path : {}", config.dataset_path);
    println!("artifact_dir : {}", config.artifact_dir);
    println!("language     : {}", config.language.as_str());
    println!("random_seed  : {:?}", config.random_seed);
    println!("max_charts   : {}\n", config.max_charts);

    let mock = MockAgent::new("data_analyst").with_response(
        "Created analysis/revenue-trends/manifest.json and analysis.py, executed the persisted \
         script, and verified result.json, environment.json, outputs/, and latest-run.json.",
    );
    let result = run_data_pipeline(&shared_agent(mock), config).await?;

    println!("steps : {}", result.steps);
    println!("path  : {:?}", result.path);
    println!(
        "script: {}",
        result
            .state
            .get::<String>("script_path")
            .unwrap_or_default()
    );
    println!(
        "summary:\n{}",
        result.state.get::<String>("summary").unwrap_or_default()
    );
    println!(
        "\nThe pipeline uses one tool-capable Agent execution. The persisted script and observed run \
         artifacts, rather than intermediate prose, are the source of truth."
    );

    Ok(())
}
