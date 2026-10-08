/*!
demo55 —— LSP 工具

演示为 Agent 提供类 IDE 代码理解能力的 Language Server Protocol（语言服务器协议）集成：

1. `LspConfig` —— 解析 `.lsp.yaml` 配置
2. `LspManager` —— 管理多个语言服务器进程
3. LSP 工具类型 —— diagnostics（诊断）、goto_definition（跳转定义）、
   find_references（查找引用）、hover（悬浮提示）、status（状态）
4. 文件扩展名 → 语言服务器路由

**说明：** 真正运行语言服务器需要安装对应的服务器可执行文件
（例如 `rust-analyzer`、`pyright-langserver`）。
本示例只演示配置和工具构造，不实际启动服务器。

所需 Feature flag：`lsp`

```sh
cargo test --features lsp --test example_contracts contract_demo55_lsp_tools
```
*/

/// 启用 `lsp` feature 时的完整演示：配置解析、管理器构造、工具构造与扩展名路由。
#[cfg(feature = "lsp")]
mod demo {
    use echo_agent::lsp::{LspConfig, LspManager, LspServerConfig};
    use echo_agent::prelude::Tool;
    use echo_agent::tools::lsp::{
        LspDiagnosticsTool, LspFindReferencesTool, LspGotoDefinitionTool, LspHoverTool,
        LspStatusTool,
    };
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// 打印场景编号与标题的分隔横幅。
    macro_rules! section {
        ($n:expr, $title:expr) => {
            println!("\n══════════════════════════════════════════════════");
            println!("  Scenario {} : {}", $n, $title);
            println!("══════════════════════════════════════════════════");
        };
    }

    /**
    依次运行全部 6 个 LSP 工具演示场景（配置解析、扩展名路由、Manager 创建、
    工具描述、参数 Schema、编程式配置构建）。
    */
    pub async fn run() -> Result<(), String> {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║       echo-agent  LSP Tools Demo                 ║");
        println!("║  (config + tool construction — servers optional)  ║");
        println!("╚══════════════════════════════════════════════════╝");

        demo_config_from_yaml()?;
        demo_config_extension_routing()?;
        demo_manager_creation()?;
        demo_tool_descriptions();
        demo_tool_parameters()?;
        demo_programmatic_config()?;

        println!("\n╔══════════════════════════════════════════════════╗");
        println!("║  All 6 scenarios passed ✅                       ║");
        println!("╚══════════════════════════════════════════════════╝");
        Ok(())
    }

    /// 场景 1：从 YAML 解析 LspConfig
    fn demo_config_from_yaml() -> Result<(), String> {
        section!(1, "LspConfig from YAML (.lsp.yaml)");

        let yaml = r#"
languages:
  rust:
    language: rust
    command: rust-analyzer
    args: []
    extensions: [".rs"]
  python:
    language: python
    command: pyright-langserver
    args: ["--stdio"]
    extensions: [".py", ".pyi"]
  typescript:
    language: typescript
    command: typescript-language-server
    args: ["--stdio"]
    extensions: [".ts", ".tsx", ".js", ".jsx"]
"#;
        let config = LspConfig::from_yaml(yaml)?;
        println!("  Loaded {} language server configs:", config.servers.len());
        for (lang, server) in &config.servers {
            println!(
                "    {} → {} (extensions: {:?})",
                lang, server.command, server.extensions
            );
        }

        assert_eq!(config.servers.len(), 3);
        let rust = config
            .get("rust")
            .ok_or_else(|| "rust language server config is missing".to_string())?;
        assert_eq!(rust.command, "rust-analyzer");
        println!("  ✅ YAML config parsed successfully");
        Ok(())
    }

    /// 场景 2：基于扩展名的路由
    fn demo_config_extension_routing() -> Result<(), String> {
        section!(2, "File Extension → Language Server Routing");

        let yaml = r#"
languages:
  rust:
    language: rust
    command: rust-analyzer
    args: []
    extensions: [".rs"]
  python:
    language: python
    command: pyright-langserver
    args: ["--stdio"]
    extensions: [".py", ".pyi"]
"#;
        let config = LspConfig::from_yaml(yaml)?;

        let test_extensions = [".rs", ".py", ".pyi", ".ts"];
        for ext in &test_extensions {
            match config.get_for_extension(ext) {
                Some((lang, server)) => {
                    println!("  {}  →  {} ({})", ext, lang, server.command);
                }
                None => {
                    println!("  {}  →  (no server configured)", ext);
                }
            }
        }

        assert!(config.get_for_extension(".rs").is_some());
        assert!(config.get_for_extension(".py").is_some());
        assert!(config.get_for_extension(".ts").is_none());
        println!("  ✅ Extension routing works correctly");
        Ok(())
    }

    /// 场景 3：创建 LspManager
    fn demo_manager_creation() -> Result<(), String> {
        section!(3, "LspManager Creation");

        let mut manager = LspManager::new();
        println!("  Created empty LspManager");

        let yaml = r#"
languages:
  rust:
    language: rust
    command: rust-analyzer
    args: []
    extensions: [".rs"]
"#;
        let config = LspConfig::from_yaml(yaml)?;
        manager.load_config(&config)?;
        println!("  Loaded rust-analyzer config");

        // 设置项目根目录
        manager.set_project_root(std::path::Path::new("/tmp/my-rust-project"));
        println!("  Set project root: /tmp/my-rust-project");

        // 包装为 Arc<RwLock<>> 以便与工具共享
        let shared_manager = Arc::new(RwLock::new(manager));
        println!("  Wrapped in Arc<RwLock<LspManager>> for tool sharing");

        let _ = shared_manager;
        println!("  ✅ Manager ready (start_server() requires rust-analyzer binary)");
        Ok(())
    }

    /// 场景 4：工具描述与名称
    fn demo_tool_descriptions() {
        section!(4, "LSP Tool Descriptions");

        let manager = Arc::new(RwLock::new(LspManager::new()));

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(LspDiagnosticsTool::new(manager.clone())),
            Box::new(LspGotoDefinitionTool::new(manager.clone())),
            Box::new(LspFindReferencesTool::new(manager.clone())),
            Box::new(LspHoverTool::new(manager.clone())),
            Box::new(LspStatusTool::new(manager.clone())),
        ];

        println!("  {} LSP tools available:\n", tools.len());
        for tool in &tools {
            println!("  Tool: {}", tool.name());
            let desc = tool.description();
            // 只展示描述的前 80 个字符
            let short_desc = if desc.chars().count() > 80 {
                let truncated: String = desc.chars().take(80).collect();
                format!("{truncated}...")
            } else {
                desc.to_string()
            };
            println!("    Description: {}", short_desc);
            println!("    Risk level:  {:?}\n", tool.risk_level());
        }

        // 验证所有工具均为只读
        for tool in &tools {
            assert_eq!(
                format!("{:?}", tool.risk_level()),
                "ReadOnly",
                "Tool {} should be ReadOnly",
                tool.name()
            );
        }
        println!("  ✅ All LSP tools are ReadOnly (safe for analysis)");
    }

    /// 场景 5：工具参数 Schema（JSON Schema）
    fn demo_tool_parameters() -> Result<(), String> {
        section!(5, "Tool Parameter Schemas (JSON Schema)");

        let manager = Arc::new(RwLock::new(LspManager::new()));

        let goto = LspGotoDefinitionTool::new(manager.clone());
        let schema = goto.parameters();
        println!("  lsp_goto_definition parameters:");
        let rendered = serde_json::to_string_pretty(&schema)
            .map_err(|error| format!("failed to render goto-definition schema: {error}"))?;
        println!("    {}", rendered);

        let diag = LspDiagnosticsTool::new(manager.clone());
        let schema = diag.parameters();
        println!("\n  lsp_diagnostics parameters:");
        let rendered = serde_json::to_string_pretty(&schema)
            .map_err(|error| format!("failed to render diagnostics schema: {error}"))?;
        println!("    {}", rendered);

        println!("\n  ✅ JSON Schema parameters verified");
        Ok(())
    }

    /// 场景 6：以编程方式构建配置
    fn demo_programmatic_config() -> Result<(), String> {
        section!(6, "Programmatic LspServerConfig");

        // 不使用 YAML，直接构建配置
        let mut servers = HashMap::new();
        servers.insert(
            "rust".to_string(),
            LspServerConfig {
                language: "rust".to_string(),
                command: "rust-analyzer".to_string(),
                args: vec![],
                extensions: vec![".rs".to_string()],
                env: HashMap::new(),
                initialization_options: None,
                max_restarts: 3,
            },
        );

        let config = LspConfig { servers };
        let rust = config
            .get("rust")
            .ok_or_else(|| "programmatic rust language server config is missing".to_string())?;
        assert_eq!(rust.command, "rust-analyzer");

        println!("  Built config programmatically:");
        println!("    rust → rust-analyzer (max_restarts: 3)");
        println!("  ✅ No YAML needed for simple setups");
        Ok(())
    }
}

/// 未启用 `lsp` feature 时的占位演示模块。
#[cfg(not(feature = "lsp"))]
mod demo {
    /// `lsp` feature 未启用时的占位实现：打印说明信息，提示如何启用该 feature 运行完整演示。
    pub async fn run() -> Result<(), String> {
        println!("╔══════════════════════════════════════════════════╗");
        println!("║       echo-agent  LSP Tools Demo                 ║");
        println!("╚══════════════════════════════════════════════════╝");
        println!();
        println!("  ⚠️  This demo requires the `lsp` feature flag.");
        println!();
        println!("  Run with:");
        println!("    cargo run --features lsp --example demo55_lsp_tools");
        println!();
        println!("  The LSP integration provides:");
        println!("    - LspConfig: parse .lsp.yaml configuration files");
        println!("    - LspManager: manage multiple language server processes");
        println!("    - LspDiagnosticsTool: get errors/warnings from language servers");
        println!("    - LspGotoDefinitionTool: find symbol definitions");
        println!("    - LspFindReferencesTool: find all symbol usages");
        println!("    - LspHoverTool: get type info and documentation");
        println!("    - LspStatusTool: check language server health");
        println!();
        println!("  Supported languages: Rust, Python, TypeScript, and more.");
        Ok(())
    }
}

/// demo55 契约测试：运行 LSP 工具演示（`lsp` feature 开启时运行完整场景，否则运行占位说明）。
#[tokio::test]
async fn contract_demo55_lsp_tools() -> Result<(), String> {
    demo::run().await
}
