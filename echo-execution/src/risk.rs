//! 工具风险分类 —— 按类别评估风险，用于生成安全提示。

use echo_core::tools::ToolParameters;

/// 工具风险类别，依据工具类型和参数判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolRiskCategory {
    /// 只读操作 —— 无风险。
    ReadOnly,
    /// 文件写入操作 —— 风险：数据丢失或损坏。
    FileWrite,
    /// Shell/命令执行 —— 风险：任意代码执行。
    ShellExec,
    /// Git 写入操作 —— 风险：仓库损坏。
    GitWrite,
    /// 数据库写入操作 —— 风险：数据损坏。
    DatabaseWrite,
    /// 网络调用 —— 风险：数据外泄。
    NetworkCall,
    /// 破坏性操作 —— 风险：不可逆的数据丢失。
    Destructive,
}

impl ToolRiskCategory {
    /// 返回可读的风险描述文本。
    pub fn description(&self) -> &str {
        match self {
            Self::ReadOnly => "Read-only — no risk",
            Self::FileWrite => "File will be modified or overwritten",
            Self::ShellExec => "Arbitrary command execution",
            Self::GitWrite => "Repository state will be changed",
            Self::DatabaseWrite => "Database will be modified",
            Self::NetworkCall => "Data may be sent over the network",
            Self::Destructive => "Irreversible operation — data loss possible",
        }
    }

    /// 风险等级：0 = 无，1 = 低，2 = 中，3 = 高。
    pub fn level(&self) -> u8 {
        match self {
            Self::ReadOnly => 0,
            Self::NetworkCall => 1,
            Self::FileWrite => 2,
            Self::GitWrite => 2,
            Self::DatabaseWrite => 2,
            Self::ShellExec => 3,
            Self::Destructive => 3,
        }
    }

    /// 用于展示的权限标签。
    pub fn permission_label(&self) -> &str {
        match self {
            Self::ReadOnly => "Read",
            Self::FileWrite => "Edit",
            Self::ShellExec => "Execute",
            Self::GitWrite => "Git",
            Self::DatabaseWrite => "Database",
            Self::NetworkCall => "Network",
            Self::Destructive => "Delete",
        }
    }
}

/// 根据工具名称和参数对工具进行风险分类。
pub struct ToolRiskClassifier;

impl ToolRiskClassifier {
    /// 根据工具名称判定其风险类别。
    pub fn classify(tool_name: &str) -> ToolRiskCategory {
        match tool_name {
            // 只读
            "read_file" | "search" | "grep" | "list_files" | "git_log" | "git_status"
            | "git_diff" | "git_blame" | "git_branch" => ToolRiskCategory::ReadOnly,
            // 文件写入
            "apply_patch" | "write_file" | "append_file" | "create_file" | "update_file"
            | "move_file" => ToolRiskCategory::FileWrite,
            // Shell
            "shell" | "execute" => ToolRiskCategory::ShellExec,
            // Git 写入
            "git_commit" | "git_add" | "git_push" | "git_tag" => ToolRiskCategory::GitWrite,
            // 数据库
            "db_query" | "db_execute" | "sql" => ToolRiskCategory::DatabaseWrite,
            // 网络
            "web_fetch" | "web_search" | "http_request" | "api_call" => {
                ToolRiskCategory::NetworkCall
            }
            // 破坏性操作
            "delete_file" | "rm" | "drop_table" | "truncate" => ToolRiskCategory::Destructive,
            // 默认：未知工具按只读处理
            _ => ToolRiskCategory::ReadOnly,
        }
    }

    /// 为一次工具调用生成可读的安全提示文本。
    pub fn safety_notice(tool_name: &str, params: &ToolParameters) -> String {
        let category = Self::classify(tool_name);
        let path = params
            .get("path")
            .or_else(|| params.get("file_path"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let command = params.get("command").and_then(|v| v.as_str()).unwrap_or("");

        match category {
            ToolRiskCategory::ReadOnly => String::new(),
            ToolRiskCategory::FileWrite => {
                format!("Editing {path} — risk: file will be modified")
            }
            ToolRiskCategory::ShellExec => {
                format!("Running: {command} — risk: arbitrary command execution")
            }
            ToolRiskCategory::GitWrite => {
                "Git write to repository — risk: repository state change".to_string()
            }
            ToolRiskCategory::DatabaseWrite => {
                "Database modification — risk: data corruption".to_string()
            }
            ToolRiskCategory::NetworkCall => {
                "Network request — risk: data may leave this machine".to_string()
            }
            ToolRiskCategory::Destructive => {
                format!("DELETING {path} — risk: irreversible data loss")
            }
        }
    }
}
