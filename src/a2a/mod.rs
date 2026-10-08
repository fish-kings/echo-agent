/*!
A2A（Agent-to-Agent）协议支持

实现了 Google A2A 协议规范，支持：
- **Agent Card** 发布：描述 Agent 的能力、技能与端点
- **Agent 发现**：通过 `/.well-known/agent.json` 查找远程 Agent
- **任务交互**：跨 Agent 框架发送任务并查询状态
- **任务状态机**：`submitted → working → [input-required] → completed / failed`
- **流式事件**：通过 `tasks/sendSubscribe` 接收实时 SSE 事件流

# 状态机

```text
submitted → working → completed
                    → failed
                    → input-required ⇄ working

任意非终态 → canceled
```

# 核心类型

| 类型 | 说明 |
|------|-------------|
| [`AgentCard`] | Agent 能力描述卡片 |
| [`A2AServer`] | 服务端，暴露 Agent Card 与任务端点 |
| [`A2AClient`] | 客户端，发现并调用远程 Agent |
| [`TaskState`] | 任务生命周期状态枚举 |
| [`A2AStreamEvent`] | 流式事件（状态变化 / artifact 更新） |

# 示例

```rust,no_run
use echo_agent::a2a::{AgentCard, AgentSkill, A2AServer, TaskState};
use echo_agent::prelude::*;

# #[tokio::main]
# async fn main() -> echo_agent::error::Result<()> {
let card = AgentCard::builder("translator", "http://localhost:8080")
    .description("Multilingual translation agent")
    .version("1.0.0")
    .skill(AgentSkill::new("translate", "Translate text"))
    .streaming()
    .build();

let agent = ReactAgentBuilder::simple("qwen3-max", "Translation assistant")?;
let server = A2AServer::new(card, agent);
// server.handle_request(...) 用于同步请求
// server.handle_request_stream(...) 用于 SSE 流式请求
# Ok(())
# }
```
*/

/// A2A JWT 认证。
mod auth;
/// A2A 客户端。
mod client;
/// A2A HTTP 服务启动。
mod serve;
/// A2A 服务端请求处理。
mod server;
/// A2A 协议数据类型。
mod types;

pub use auth::{JwtClaims, JwtConfig, JwtConfigError, get_claims};
pub use client::A2AClient;
pub use serve::{serve, serve_with_auth};
pub use server::A2AServer;
pub use types::*;
