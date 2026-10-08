/*!
多层级代码执行沙箱。

提供三种隔离级别，统一封装在 [`SandboxExecutor`] trait 之下。
[`SandboxManager`] 会自动选择当前可用的最佳层级。

# 隔离级别

| 级别 | 后端 | 使用场景 |
|-------|---------|----------|
| [`LocalSandbox`] | 直接进程启动 | 开发环境、可信代码 |
| [`DockerSandbox`] | Docker 容器 | CI/CD、不可信代码 |
| [`K8sSandbox`] | Kubernetes Pod | 生产环境、多租户 |

# 快速开始

```rust,ignore
use echo_agent::prelude::*;

# fn main() -> echo_agent::error::Result<()> {
let sandbox = LocalSandbox::new();
let result = sandbox.execute(SandboxCommand::shell("echo hello"))?;
println!("{}", result.stdout);
# Ok(())
# }
```
*/

pub use echo_execution::sandbox::*;
