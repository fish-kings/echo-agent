/*!
 * Kubernetes Pod 沙箱执行器
 *
 * 通过 `kubectl` CLI 在 K8s 集群中创建临时 Pod 执行代码：
 * - Pod 级隔离（SecurityContext + ResourceQuota）
 * - 分离的所属任务（detached owner）显式结算临时 Pod（success / error / timeout / cancel / caller drop）
 * - 支持 Pod SecurityPolicy / SecurityStandards
 *
 * 适用于大规模并发和企业级部署。需要 `kubectl` 已配置集群访问。
 * 当前实现不能像 Docker 一样逐 Pod 强制断网；因此当
 * `ResourceLimits.network=false` 时会拒绝执行，避免违反调用方声明的隔离契约。
 */

use super::{
    CommandKind, ExecutionResult, IsolationLevel, ResourceLimits, SandboxCommand, SandboxExecutor,
    select_image_for_command,
};
use echo_core::error::Result;
use echo_core::error::SandboxError;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// kubectl 控制命令的默认超时
const DEFAULT_K8S_CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
/// 清理 Pod 时的轮询间隔
const K8S_CLEANUP_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// 终态错误信息的最大字符数
const MAX_K8S_TERMINAL_FACT_CHARS: usize = 1_024;
/// kubectl 启动失败时的最大重试次数
const K8S_CONTROL_START_RETRY_LIMIT: usize = 2;
/// kubectl 启动重试之间的延迟
const K8S_CONTROL_START_RETRY_DELAY: Duration = Duration::from_millis(5);
/// ETXTBSY（文本文件忙）的系统错误码
const ETXTBSY_OS_ERROR: i32 = 26;

/// K8s 沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sConfig {
    /// kubectl 使用的 namespace
    pub namespace: String,
    /// 默认镜像
    pub default_image: String,
    /// 语言与镜像映射
    pub language_images: std::collections::HashMap<String, String>,
    /// Pod 的 service account
    pub service_account: Option<String>,
    /// CPU request（如 "100m"）
    pub cpu_request: String,
    /// CPU limit（如 "500m"）
    pub cpu_limit: String,
    /// Memory request（如 "64Mi"）
    pub memory_request: String,
    /// Memory limit（如 "256Mi"）
    pub memory_limit: String,
    /**
     * 预留的集群侧清理时间（秒）
     *
     * 当前执行路径使用显式 `kubectl delete` 清理临时 Pod，
     * 该字段保留给未来可能的 Job/TTL 控制策略。
     */
    pub ttl_seconds: u32,
    /// 节点选择器
    pub node_selector: std::collections::HashMap<String, String>,
}

impl Default for K8sConfig {
    /// 默认配置：内置常见语言镜像，namespace 为 echo-sandbox
    fn default() -> Self {
        let mut language_images = std::collections::HashMap::new();
        language_images.insert("python".to_string(), "python:3.12-slim".to_string());
        language_images.insert("python3".to_string(), "python:3.12-slim".to_string());
        language_images.insert("node".to_string(), "node:20-slim".to_string());
        language_images.insert("javascript".to_string(), "node:20-slim".to_string());
        language_images.insert("ruby".to_string(), "ruby:3.3-slim".to_string());
        language_images.insert("go".to_string(), "golang:1.22-alpine".to_string());
        language_images.insert("rust".to_string(), "rust:1.77-slim".to_string());

        Self {
            namespace: "echo-sandbox".to_string(),
            default_image: "ubuntu:22.04".to_string(),
            language_images,
            service_account: None,
            cpu_request: "100m".to_string(),
            cpu_limit: "500m".to_string(),
            memory_request: "64Mi".to_string(),
            memory_limit: "256Mi".to_string(),
            ttl_seconds: 300,
            node_selector: std::collections::HashMap::new(),
        }
    }
}

/// Kubernetes Pod 沙箱
#[derive(Debug, Clone)]
pub struct K8sSandbox {
    /// K8s 沙箱配置
    config: K8sConfig,
    /// kubectl 可执行程序路径
    kubectl_program: PathBuf,
    /// kubectl 控制命令超时
    control_timeout: Duration,
}

/// 交给 Pod 生命周期所属任务的执行请求
struct K8sOwnerRequest {
    /// 待执行的命令
    command: SandboxCommand,
    /// 可选资源限制
    limits: Option<ResourceLimits>,
    /// 可选取消令牌
    cancel: Option<Arc<CancellationToken>>,
    /// 执行超时
    timeout: Duration,
    /// 临时 Pod 名称
    pod_name: String,
    /// `kubectl run` 参数
    run_args: Vec<String>,
    /// 调用方放弃等待时触发的令牌
    caller_abandoned: CancellationToken,
}

/// 调用方放弃守卫：未解除时在 drop 时触发放弃令牌
struct K8sCallerAbandonmentGuard {
    /// 调用方放弃令牌
    token: CancellationToken,
    /// 是否仍处于激活状态
    armed: bool,
}

impl K8sCallerAbandonmentGuard {
    /// 创建处于激活状态的守卫
    fn new(token: CancellationToken) -> Self {
        Self { token, armed: true }
    }

    /// 解除守卫，drop 时不再触发令牌
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for K8sCallerAbandonmentGuard {
    /// 若守卫仍激活，则取消放弃令牌
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

/// 执行被中断的原因
#[derive(Clone, Copy)]
enum K8sInterruption {
    /// 超时
    TimedOut,
    /// 被取消
    Cancelled,
}

/// 写入 stdin 的结果
enum K8sStdinOutcome {
    /// 写入完成（可能出错）
    Written(std::io::Result<()>),
    /// 写入期间超时
    TimedOut,
    /// 写入期间被取消
    Cancelled,
}

/// 等待 kubectl 进程退出的结果
enum K8sWaitOutcome {
    /// 进程已退出
    Completed(std::io::Result<std::process::ExitStatus>),
    /// 等待期间超时
    TimedOut,
    /// 等待期间被取消
    Cancelled,
}

/// 读取输出管道的结果
enum K8sPipeOutcome {
    /// 读取任务已结束
    Completed(std::result::Result<std::io::Result<Vec<u8>>, tokio::task::JoinError>),
    /// 读取期间超时
    TimedOut,
    /// 调用方已放弃
    CallerAbandoned,
}

/// 输出管道读取任务句柄
type K8sPipeReader = Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>;
/// stdout/stderr 读取结果
type K8sPipeOutput = std::result::Result<(Vec<u8>, Vec<u8>), String>;

impl K8sSandbox {
    /// 使用给定配置创建 K8s 沙箱
    pub fn new(config: K8sConfig) -> Self {
        Self {
            config,
            kubectl_program: PathBuf::from("kubectl"),
            control_timeout: DEFAULT_K8S_CONTROL_TIMEOUT,
        }
    }

    /// 测试用：指定 kubectl 程序路径创建沙箱
    #[cfg(test)]
    fn with_program(config: K8sConfig, kubectl_program: PathBuf) -> Self {
        Self {
            config,
            kubectl_program,
            /* 假客户端每次清理会启动多个真实子进程。
            在全特性测试套件并行运行时为调度器预留余量。 */
            control_timeout: Duration::from_secs(1),
        }
    }

    /// 检测 kubectl 是否可用
    async fn check_kubectl(&self) -> bool {
        self.run_kubectl_control(&["version".to_string(), "--client".to_string()], "version")
            .await
            .is_ok_and(|output| output.status.success())
    }

    /// 为命令选择合适的镜像
    fn select_image(&self, command: &SandboxCommand) -> String {
        select_image_for_command(
            command,
            &self.config.language_images,
            &self.config.default_image,
        )
    }

    /// 构建容器内执行的命令
    fn build_inner_command(command: &SandboxCommand) -> Vec<String> {
        match &command.kind {
            CommandKind::Shell(cmd) => vec!["sh".to_string(), "-c".to_string(), cmd.clone()],
            CommandKind::Program { program, args } => {
                let mut v = vec![program.clone()];
                v.extend(args.clone());
                v
            }
            CommandKind::Code { language, code } => {
                let (interpreter, flag) = match language.as_str() {
                    "python" | "python3" => ("python3", "-c"),
                    "node" | "javascript" | "js" => ("node", "-e"),
                    "ruby" => ("ruby", "-e"),
                    "perl" => ("perl", "-e"),
                    "php" => ("php", "-r"),
                    _ => ("sh", "-c"),
                };
                vec![interpreter.to_string(), flag.to_string(), code.clone()]
            }
        }
    }

    /// 生成 `kubectl delete` 的超时参数（控制超时的 3/4）
    fn deletion_timeout_arg(&self) -> String {
        let milliseconds = self
            .control_timeout
            .as_millis()
            .saturating_mul(3)
            .checked_div(4)
            .unwrap_or(1)
            .max(1);
        format!("--timeout={milliseconds}ms")
    }

    /// 构建删除 Pod 的 kubectl 参数
    fn pod_delete_args(&self, pod_name: &str) -> Vec<String> {
        vec![
            "delete".to_string(),
            "pod".to_string(),
            pod_name.to_string(),
            "-n".to_string(),
            self.config.namespace.clone(),
            "--grace-period=1".to_string(),
            "--ignore-not-found=true".to_string(),
            "--wait=true".to_string(),
            "--output=name".to_string(),
            self.deletion_timeout_arg(),
        ]
    }

    /// 构建查询 Pod 的 kubectl 参数
    fn pod_get_args(&self, pod_name: &str) -> Vec<String> {
        vec![
            "get".to_string(),
            "pod".to_string(),
            pod_name.to_string(),
            "-n".to_string(),
            self.config.namespace.clone(),
            "--ignore-not-found=true".to_string(),
            "--output=name".to_string(),
        ]
    }

    /// 计算控制命令的截止时间
    fn control_deadline(&self) -> tokio::time::Instant {
        tokio::time::sleep(self.control_timeout).deadline()
    }

    /// 在默认截止时间内运行 kubectl 控制命令
    async fn run_kubectl_control(
        &self,
        args: &[String],
        stage: &str,
    ) -> Result<std::process::Output> {
        self.run_kubectl_control_until(args, stage, self.control_deadline())
            .await
    }

    /// 在指定截止时间前运行 kubectl 控制命令，启动失败时有限重试
    async fn run_kubectl_control_until(
        &self,
        args: &[String],
        stage: &str,
        deadline: tokio::time::Instant,
    ) -> Result<std::process::Output> {
        let mut retries = 0;
        loop {
            let mut command = Command::new(&self.kubectl_program);
            command
                .args(args)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true);
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(k8s_control_deadline_error(stage, self.control_timeout));
            }
            match tokio::time::timeout(remaining, command.output()).await {
                Ok(Ok(output)) => return Ok(output),
                Ok(Err(error)) if is_retryable_kubectl_start_error(&error) => {
                    if retries >= K8S_CONTROL_START_RETRY_LIMIT {
                        return Err(echo_core::error::ReactError::Sandbox(Box::new(
                            SandboxError::IoError(format!(
                                "Failed to run kubectl {stage}: {error}"
                            )),
                        )));
                    }
                    retries += 1;
                    let delay = std::cmp::min(
                        K8S_CONTROL_START_RETRY_DELAY,
                        deadline.saturating_duration_since(tokio::time::Instant::now()),
                    );
                    if delay.is_zero() {
                        return Err(k8s_control_deadline_error(stage, self.control_timeout));
                    }
                    tokio::time::sleep(delay).await;
                }
                Ok(Err(error)) => {
                    return Err(echo_core::error::ReactError::Sandbox(Box::new(
                        SandboxError::IoError(format!("Failed to run kubectl {stage}: {error}")),
                    )));
                }
                Err(_) => return Err(k8s_control_deadline_error(stage, self.control_timeout)),
            }
        }
    }

    /// 删除 Pod 并确认其已不存在
    async fn delete_pod(&self, pod_name: &str) -> Result<()> {
        let deadline = self.control_deadline();
        loop {
            let output = self
                .run_kubectl_control_until(
                    &self.pod_delete_args(pod_name),
                    "pod deletion",
                    deadline,
                )
                .await?;
            if !output.status.success() {
                return Err(echo_core::error::ReactError::Sandbox(Box::new(
                    SandboxError::IoError(format!(
                        "kubectl failed to delete Pod {pod_name} ({}): {}",
                        format_exit_status(&output.status),
                        bounded_k8s_fact(String::from_utf8_lossy(&output.stderr).trim())
                    )),
                )));
            }

            let deletion_receipt = !String::from_utf8_lossy(&output.stdout).trim().is_empty();
            if self.pod_is_present_until(pod_name, deadline).await? {
                continue;
            }
            if deletion_receipt {
                return Ok(());
            }

            loop {
                self.wait_for_cleanup_observation(pod_name, deadline)
                    .await?;
                if self.pod_is_present_until(pod_name, deadline).await? {
                    break;
                }
            }
        }
    }

    /// 在截止时间前查询 Pod 是否仍存在
    async fn pod_is_present_until(
        &self,
        pod_name: &str,
        deadline: tokio::time::Instant,
    ) -> Result<bool> {
        let output = self
            .run_kubectl_control_until(
                &self.pod_get_args(pod_name),
                "pod absence confirmation",
                deadline,
            )
            .await?;
        if !output.status.success() {
            return Err(echo_core::error::ReactError::Sandbox(Box::new(
                SandboxError::IoError(format!(
                    "kubectl failed to confirm Pod {pod_name} absence ({}): {}",
                    format_exit_status(&output.status),
                    bounded_k8s_fact(String::from_utf8_lossy(&output.stderr).trim())
                )),
            )));
        }
        Ok(!String::from_utf8_lossy(&output.stdout).trim().is_empty())
    }

    /// 在两次清理观察之间等待，超出截止时间则报错
    async fn wait_for_cleanup_observation(
        &self,
        pod_name: &str,
        deadline: tokio::time::Instant,
    ) -> Result<()> {
        let now = tokio::time::Instant::now();
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            return Err(k8s_ambiguous_absence_error(pod_name, self.control_timeout));
        }
        tokio::time::sleep(K8S_CLEANUP_POLL_INTERVAL.min(remaining)).await;
        if tokio::time::Instant::now() >= deadline {
            return Err(k8s_ambiguous_absence_error(pod_name, self.control_timeout));
        }
        Ok(())
    }

    /// 在后台任务中清理 Pod，失败时记录清理欠账
    fn spawn_pod_cleanup(
        &self,
        pod_name: String,
        reason: &'static str,
    ) -> tokio::task::JoinHandle<Result<()>> {
        let owner = self.clone();
        tokio::spawn(async move {
            let cleanup = owner.delete_pod(&pod_name).await;
            if let Err(error) = cleanup.as_ref() {
                tracing::error!(
                    pod = %pod_name,
                    error = %error,
                    reason,
                    "K8s sandbox detached Pod cleanup debt"
                );
            }
            cleanup
        })
    }

    /// 生成 Pod JSON spec 并通过 kubectl run 执行
    async fn run_pod(
        &self,
        command: &SandboxCommand,
        limits: Option<&ResourceLimits>,
        cancel: Option<Arc<CancellationToken>>,
    ) -> Result<ExecutionResult> {
        if limits.is_some_and(|limits| !limits.network) {
            return Err(echo_core::error::ReactError::Sandbox(Box::new(
                SandboxError::PermissionDenied(
                    "K8sSandbox cannot enforce network=false for an individual pod".to_string(),
                ),
            )));
        }
        if cancel.as_ref().is_some_and(|token| token.is_cancelled()) {
            return Ok(empty_k8s_interrupted_result(
                K8sInterruption::Cancelled,
                Duration::ZERO,
            ));
        }

        let pod_name = format!("echo-sandbox-{}", uuid::Uuid::new_v4().simple());
        let image = self.select_image(command);
        let inner_cmd = Self::build_inner_command(command);
        let timeout = limits
            .and_then(|l| l.cpu_time_secs)
            .map(std::time::Duration::from_secs)
            .unwrap_or(command.timeout);

        // 根据 limits 动态设置资源
        let (cpu_req, cpu_lim, mem_req, mem_lim) = if let Some(l) = limits {
            let mr = l
                .memory_bytes
                .map(|b| format!("{b}"))
                .unwrap_or(self.config.memory_request.clone());
            let ml = l
                .memory_bytes
                .map(|b| b.saturating_mul(2).to_string())
                .unwrap_or(self.config.memory_limit.clone());
            (
                self.config.cpu_request.clone(),
                self.config.cpu_limit.clone(),
                mr,
                ml,
            )
        } else {
            (
                self.config.cpu_request.clone(),
                self.config.cpu_limit.clone(),
                self.config.memory_request.clone(),
                self.config.memory_limit.clone(),
            )
        };

        // 构建 kubectl run 命令
        let mut args = vec![
            "run".to_string(),
            pod_name.clone(),
            format!("--image={image}"),
            format!("--namespace={}", self.config.namespace),
            "--restart=Never".to_string(),
            "--attach".to_string(),
            format!(
                "--stdin={}",
                if command.stdin.is_some() {
                    "true"
                } else {
                    "false"
                }
            ),
            format!("--requests=cpu={cpu_req},memory={mem_req}"),
            format!("--limits=cpu={cpu_lim},memory={mem_lim}"),
        ];

        // 注入环境变量作为 Pod env（使用 --env）
        for (k, v) in &command.env {
            args.push(format!("--env={k}={v}"));
        }

        if let Some(ref sa) = self.config.service_account {
            args.push(format!("--serviceaccount={sa}"));
        }

        // SecurityContext 覆盖配置
        args.push("--overrides".to_string());
        args.push(
            serde_json::json!({
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "labels": {
                        "echo-sandbox": "true"
                    }
                },
                "spec": {
                    "securityContext": {
                        "runAsNonRoot": true,
                        "runAsUser": 65534,
                        "fsGroup": 65534
                    },
                    "containers": [{
                        "name": pod_name,
                        "securityContext": {
                            "allowPrivilegeEscalation": false,
                            "readOnlyRootFilesystem": false,
                            "capabilities": { "drop": ["ALL"] }
                        }
                    }],
                    "automountServiceAccountToken": false,
                    "enableServiceLinks": false
                }
            })
            .to_string(),
        );

        // 命令分隔
        args.push("--command".to_string());
        args.push("--".to_string());
        args.extend(inner_cmd);

        let caller_abandoned = CancellationToken::new();
        let mut caller_guard = K8sCallerAbandonmentGuard::new(caller_abandoned.clone());
        let owner = self.clone();
        let owned_command = command.clone();
        let owned_limits = limits.cloned();
        let recovery_name = pod_name.clone();
        let owner_task = tokio::spawn(async move {
            owner
                .run_pod_owner(K8sOwnerRequest {
                    command: owned_command,
                    limits: owned_limits,
                    cancel,
                    timeout,
                    pod_name,
                    run_args: args,
                    caller_abandoned,
                })
                .await
        });
        let joined = owner_task.await;
        match joined {
            Ok(terminal) => {
                caller_guard.disarm();
                terminal
            }
            Err(join_error) => {
                let primary =
                    echo_core::error::ReactError::Sandbox(Box::new(SandboxError::IoError(
                        format!("K8s sandbox lifecycle owner failed to join: {join_error}"),
                    )));
                let recovery = self
                    .spawn_pod_cleanup(recovery_name, "lifecycle owner join failure")
                    .await;
                caller_guard.disarm();
                match recovery {
                    Ok(Ok(())) => Err(primary),
                    Ok(Err(cleanup)) => Err(combined_k8s_cleanup_failure(
                        &primary.to_string(),
                        &cleanup.to_string(),
                    )),
                    Err(recovery_join_error) => Err(combined_k8s_cleanup_failure(
                        &primary.to_string(),
                        &format!("detached recovery cleanup failed to join: {recovery_join_error}"),
                    )),
                }
            }
        }
    }

    /// Pod 生命周期所属任务：执行 kubectl run 并在任何结局下结算 Pod
    async fn run_pod_owner(&self, request: K8sOwnerRequest) -> Result<ExecutionResult> {
        let K8sOwnerRequest {
            command,
            limits,
            cancel,
            timeout,
            pod_name,
            run_args,
            caller_abandoned,
        } = request;

        if caller_abandoned.is_cancelled() {
            return Ok(empty_k8s_interrupted_result(
                K8sInterruption::Cancelled,
                Duration::ZERO,
            ));
        }

        let mut cmd = Command::new(&self.kubectl_program);
        cmd.args(&run_args);
        cmd.stdin(if command.stdin.is_some() {
            std::process::Stdio::piped()
        } else {
            std::process::Stdio::null()
        });
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        cmd.kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);

        let start = Instant::now();

        let mut start_retries = 0;
        let mut child = loop {
            match cmd.spawn() {
                Ok(child) => break child,
                Err(error)
                    if is_retryable_kubectl_start_error(&error)
                        && start_retries < K8S_CONTROL_START_RETRY_LIMIT =>
                {
                    start_retries = start_retries.saturating_add(1);
                    tokio::time::sleep(K8S_CONTROL_START_RETRY_DELAY).await;
                }
                Err(error) => {
                    return Err(echo_core::error::ReactError::Sandbox(Box::new(
                        SandboxError::StartFailed(format!("Failed to run kubectl: {error}")),
                    )));
                }
            }
        };
        let process_group_id = child.id();
        let stdout = spawn_k8s_pipe_reader(child.stdout.take());
        let stderr = spawn_k8s_pipe_reader(child.stderr.take());
        let deadline = tokio::time::sleep(timeout);
        let execution_deadline = deadline.deadline();
        tokio::pin!(deadline);

        if let Some(input) = command.stdin.as_deref() {
            match child.stdin.take() {
                Some(mut stdin) => {
                    let write = stdin.write_all(input.as_bytes());
                    tokio::pin!(write);
                    match tokio::select! {
                        _ = wait_for_k8s_cancel(cancel.as_ref(), Some(&caller_abandoned)) => K8sStdinOutcome::Cancelled,
                        _ = &mut deadline => K8sStdinOutcome::TimedOut,
                        result = &mut write => K8sStdinOutcome::Written(result),
                    } {
                        K8sStdinOutcome::Written(Ok(())) => {}
                        K8sStdinOutcome::Written(Err(error)) => {
                            let terminal = k8s_client_failure(
                                &format!("Failed to write kubectl stdin: {error}"),
                                super::local::cleanup_child_process(&mut child, process_group_id)
                                    .await,
                                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                                    .await,
                            );
                            return self.finish_with_cleanup(&pod_name, terminal).await;
                        }
                        K8sStdinOutcome::TimedOut => {
                            let terminal = k8s_interrupted_terminal(
                                K8sInterruption::TimedOut,
                                start.elapsed(),
                                super::local::cleanup_child_process(&mut child, process_group_id)
                                    .await,
                                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                                    .await,
                                limits.as_ref(),
                            );
                            return self.finish_with_cleanup(&pod_name, terminal).await;
                        }
                        K8sStdinOutcome::Cancelled => {
                            let terminal = k8s_interrupted_terminal(
                                K8sInterruption::Cancelled,
                                start.elapsed(),
                                super::local::cleanup_child_process(&mut child, process_group_id)
                                    .await,
                                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                                    .await,
                                limits.as_ref(),
                            );
                            return self.finish_with_cleanup(&pod_name, terminal).await;
                        }
                    }
                }
                None => {
                    let terminal = k8s_client_failure(
                        "kubectl stdin pipe was not available",
                        super::local::cleanup_child_process(&mut child, process_group_id).await,
                        self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                            .await,
                    );
                    return self.finish_with_cleanup(&pod_name, terminal).await;
                }
            }
        }

        let outcome = tokio::select! {
            _ = wait_for_k8s_cancel(cancel.as_ref(), Some(&caller_abandoned)) => K8sWaitOutcome::Cancelled,
            _ = &mut deadline => K8sWaitOutcome::TimedOut,
            result = child.wait() => K8sWaitOutcome::Completed(result),
        };
        let terminal = match outcome {
            K8sWaitOutcome::Completed(Ok(status)) => {
                let client_cleanup =
                    super::local::cleanup_child_process(&mut child, process_group_id).await;
                let output =
                    collect_k8s_pipes_until(stdout, stderr, execution_deadline, &caller_abandoned)
                        .await;
                if client_cleanup.is_ok() {
                    k8s_completed_terminal(status, output, start.elapsed(), limits.as_ref())
                } else {
                    k8s_client_failure(
                        &format!(
                            "kubectl completed with {} but descendant cleanup failed",
                            format_exit_status(&status)
                        ),
                        client_cleanup,
                        output,
                    )
                }
            }
            K8sWaitOutcome::Completed(Err(error)) => k8s_client_failure(
                &format!("kubectl IO error: {error}"),
                super::local::cleanup_child_process(&mut child, process_group_id).await,
                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                    .await,
            ),
            K8sWaitOutcome::TimedOut => k8s_interrupted_terminal(
                K8sInterruption::TimedOut,
                start.elapsed(),
                super::local::cleanup_child_process(&mut child, process_group_id).await,
                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                    .await,
                limits.as_ref(),
            ),
            K8sWaitOutcome::Cancelled => k8s_interrupted_terminal(
                K8sInterruption::Cancelled,
                start.elapsed(),
                super::local::cleanup_child_process(&mut child, process_group_id).await,
                self.collect_cleanup_pipes(stdout, stderr, &caller_abandoned)
                    .await,
                limits.as_ref(),
            ),
        };
        self.finish_with_cleanup(&pod_name, terminal).await
    }

    /// 在控制超时内收集清理阶段的管道输出
    async fn collect_cleanup_pipes(
        &self,
        stdout: K8sPipeReader,
        stderr: K8sPipeReader,
        caller_abandoned: &CancellationToken,
    ) -> K8sPipeOutput {
        collect_k8s_pipes_until(
            stdout,
            stderr,
            tokio::time::Instant::now() + self.control_timeout,
            caller_abandoned,
        )
        .await
    }

    /// 删除 Pod 并将清理结果与主终态合并
    async fn finish_with_cleanup(
        &self,
        pod_name: &str,
        terminal: Result<ExecutionResult>,
    ) -> Result<ExecutionResult> {
        match (terminal, self.delete_pod(pod_name).await) {
            (terminal, Ok(())) => terminal,
            (Ok(primary), Err(cleanup)) => {
                let primary_fact = format!("K8s terminal [{}]", k8s_result_facts(&primary));
                log_k8s_cleanup_debt(pod_name, &primary_fact, &cleanup.to_string());
                Err(combined_k8s_cleanup_failure(
                    &primary_fact,
                    &cleanup.to_string(),
                ))
            }
            (Err(primary), Err(cleanup)) => {
                let primary_fact = format!(
                    "K8s execution failed: {}",
                    bounded_k8s_fact(&primary.to_string())
                );
                log_k8s_cleanup_debt(pod_name, &primary_fact, &cleanup.to_string());
                Err(combined_k8s_cleanup_failure(
                    &primary_fact,
                    &cleanup.to_string(),
                ))
            }
        }
    }

    /// 清理所有带有 echo-sandbox 标签的 Pod
    pub async fn cleanup_sandbox_pods(&self) -> Result<()> {
        let list_args = vec![
            "get".to_string(),
            "pods".to_string(),
            "-n".to_string(),
            self.config.namespace.clone(),
            "-l".to_string(),
            "echo-sandbox=true".to_string(),
            "-o".to_string(),
            "name".to_string(),
        ];
        let output = self
            .run_kubectl_control(&list_args, "sandbox Pod listing")
            .await?;
        if !output.status.success() {
            return Err(echo_core::error::ReactError::Sandbox(Box::new(
                SandboxError::IoError(format!(
                    "Failed to list sandbox Pods ({}): {}",
                    format_exit_status(&output.status),
                    bounded_k8s_fact(String::from_utf8_lossy(&output.stderr).trim())
                )),
            )));
        }

        let pods = String::from_utf8_lossy(&output.stdout);
        if !pods.trim().is_empty() {
            let delete_args = vec![
                "delete".to_string(),
                "pods".to_string(),
                "-n".to_string(),
                self.config.namespace.clone(),
                "-l".to_string(),
                "echo-sandbox=true".to_string(),
                "--grace-period=1".to_string(),
                "--ignore-not-found=true".to_string(),
                "--wait=true".to_string(),
                self.deletion_timeout_arg(),
            ];
            let deleted = self
                .run_kubectl_control(&delete_args, "sandbox Pod cleanup")
                .await?;
            if !deleted.status.success() {
                return Err(echo_core::error::ReactError::Sandbox(Box::new(
                    SandboxError::IoError(format!(
                        "Failed to clean sandbox Pods ({}): {}",
                        format_exit_status(&deleted.status),
                        bounded_k8s_fact(String::from_utf8_lossy(&deleted.stderr).trim())
                    )),
                )));
            }
        }
        Ok(())
    }
}

/// 在后台任务中读取管道全部内容
fn spawn_k8s_pipe_reader<R>(pipe: Option<R>) -> K8sPipeReader
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    pipe.map(|mut pipe| {
        tokio::spawn(async move {
            let mut bytes = Vec::new();
            pipe.read_to_end(&mut bytes).await?;
            Ok(bytes)
        })
    })
}

/// 在截止时间前收集 stdout 与 stderr
async fn collect_k8s_pipes_until(
    stdout: K8sPipeReader,
    stderr: K8sPipeReader,
    deadline: tokio::time::Instant,
    caller_abandoned: &CancellationToken,
) -> K8sPipeOutput {
    let stdout = collect_k8s_pipe_until("stdout", stdout, deadline, caller_abandoned).await;
    let stderr = collect_k8s_pipe_until("stderr", stderr, deadline, caller_abandoned).await;
    match (stdout, stderr) {
        (Ok(stdout), Ok(stderr)) => Ok((stdout, stderr)),
        (stdout, stderr) => {
            let failures = [stdout.err(), stderr.err()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Err(failures.join("; "))
        }
    }
}

/// 在截止时间前收集单个管道输出，超时或调用方放弃时中止读取
async fn collect_k8s_pipe_until(
    name: &str,
    reader: K8sPipeReader,
    deadline: tokio::time::Instant,
    caller_abandoned: &CancellationToken,
) -> std::result::Result<Vec<u8>, String> {
    let Some(mut reader) = reader else {
        return Ok(Vec::new());
    };
    let outcome = tokio::select! {
        result = &mut reader => K8sPipeOutcome::Completed(result),
        _ = tokio::time::sleep_until(deadline) => K8sPipeOutcome::TimedOut,
        _ = caller_abandoned.cancelled() => K8sPipeOutcome::CallerAbandoned,
    };
    match outcome {
        K8sPipeOutcome::Completed(result) => result
            .map_err(|error| format!("kubectl {name} reader failed to join: {error}"))?
            .map_err(|error| format!("failed to read kubectl {name}: {error}")),
        K8sPipeOutcome::TimedOut => {
            reader.abort();
            let _ = reader.await;
            Err(format!("kubectl {name} drain timed out before cleanup"))
        }
        K8sPipeOutcome::CallerAbandoned => {
            reader.abort();
            let _ = reader.await;
            Err(format!(
                "kubectl {name} drain was abandoned by the caller before cleanup"
            ))
        }
    }
}

/// 根据 kubectl 正常退出构造执行结果
fn k8s_completed_terminal(
    status: std::process::ExitStatus,
    output: K8sPipeOutput,
    duration: Duration,
    limits: Option<&ResourceLimits>,
) -> Result<ExecutionResult> {
    let (stdout, stderr) = output.map_err(|error| {
        echo_core::error::ReactError::Sandbox(Box::new(SandboxError::IoError(error)))
    })?;
    let mut result = ExecutionResult {
        exit_code: status.code().unwrap_or(-1),
        stdout_bytes: u64::try_from(stdout.len()).unwrap_or(u64::MAX),
        stderr_bytes: u64::try_from(stderr.len()).unwrap_or(u64::MAX),
        stdout: String::from_utf8_lossy(&stdout).to_string(),
        stderr: String::from_utf8_lossy(&stderr).to_string(),
        duration,
        sandbox_type: "k8s".to_string(),
        timed_out: false,
        cancelled: false,
        output_truncated: false,
    };
    if let Some(max_output_bytes) = limits.and_then(|value| value.max_output_bytes) {
        result.enforce_output_limit(max_output_bytes);
    }
    Ok(result)
}

/// 构造超时或取消后的终态结果
fn k8s_interrupted_terminal(
    interruption: K8sInterruption,
    duration: Duration,
    client_cleanup: std::result::Result<(), String>,
    output: K8sPipeOutput,
    limits: Option<&ResourceLimits>,
) -> Result<ExecutionResult> {
    if client_cleanup.is_err() || output.is_err() {
        return k8s_client_failure(
            match interruption {
                K8sInterruption::TimedOut => "K8s Pod execution timed out",
                K8sInterruption::Cancelled => "K8s Pod execution was cancelled",
            },
            client_cleanup,
            output,
        );
    }
    let (stdout, stderr) = output.unwrap_or_else(|_| (Vec::new(), Vec::new()));
    let mut result = ExecutionResult {
        exit_code: -1,
        stdout_bytes: u64::try_from(stdout.len()).unwrap_or(u64::MAX),
        stderr_bytes: u64::try_from(stderr.len()).unwrap_or(u64::MAX),
        stdout: String::from_utf8_lossy(&stdout).to_string(),
        stderr: String::from_utf8_lossy(&stderr).to_string(),
        duration,
        sandbox_type: "k8s".to_string(),
        timed_out: matches!(interruption, K8sInterruption::TimedOut),
        cancelled: matches!(interruption, K8sInterruption::Cancelled),
        output_truncated: false,
    };
    if let Some(max_output_bytes) = limits.and_then(|value| value.max_output_bytes) {
        result.enforce_output_limit(max_output_bytes);
    }
    Ok(result)
}

/// 构造 kubectl 客户端失败错误，附带清理与输出信息
fn k8s_client_failure(
    primary: &str,
    client_cleanup: std::result::Result<(), String>,
    output: K8sPipeOutput,
) -> Result<ExecutionResult> {
    let mut facts = vec![bounded_k8s_fact(primary)];
    if let Err(error) = client_cleanup {
        facts.push(format!(
            "kubectl client cleanup failed: {}",
            bounded_k8s_fact(&error)
        ));
    }
    match output {
        Ok((stdout, stderr)) => facts.push(format!(
            "stdout_bytes={}, stderr_bytes={}, stderr={}",
            stdout.len(),
            stderr.len(),
            bounded_k8s_fact(&String::from_utf8_lossy(&stderr))
        )),
        Err(error) => facts.push(format!(
            "kubectl output drain failed: {}",
            bounded_k8s_fact(&error)
        )),
    }
    Err(echo_core::error::ReactError::Sandbox(Box::new(
        SandboxError::IoError(facts.join("; ")),
    )))
}

/// 构造未产生输出的中断结果
fn empty_k8s_interrupted_result(
    interruption: K8sInterruption,
    duration: Duration,
) -> ExecutionResult {
    ExecutionResult {
        exit_code: -1,
        stdout: String::new(),
        stderr: match interruption {
            K8sInterruption::TimedOut => "K8s Pod execution timed out".to_string(),
            K8sInterruption::Cancelled => "K8s Pod execution cancelled by owning run".to_string(),
        },
        duration,
        sandbox_type: "k8s".to_string(),
        timed_out: matches!(interruption, K8sInterruption::TimedOut),
        cancelled: matches!(interruption, K8sInterruption::Cancelled),
        output_truncated: false,
        stdout_bytes: 0,
        stderr_bytes: 0,
    }
}

/// 等待外部取消或调用方放弃信号；二者均无时永久挂起
async fn wait_for_k8s_cancel(
    cancel: Option<&Arc<CancellationToken>>,
    caller_abandoned: Option<&CancellationToken>,
) {
    match (cancel, caller_abandoned) {
        (Some(cancel), Some(caller_abandoned)) => {
            tokio::select! {
                _ = cancel.cancelled() => {}
                _ = caller_abandoned.cancelled() => {}
            }
        }
        (Some(cancel), None) => cancel.cancelled().await,
        (None, Some(caller_abandoned)) => caller_abandoned.cancelled().await,
        (None, None) => std::future::pending().await,
    }
}

/// 构造控制命令超出截止时间的错误
fn k8s_control_deadline_error(stage: &str, timeout: Duration) -> echo_core::error::ReactError {
    echo_core::error::ReactError::Sandbox(Box::new(SandboxError::IoError(format!(
        "kubectl {stage} did not settle within the shared {}ms control deadline",
        timeout.as_millis()
    ))))
}

/// 判断 kubectl 启动错误是否可重试（仅 ETXTBSY）
fn is_retryable_kubectl_start_error(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(ETXTBSY_OS_ERROR)
}

/// 构造无法确认 Pod 已消失的错误
fn k8s_ambiguous_absence_error(pod_name: &str, timeout: Duration) -> echo_core::error::ReactError {
    echo_core::error::ReactError::Sandbox(Box::new(SandboxError::IoError(format!(
        "K8s Pod cleanup could not confirm {pod_name} absence within {}ms after an ambiguous delete; the create request may still commit",
        timeout.as_millis()
    ))))
}

/// 合并主错误与 Pod 清理失败信息
fn combined_k8s_cleanup_failure(primary: &str, cleanup: &str) -> echo_core::error::ReactError {
    echo_core::error::ReactError::Sandbox(Box::new(SandboxError::IoError(format!(
        "{}; Pod cleanup also failed: {}",
        bounded_k8s_fact(primary),
        bounded_k8s_fact(cleanup)
    ))))
}

/// 记录 Pod 清理欠账日志
fn log_k8s_cleanup_debt(pod_name: &str, primary: &str, cleanup: &str) {
    tracing::error!(
        pod = pod_name,
        primary = %bounded_k8s_fact(primary),
        cleanup = %bounded_k8s_fact(cleanup),
        "K8s sandbox Pod cleanup debt"
    );
}

/// 将执行结果摘要为可读的事实字符串
fn k8s_result_facts(result: &ExecutionResult) -> String {
    format!(
        "exit_code={}, timed_out={}, cancelled={}, stdout_bytes={}, stderr_bytes={}, output_truncated={}",
        result.exit_code,
        result.timed_out,
        result.cancelled,
        result.stdout_bytes,
        result.stderr_bytes,
        result.output_truncated
    )
}

/// 将字符串截断到最大长度
fn bounded_k8s_fact(value: &str) -> String {
    let mut fact = value
        .chars()
        .take(MAX_K8S_TERMINAL_FACT_CHARS)
        .collect::<String>();
    if value.chars().count() > MAX_K8S_TERMINAL_FACT_CHARS {
        fact.push_str("...");
    }
    fact
}

/// 格式化进程退出状态
fn format_exit_status(status: &std::process::ExitStatus) -> String {
    status
        .code()
        .map(|code| format!("exit code {code}"))
        .unwrap_or_else(|| "terminated by signal".to_string())
}

impl SandboxExecutor for K8sSandbox {
    /// 执行器名称
    fn name(&self) -> &str {
        "k8s"
    }

    /// 返回隔离级别（编排级）
    fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::Orchestrated
    }

    /// 检查 kubectl 是否可用
    fn is_available(&self) -> BoxFuture<'_, bool> {
        Box::pin(self.check_kubectl())
    }

    /// 清理所有沙箱 Pod
    fn cleanup(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(self.cleanup_sandbox_pods())
    }

    /// 在临时 Pod 中执行命令
    fn execute(&self, command: SandboxCommand) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move { self.run_pod(&command, None, None).await })
    }

    /// 按资源限制在临时 Pod 中执行命令
    fn execute_with_limits(
        &self,
        command: SandboxCommand,
        limits: ResourceLimits,
    ) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move { self.run_pod(&command, Some(&limits), None).await })
    }

    /// 按资源限制执行命令，并支持外部取消
    fn execute_with_limits_and_cancel(
        &self,
        command: SandboxCommand,
        limits: ResourceLimits,
        cancel: Option<Arc<CancellationToken>>,
    ) -> BoxFuture<'_, Result<ExecutionResult>> {
        Box::pin(async move { self.run_pod(&command, Some(&limits), cancel).await })
    }
}

// ── 单元测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证仅 ETXTBSY 启动错误可重试
    #[test]
    fn only_etxtbsy_is_retryable_when_starting_kubectl() {
        assert!(is_retryable_kubectl_start_error(
            &std::io::Error::from_raw_os_error(ETXTBSY_OS_ERROR)
        ));
        assert!(!is_retryable_kubectl_start_error(
            &std::io::Error::from_raw_os_error(2)
        ));
    }

    /// 测试用的假 kubectl 脚本
    #[cfg(unix)]
    struct FakeKubectl {
        /// 临时目录
        directory: PathBuf,
        /// 假 kubectl 可执行文件路径
        program: PathBuf,
        /// 操作日志文件路径
        log: PathBuf,
    }

    #[cfg(unix)]
    impl FakeKubectl {
        /// 按指定模式生成假 kubectl 脚本
        fn new(mode: &str) -> std::result::Result<Self, Box<dyn std::error::Error>> {
            use std::os::unix::fs::PermissionsExt;

            let directory = std::env::temp_dir().join(format!(
                "echo-fake-kubectl-{}",
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&directory)?;
            let program = directory.join("kubectl");
            let log = directory.join("kubectl.log");
            let script = format!(
                r#"#!/bin/sh
LOG="$0.log"
printf '%s\n' "$1" >> "$LOG"
case "$1" in
  version) exit 0 ;;
  run)
    case "{mode}" in
      delayed-visible|never-visible)
        touch "$0.submitted"
        printf 'completed\n'
        exit 0
        ;;
      success|success-cleanup-fail|delete-timeout) printf 'completed\n'; exit 0 ;;
      delete-spawn-fail) rm "$0"; printf 'completed\n'; exit 0 ;;
      command-fail|command-cleanup-fail) printf 'command failed\n' >&2; exit 17 ;;
      stdin-fail) exit 19 ;;
      leader-pipe)
        (sleep 10) &
        printf 'leader-exit\n' >> "$LOG"
        exit 0
        ;;
      timeout|timeout-cleanup-fail|cancel|cancel-cleanup-fail|abort|abort-cleanup-fail|blocked-stdin) exec sleep 10 ;;
    esac
    ;;
  delete)
    case "{mode}" in
      delayed-visible)
        if [ ! -f "$0.first-delete" ]; then
          touch "$0.first-delete"
          exit 0
        fi
        if [ ! -f "$0.visible" ]; then
          printf 'delete retried before delayed Pod became visible\n' >&2
          exit 25
        fi
        rm -f "$0.visible"
        touch "$0.deleted"
        printf 'pod/echo-sandbox-delayed\n'
        exit 0
        ;;
      never-visible) exit 0 ;;
      abort-cleanup-fail)
        sleep 0.05
        printf 'delete-failed\n' >> "$LOG"
        printf 'forced cleanup failure\n' >&2
        exit 23
        ;;
      *-cleanup-fail) printf 'forced cleanup failure\n' >&2; exit 23 ;;
      delete-timeout) exec sleep 10 ;;
      abort|blocked-stdin|leader-pipe|join-recovery)
        sleep 0.05
        printf 'delete-complete\n' >> "$LOG"
        printf 'pod/echo-sandbox-test\n'
        exit 0
        ;;
      *) printf 'pod/echo-sandbox-test\n'; exit 0 ;;
    esac
    ;;
  get)
    if [ "{mode}" = "delayed-visible" ] && [ ! -f "$0.deleted" ]; then
      if [ ! -f "$0.first-get" ]; then
        touch "$0.first-get"
        exit 0
      fi
      touch "$0.visible"
      printf 'pod/echo-sandbox-delayed\n'
    fi
    exit 0
    ;;
esac
exit 64
"#
            );
            let script_path = directory.join("kubectl.script");
            std::fs::write(&script_path, script)?;
            let mut permissions = std::fs::metadata(&script_path)?.permissions();
            permissions.set_mode(0o700);
            std::fs::set_permissions(&script_path, permissions)?;
            std::fs::rename(&script_path, &program)?;
            Ok(Self {
                directory,
                program,
                log,
            })
        }

        /// 创建使用该假 kubectl 的沙箱
        fn sandbox(&self) -> K8sSandbox {
            K8sSandbox::with_program(K8sConfig::default(), self.program.clone())
        }

        /// 读取已记录的 kubectl 操作序列
        fn operations(&self) -> std::result::Result<Vec<String>, std::io::Error> {
            std::fs::read_to_string(&self.log).map(|contents| {
                contents
                    .lines()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
        }

        /// 查询延迟可见的 Pod 是否存在
        async fn probe_pod(&self) -> std::result::Result<String, std::io::Error> {
            let output = Command::new(&self.program)
                .args(["get", "pod", "echo-sandbox-delayed", "-o", "name"])
                .output()
                .await?;
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        }
    }

    #[cfg(unix)]
    impl Drop for FakeKubectl {
        /// 删除临时目录
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    /// 等待日志中出现指定操作
    #[cfg(unix)]
    async fn wait_for_operation(log: &std::path::Path, operation: &str) -> Result<()> {
        for _ in 0..100 {
            if let Ok(contents) = tokio::fs::read_to_string(log).await
                && contents.lines().any(|line| line == operation)
            {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        Err(echo_core::error::ReactError::Sandbox(Box::new(
            SandboxError::IoError(format!("fake kubectl never observed operation {operation}")),
        )))
    }

    /// 验证 K8s 沙箱不支持实时流式输出（使用缓冲回退）
    #[test]
    fn k8s_streaming_is_explicitly_buffered_fallback() {
        let sandbox = K8sSandbox::new(K8sConfig::default());
        assert!(!sandbox.supports_streaming());
    }

    /// 验证默认配置
    #[test]
    fn test_k8s_config_default() {
        let config = K8sConfig::default();
        assert_eq!(config.namespace, "echo-sandbox");
        assert_eq!(config.cpu_limit, "500m");
        assert_eq!(config.memory_limit, "256Mi");
    }

    /// 验证 Pod 清理参数会等待 API 优雅删除
    #[test]
    fn pod_cleanup_waits_for_graceful_api_removal() {
        let sandbox = K8sSandbox::new(K8sConfig::default());
        let args = sandbox.pod_delete_args("echo-sandbox-test");
        assert!(args.iter().any(|arg| arg == "--grace-period=1"));
        assert!(args.iter().any(|arg| arg == "--wait=true"));
        assert!(args.iter().any(|arg| arg == "--output=name"));
        assert!(args.iter().any(|arg| arg.starts_with("--timeout=")));
        assert!(!args.iter().any(|arg| arg == "--force"));
    }

    /// 验证延迟提交的 Pod 在清理返回前被删除
    #[cfg(unix)]
    #[tokio::test]
    async fn delayed_api_commit_is_deleted_before_cleanup_returns()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let fake = FakeKubectl::new("delayed-visible")?;
        let result = fake
            .sandbox()
            .execute(SandboxCommand::shell("complete after API submission"))
            .await?;
        assert!(result.success());
        assert_eq!(
            fake.operations()?,
            ["run", "delete", "get", "get", "delete", "get"]
        );

        assert!(fake.probe_pod().await?.is_empty());
        let delayed = fake.probe_pod().await?;
        assert!(
            delayed.is_empty(),
            "delayed Pod escaped cleanup after an ambiguous NotFound: {delayed}"
        );
        Ok(())
    }

    /// 验证无法确认 Pod 消失时报告为清理欠账
    #[cfg(unix)]
    #[tokio::test]
    async fn ambiguous_absence_exhaustion_is_typed_cleanup_debt()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let fake = FakeKubectl::new("never-visible")?;
        let error = fake
            .sandbox()
            .execute(SandboxCommand::shell("create outcome stays ambiguous"))
            .await
            .err()
            .ok_or("ambiguous Pod absence was reported as successful cleanup")?;
        let message = error.to_string();
        assert!(message.contains("exit_code=0"));
        assert!(message.contains("could not confirm"));
        assert!(message.contains("create request may still commit"));
        Ok(())
    }

    /// 验证成功与非零退出时均在返回前删除 Pod
    #[cfg(unix)]
    #[tokio::test]
    async fn success_and_nonzero_completion_remove_pod_before_return()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (mode, expected_exit, expected_success) in
            [("success", 0, true), ("command-fail", 17, false)]
        {
            let fake = FakeKubectl::new(mode)?;
            let result = fake
                .sandbox()
                .execute(SandboxCommand::shell("run command"))
                .await?;
            assert_eq!(result.exit_code, expected_exit);
            assert_eq!(result.success(), expected_success);
            assert_eq!(fake.operations()?, ["run", "delete", "get"]);
        }
        Ok(())
    }

    /// 验证超时与取消时均在返回前删除 Pod
    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_and_cancellation_remove_pod_before_return()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let timed_out = FakeKubectl::new("timeout")?;
        let timeout_result = timed_out
            .sandbox()
            .execute(
                SandboxCommand::shell("sleep forever")
                    .with_timeout(std::time::Duration::from_millis(30)),
            )
            .await?;
        assert!(timeout_result.timed_out);
        assert_eq!(timed_out.operations()?, ["run", "delete", "get"]);

        let cancelled = FakeKubectl::new("cancel")?;
        let cancel = Arc::new(CancellationToken::new());
        let cancellation = cancel.clone();
        let sandbox = cancelled.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute_with_limits_and_cancel(
                    SandboxCommand::shell("sleep forever"),
                    ResourceLimits {
                        cpu_time_secs: Some(30),
                        network: true,
                        ..ResourceLimits::default()
                    },
                    Some(cancellation),
                )
                .await
        });
        wait_for_operation(&cancelled.log, "run").await?;
        cancel.cancel();
        let cancel_result = execution.await??;
        assert!(cancel_result.cancelled);
        assert_eq!(cancelled.operations()?, ["run", "delete", "get"]);
        Ok(())
    }

    /// 验证清理失败时保留主终态信息
    #[cfg(unix)]
    #[tokio::test]
    async fn cleanup_failure_preserves_primary_terminal_facts()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (mode, command, fact) in [
            (
                "success-cleanup-fail",
                SandboxCommand::shell("complete"),
                "exit_code=0",
            ),
            (
                "command-cleanup-fail",
                SandboxCommand::shell("fail"),
                "exit_code=17",
            ),
            (
                "timeout-cleanup-fail",
                SandboxCommand::shell("timeout").with_timeout(std::time::Duration::from_millis(30)),
                "timed_out=true",
            ),
        ] {
            let fake = FakeKubectl::new(mode)?;
            let error = fake
                .sandbox()
                .execute(command)
                .await
                .err()
                .ok_or("cleanup failure unexpectedly returned a terminal result")?;
            let message = error.to_string();
            assert!(message.contains(fact), "missing {fact} in {message}");
            assert!(message.contains("forced cleanup failure"));
            assert_eq!(fake.operations()?, ["run", "delete"]);
        }

        let cancelled = FakeKubectl::new("cancel-cleanup-fail")?;
        let cancel = Arc::new(CancellationToken::new());
        let cancellation = cancel.clone();
        let sandbox = cancelled.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute_with_limits_and_cancel(
                    SandboxCommand::shell("cancel"),
                    ResourceLimits {
                        cpu_time_secs: Some(30),
                        network: true,
                        ..ResourceLimits::default()
                    },
                    Some(cancellation),
                )
                .await
        });
        wait_for_operation(&cancelled.log, "run").await?;
        cancel.cancel();
        let error = execution
            .await?
            .err()
            .ok_or("cancel cleanup failure unexpectedly returned a terminal result")?;
        let message = error.to_string();
        assert!(message.contains("cancelled=true"));
        assert!(message.contains("forced cleanup failure"));
        assert_eq!(cancelled.operations()?, ["run", "delete"]);
        Ok(())
    }

    /// 验证清理命令启动失败与超时对调用方可见
    #[cfg(unix)]
    #[tokio::test]
    async fn cleanup_spawn_and_timeout_failures_are_visible()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        for (mode, fact) in [
            ("delete-spawn-fail", "Failed to run kubectl pod deletion"),
            (
                "delete-timeout",
                "kubectl pod deletion did not settle within the shared",
            ),
        ] {
            let fake = FakeKubectl::new(mode)?;
            let error = fake
                .sandbox()
                .execute(SandboxCommand::shell("complete"))
                .await
                .err()
                .ok_or("cleanup control failure unexpectedly returned success")?;
            let message = error.to_string();
            assert!(message.contains("exit_code=0"));
            assert!(message.contains(fact), "missing {fact} in {message}");
        }
        Ok(())
    }

    /// 验证 Pod 提交后调用方中止时清理所属任务仍会运行
    #[cfg(unix)]
    #[tokio::test]
    async fn caller_abort_after_pod_submission_keeps_cleanup_owner()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let fake = FakeKubectl::new("abort")?;
        let sandbox = fake.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute(
                    SandboxCommand::shell("sleep forever")
                        .with_timeout(std::time::Duration::from_secs(30)),
                )
                .await
        });
        wait_for_operation(&fake.log, "run").await?;
        execution.abort();
        let join = execution.await;
        assert!(matches!(join, Err(error) if error.is_cancelled()));
        wait_for_operation(&fake.log, "delete-complete").await?;
        Ok(())
    }

    /// 验证主进程退出后调用方中止仍会清理管道持有者与 Pod
    #[cfg(unix)]
    #[tokio::test]
    async fn caller_abort_after_leader_exit_cleans_pipe_holder_and_pod()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let fake = FakeKubectl::new("leader-pipe")?;
        let sandbox = fake.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute(SandboxCommand::shell("helper inherits output"))
                .await
        });
        wait_for_operation(&fake.log, "leader-exit").await?;
        execution.abort();
        let join = execution.await;
        assert!(matches!(join, Err(error) if error.is_cancelled()));
        wait_for_operation(&fake.log, "delete-complete").await?;
        Ok(())
    }

    /// 验证 stdin 写入失败与阻塞时调用方丢弃均能完成清理
    #[cfg(unix)]
    #[tokio::test]
    async fn stdin_failure_and_blocked_stdin_caller_drop_reach_settled_cleanup()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let failed = FakeKubectl::new("stdin-fail")?;
        let error = failed
            .sandbox()
            .execute(SandboxCommand::shell("close stdin").with_stdin("x".repeat(8 * 1024 * 1024)))
            .await
            .err()
            .ok_or("closed kubectl stdin unexpectedly accepted the entire payload")?;
        assert!(error.to_string().contains("Failed to write kubectl stdin"));
        assert_eq!(failed.operations()?, ["run", "delete", "get"]);

        let blocked = FakeKubectl::new("blocked-stdin")?;
        let sandbox = blocked.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute(
                    SandboxCommand::shell("never reads stdin")
                        .with_stdin("x".repeat(8 * 1024 * 1024)),
                )
                .await
        });
        wait_for_operation(&blocked.log, "run").await?;
        execution.abort();
        let join = execution.await;
        assert!(matches!(join, Err(error) if error.is_cancelled()));
        wait_for_operation(&blocked.log, "delete-complete").await?;
        Ok(())
    }

    /// 验证调用方丢弃不会隐藏清理失败或中断恢复清理
    #[cfg(unix)]
    #[tokio::test]
    async fn caller_drop_cannot_hide_cleanup_failure_or_interrupt_join_recovery()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let failed = FakeKubectl::new("abort-cleanup-fail")?;
        let sandbox = failed.sandbox();
        let execution = tokio::spawn(async move {
            sandbox
                .execute(SandboxCommand::shell("sleep forever"))
                .await
        });
        wait_for_operation(&failed.log, "run").await?;
        execution.abort();
        let join = execution.await;
        assert!(matches!(join, Err(error) if error.is_cancelled()));
        wait_for_operation(&failed.log, "delete-failed").await?;

        let recovered = FakeKubectl::new("join-recovery")?;
        let cleanup = recovered
            .sandbox()
            .spawn_pod_cleanup("echo-sandbox-recovery".to_string(), "test join recovery");
        let waiter = tokio::spawn(cleanup);
        wait_for_operation(&recovered.log, "delete").await?;
        waiter.abort();
        let join = waiter.await;
        assert!(matches!(join, Err(error) if error.is_cancelled()));
        wait_for_operation(&recovered.log, "delete-complete").await?;
        Ok(())
    }

    /// 验证默认镜像选择
    #[test]
    fn test_select_image_default() {
        let sandbox = K8sSandbox::new(K8sConfig::default());
        let cmd = SandboxCommand::shell("echo test");
        assert_eq!(sandbox.select_image(&cmd), "ubuntu:22.04");
    }

    /// 验证代码模式的容器内命令
    #[test]
    fn test_inner_command_code() {
        let cmd = SandboxCommand::code("python", "print('hello')");
        let inner = K8sSandbox::build_inner_command(&cmd);
        assert_eq!(inner, vec!["python3", "-c", "print('hello')"]);
    }

    /// 验证程序模式的容器内命令
    #[test]
    fn test_inner_command_program() {
        let cmd = SandboxCommand::program("ls", vec!["-la".to_string()]);
        let inner = K8sSandbox::build_inner_command(&cmd);
        assert_eq!(inner, vec!["ls", "-la"]);
    }

    /// 验证 PHP 代码的容器内命令
    #[test]
    fn test_inner_command_php() {
        let cmd = SandboxCommand::code("php", "echo 'hi';");
        let inner = K8sSandbox::build_inner_command(&cmd);
        assert_eq!(inner, vec!["php", "-r", "echo 'hi';"]);
    }

    /// 验证 network=false 时在调用 kubectl 前即拒绝执行
    #[tokio::test]
    async fn network_denial_fails_closed_before_kubectl() {
        let sandbox = K8sSandbox::new(K8sConfig::default());
        let limits = ResourceLimits {
            network: false,
            ..ResourceLimits::default()
        };
        let error = sandbox
            .execute_with_limits(SandboxCommand::shell("echo test"), limits)
            .await
            .err();
        assert!(matches!(
            error,
            Some(echo_core::error::ReactError::Sandbox(inner))
                if matches!(*inner, SandboxError::PermissionDenied(_))
        ));
    }
}
