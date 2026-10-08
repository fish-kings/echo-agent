use super::prompt::{MappedPrompt, map_prompt};
use super::runtime::{
    AcpConnectionProfile, AcpConnectionServices, ConnectionMode, RunObserverContext, RunStartSpec,
    StandardBridgeOutcome,
};
use super::session::{
    AcpSession, AcpSessionFactory, ActiveTurnLease, SessionRegistry, validate_session_paths,
};
use crate::agent::EventIdentity;
use crate::runtime::{TurnDeliveryOutcome, TurnMode, TurnOutcome, TurnRequest};
use agent_client_protocol::schema::{ProtocolVersion, v1};
use agent_client_protocol::{Agent as AcpRole, Client, ConnectTo, ConnectionTo, Error, Responder};
use futures::future::BoxFuture;
use std::future::Future;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const DEFAULT_MAX_SESSIONS: usize = 128;
const DEFAULT_MAX_PROMPT_CHARS: usize = 1_000_000;
const DEFAULT_MAX_UPDATE_CHARS: usize = 1_000_000;
const DEFAULT_MAX_UPDATES_PER_TURN: usize = 10_000;
const DEFAULT_MAX_TOTAL_UPDATE_CHARS: usize = 8_000_000;
const DEFAULT_MAX_EXTENSION_CONCURRENCY: usize = 8;
const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_ERROR_CHARS: usize = 512;

trait AcpCloseContext: Send + Sync {
    fn close(&self) -> BoxFuture<'_, agent_client_protocol::Result<()>>;
}

/** 一个 ACP 连接的框架 Session 与清理债务的持有者。

在把 adapter 交给官方 `ConnectTo` API 之前，先获取并保留此句柄。连接建立会
拒绝没有保留句柄的调用方，因为协议 trait 会消费掉 adapter，无法在关闭出错时
把资源持有者再返还给调用方。手动调用方必须一直持有该句柄直到关闭成功；
在被接纳后丢弃它会放弃重试的所有权。[`AcpAgentAdapter::connect_retaining_close_owner`]
会为直接连接的调用方持有并返回该句柄。
*/
#[derive(Clone, Default)]
pub struct AcpAdapterCloseHandle {
    context: Arc<StdMutex<Option<Arc<dyn AcpCloseContext>>>>,
}

impl AcpAdapterCloseHandle {
    fn is_retained(&self) -> bool {
        Arc::strong_count(&self.context) > 1
    }

    fn bind(&self, context: Arc<dyn AcpCloseContext>) -> agent_client_protocol::Result<()> {
        let mut slot = self.context.lock().map_err(|_| {
            Error::internal_error().data("ACP close owner lock poisoned".to_string())
        })?;
        if slot.is_some() {
            return Err(Error::internal_error()
                .data("ACP close owner is already bound to a connection".to_string()));
        }
        *slot = Some(context);
        Ok(())
    }

    /// 在传输层返回之后，对同一个 Session/Run 持有者重试清理。
    pub async fn close(&self) -> agent_client_protocol::Result<()> {
        let context = self
            .context
            .lock()
            .map_err(|_| Error::internal_error().data("ACP close owner lock poisoned".to_string()))?
            .clone()
            .ok_or_else(|| {
                Error::internal_error().data("ACP close owner has no connection".to_string())
            })?;
        context.close().await
    }
}

struct ConnectionCloseContext<P: AcpConnectionProfile> {
    services: Arc<AcpConnectionServices>,
    profile: Arc<P>,
    timeout: Duration,
    gate: tokio::sync::Mutex<()>,
    settled: AtomicBool,
}

impl<P: AcpConnectionProfile> AcpCloseContext for ConnectionCloseContext<P> {
    fn close(&self) -> BoxFuture<'_, agent_client_protocol::Result<()>> {
        Box::pin(async move {
            /*
            这个 gate 只用于串行化清理尝试。资源等待期间既不会复制也不会
            锁定 Session 或 Run 的状态。
            */
            let _attempt = self.gate.lock().await;
            if self.settled.load(Ordering::Acquire) {
                return Ok(());
            }
            let result =
                close_connection(self.services.as_ref(), self.profile.as_ref(), self.timeout).await;
            if result.is_ok() {
                self.settled.store(true, Ordering::Release);
            }
            result
        })
    }
}

/// 一个 ACP Agent adapter 的资源与实现元数据。
#[derive(Debug, Clone)]
pub struct AcpAdapterConfig {
    /// 在 `initialize` 中上报的程序化实现名称。
    pub name: String,
    /// 在 `initialize` 中上报的人类可读实现标题。
    pub title: String,
    /// 在 `initialize` 中上报的实现版本号。
    pub version: String,
    /// 一个 ACP 连接持有的最大存活 Session 数量。
    pub max_sessions: usize,
    /// 一次映射后的 Prompt 中允许接受的最大 Unicode 标量数。
    pub max_prompt_chars: usize,
    /// 一次投影后的 update 负载中允许发出的最大 Unicode 标量数。
    pub max_update_chars: usize,
    /// 一个 Turn 允许发出的最大标准 `session/update` 通知数量。
    pub max_updates_per_turn: usize,
    /// 一个 Turn 允许发出的累计序列化 update 字符数上限。
    pub max_total_update_chars: usize,
    /// 每个连接允许同时在途的扩展（反向回调）调用数量上限（设计文档 §12.3）。
    pub max_extension_concurrency: usize,
    /// 连接级别的 Agent 关闭所允许的总耗时。
    pub shutdown_timeout: Duration,
}

impl Default for AcpAdapterConfig {
    fn default() -> Self {
        Self {
            name: "echo-agent".to_string(),
            title: "echo-agent".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_prompt_chars: DEFAULT_MAX_PROMPT_CHARS,
            max_update_chars: DEFAULT_MAX_UPDATE_CHARS,
            max_updates_per_turn: DEFAULT_MAX_UPDATES_PER_TURN,
            max_total_update_chars: DEFAULT_MAX_TOTAL_UPDATE_CHARS,
            max_extension_concurrency: DEFAULT_MAX_EXTENSION_CONCURRENCY,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
        }
    }
}

impl AcpAdapterConfig {
    /// 校验实现元数据非空，且资源上限均为正数。
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.name.trim().is_empty()
            || self.title.trim().is_empty()
            || self.version.trim().is_empty()
        {
            return Err(crate::error::ReactError::Other(
                "ACP adapter name, title, and version must not be empty".to_string(),
            ));
        }
        if self.max_sessions == 0
            || self.max_prompt_chars == 0
            || self.max_update_chars == 0
            || self.max_updates_per_turn == 0
            || self.max_total_update_chars == 0
            || self.max_extension_concurrency == 0
        {
            return Err(crate::error::ReactError::Other(
                "ACP adapter resource limits must be positive".to_string(),
            ));
        }
        if self.shutdown_timeout.is_zero() {
            return Err(crate::error::ReactError::Other(
                "ACP adapter shutdown timeout must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// 由框架 Agent 支撑的、与传输层无关的稳定 ACP v1 Agent 角色。
pub struct AcpAgentAdapter {
    factory: Arc<dyn AcpSessionFactory>,
    config: AcpAdapterConfig,
    close_owner: AcpAdapterCloseHandle,
}

impl AcpAgentAdapter {
    /// 使用默认元数据和资源上限构造一个 adapter。
    pub fn new(factory: impl AcpSessionFactory) -> Self {
        Self {
            factory: Arc::new(factory),
            config: AcpAdapterConfig::default(),
            close_owner: AcpAdapterCloseHandle::default(),
        }
    }

    /// 使用经过显式校验的元数据与上限构造一个 adapter。
    pub fn with_config(
        factory: impl AcpSessionFactory,
        config: AcpAdapterConfig,
    ) -> crate::error::Result<Self> {
        config.validate()?;
        Ok(Self {
            factory: Arc::new(factory),
            config,
            close_owner: AcpAdapterCloseHandle::default(),
        })
    }

    /// 返回不可变的 adapter 配置。
    pub fn config(&self) -> &AcpAdapterConfig {
        &self.config
    }

    /**
    在调用 `ConnectTo::connect_to` 或把此 adapter 交给官方 Client 之前，
    先保留该连接可重试的关闭持有者。要一直持有该句柄，直到连接返回，
    并且清理成功为止。
    */
    pub fn close_owner(&self) -> AcpAdapterCloseHandle {
        self.close_owner.clone()
    }

    /**
    在连接 future 可被轮询之前就返回 close owner。
    丢弃或中止该 future 都不会导致调用方的句柄被丢弃。
    */
    pub fn connect_retaining_close_owner(
        self,
        client: impl ConnectTo<AcpRole>,
    ) -> (
        AcpAdapterCloseHandle,
        impl Future<Output = agent_client_protocol::Result<()>>,
    ) {
        let close_owner = self.close_owner();
        (close_owner, async move {
            ConnectTo::connect_to(self, client).await
        })
    }

    /**
    挂载一个已协商的扩展 profile。该 profile 贡献能力声明、hello 决策、
    逐 run 的观察者、`_meta` 桥接以及类型化的处理器；而官方连接、标准
    处理器与关闭链路仍然只会被构建一次。
    */
    pub fn with_profile<P: AcpConnectionProfile>(
        self,
        profile: P,
    ) -> AcpAgentAdapterWithProfile<P> {
        AcpAgentAdapterWithProfile {
            adapter: self,
            profile: Arc::new(profile),
        }
    }
}

/// 挂载了一个具体扩展 profile 的 [`AcpAgentAdapter`]。
pub struct AcpAgentAdapterWithProfile<P: AcpConnectionProfile> {
    adapter: AcpAgentAdapter,
    profile: Arc<P>,
}

impl<P: AcpConnectionProfile> AcpAgentAdapterWithProfile<P> {
    /**
    保留标准 ACP 处理器所使用的同一个 close owner。
    要一直持有它，直到连接返回，并且清理成功为止。
    */
    pub fn close_owner(&self) -> AcpAdapterCloseHandle {
        self.adapter.close_owner()
    }

    /// 在连接 future 被轮询之前就返回该 profile 的 close owner。
    pub fn connect_retaining_close_owner(
        self,
        client: impl ConnectTo<AcpRole>,
    ) -> (
        AcpAdapterCloseHandle,
        impl Future<Output = agent_client_protocol::Result<()>>,
    ) {
        let close_owner = self.close_owner();
        (close_owner, async move {
            ConnectTo::connect_to(self, client).await
        })
    }
}

impl<P: AcpConnectionProfile> ConnectTo<Client> for AcpAgentAdapterWithProfile<P> {
    async fn connect_to(
        self,
        client: impl ConnectTo<AcpRole>,
    ) -> agent_client_protocol::Result<()> {
        run_connection(self.adapter, self.profile, client).await
    }
}

impl ConnectTo<Client> for AcpAgentAdapter {
    async fn connect_to(
        self,
        client: impl ConnectTo<AcpRole>,
    ) -> agent_client_protocol::Result<()> {
        run_connection(self, Arc::new(NoProfile), client).await
    }
}

/**
仅标准路径所使用的 profile：从不声明能力、从不协商，也不贡献任何
处理器或观察者。它存在的意义在于让两条连接路径共享同一个泛型实现。
*/
#[derive(Clone, Copy, Default)]
struct NoProfile;

impl AcpConnectionProfile for NoProfile {
    fn negotiate_hello(&self, _hello: &serde_json::Value) -> std::result::Result<(), String> {
        Err("no extension profile is configured".to_string())
    }

    fn attach(
        &self,
        _services: Arc<AcpConnectionServices>,
    ) -> agent_client_protocol::Builder<
        AcpRole,
        impl agent_client_protocol::HandleDispatchFrom<Client>,
        impl agent_client_protocol::RunWithConnectionTo<Client>,
        impl agent_client_protocol::HandleConnectionClose<Client>,
        agent_client_protocol::RawConnectionContext,
    > {
        AcpRole.builder()
    }
}

async fn run_connection<P: AcpConnectionProfile>(
    adapter: AcpAgentAdapter,
    profile: Arc<P>,
    client: impl ConnectTo<AcpRole>,
) -> agent_client_protocol::Result<()> {
    if !adapter.close_owner.is_retained() {
        return Err(Error::internal_error().data(
            "ACP connection requires caller to retain a close owner via adapter.close_owner()"
                .to_string(),
        ));
    }
    adapter.config.validate().map_err(framework_error)?;
    let registry = Arc::new(SessionRegistry::new(
        adapter.factory,
        adapter.config.max_sessions,
    ));
    let config = Arc::new(adapter.config);
    let services = Arc::new(AcpConnectionServices::new(registry.clone(), config.clone()));
    let close_context = Arc::new(ConnectionCloseContext {
        services: Arc::clone(&services),
        profile: Arc::clone(&profile),
        timeout: config.shutdown_timeout,
        gate: tokio::sync::Mutex::new(()),
        settled: AtomicBool::new(false),
    });
    adapter.close_owner.bind(close_context.clone())?;

    /*
    `initialize`：唯一的注册点。profile 的能力声明借助
    `agentCapabilities._meta` 传递，其 hello 决策会把整个连接升级为
    Extended 模式。普通 Client（没有 `_meta` 条目）以及协商失败的情况，
    都会停留在 Standard 模式，且不会破坏标准的 initialize 流程。
    */
    let initialize_services = services.clone();
    let initialize_profile = profile.clone();
    let initialize_registry = registry.clone();
    let initialize_config = config.clone();
    /*
    `new`：标准处理器始终是 `session/new` 的唯一实现；扩展 profile
    通过 `_meta` 桥接自己的句柄。
    */
    let new_session_services = services.clone();
    let new_session_profile = profile.clone();
    /*
    `session/prompt`：和任何扩展 Run 一样进入共享的 run authority，
    因此两种 profile 观察到的都是相同的 run id 以及“同一时刻只有一个
    活跃 run”的约束。
    */
    let prompt_services = services.clone();
    let prompt_profile = profile.clone();
    let prompt_registry = registry.clone();
    let prompt_config = config.clone();
    let cancel_registry = registry.clone();
    let close_on_transport = Arc::clone(&close_context);

    let connection_result = AcpRole
        .builder()
        .name(config.name.clone())
        .on_receive_request(
            async move |request: v1::InitializeRequest,
                        responder: Responder<v1::InitializeResponse>,
                        _connection: ConnectionTo<Client>| {
                initialize_registry
                    .initialize(request.client_capabilities.clone())
                    .await;
                let mut capabilities = v1::AgentCapabilities::new();
                if let Some((key, value)) = initialize_profile.advertisement_meta() {
                    let mut meta = serde_json::Map::new();
                    meta.insert(key.clone(), value);
                    capabilities = capabilities.meta(meta);
                    let hello = request
                        .client_capabilities
                        .meta
                        .as_ref()
                        .and_then(|meta| meta.get(&key));
                    match hello {
                        None => {}
                        Some(hello) => match initialize_profile.negotiate_hello(hello) {
                            Ok(()) => {
                                initialize_services.set_mode(ConnectionMode::Extended).await;
                            }
                            Err(reason) => {
                                tracing::warn!(
                                    "echo-agent extension negotiation stayed standard: {reason}"
                                );
                            }
                        },
                    }
                }
                let response = v1::InitializeResponse::new(ProtocolVersion::V1)
                    .agent_capabilities(capabilities)
                    .agent_info(
                        v1::Implementation::new(
                            initialize_config.name.clone(),
                            initialize_config.version.clone(),
                        )
                        .title(initialize_config.title.clone()),
                    );
                responder.respond(response)
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: v1::NewSessionRequest,
                        responder: Responder<v1::NewSessionResponse>,
                        connection: ConnectionTo<Client>| {
                if let Err(error) = validate_session_paths(&request) {
                    return responder.respond_with_error(invalid_params(error));
                }
                if !new_session_services.admission_open() {
                    return responder.respond_with_error(
                        Error::internal_error().data("ACP Host is shutting down".to_string()),
                    );
                }
                let requested_cwd = request.cwd.clone();
                let services = new_session_services.clone();
                let profile = new_session_profile.clone();
                connection.spawn(async move {
                    match services.create_session(request).await {
                        Ok(session_id) => {
                            let session_text = session_id.to_string();
                            let mut response = v1::NewSessionResponse::new(session_id.clone());
                            if services.is_extended().await {
                                if let Err(error) =
                                    profile.register_standard_session(&session_text, &requested_cwd)
                                {
                                    let _ = services.sessions().close_session(&session_id).await;
                                    return responder.respond_with_error(framework_error(error));
                                }
                                if let Some(meta) = profile.annotate_standard(
                                    StandardBridgeOutcome::SessionCreated {
                                        session_id: session_text.as_str(),
                                    },
                                ) {
                                    response = response.meta(meta);
                                }
                            }
                            responder.respond(response)
                        }
                        Err(error) => responder.respond_with_error(framework_error(error)),
                    }
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |request: v1::PromptRequest,
                        responder: Responder<v1::PromptResponse>,
                        connection: ConnectionTo<Client>| {
                if !prompt_services.admission_open() {
                    return responder.respond_with_error(
                        Error::internal_error().data("ACP Host is shutting down".to_string()),
                    );
                }
                let prepared = match prepare_prompt(&prompt_registry, &prompt_config, request).await
                {
                    Ok(prepared) => prepared,
                    Err(error) => return responder.respond_with_error(error),
                };
                if !prompt_services.admission_open() {
                    return responder.respond_with_error(
                        Error::internal_error().data("ACP Host is shutting down".to_string()),
                    );
                }
                let request_cancellation = responder.cancellation();
                let task_connection = connection.clone();
                let services = prompt_services.clone();
                let profile = prompt_profile.clone();
                connection.spawn(async move {
                    drive_prompt(
                        prepared,
                        responder,
                        task_connection,
                        request_cancellation,
                        services,
                        profile,
                    )
                    .await
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |notification: v1::CancelNotification, _connection: ConnectionTo<Client>| {
                cancel_registry.cancel(&notification.session_id).await;
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_close(async move |_connection: ConnectionTo<Client>| close_on_transport.close().await)
        .with_connection_builder(profile.attach(services.clone()))
        .connect_to(client)
        .await;
    let cleanup_result = close_context.close().await;
    match (connection_result, cleanup_result) {
        (Err(connection_error), Err(cleanup_error)) => Err(Error::internal_error().data(bounded(
            &format!("ACP connection failed: {connection_error}; cleanup failed: {cleanup_error}"),
        ))),
        (Err(error), _) => Err(error),
        (Ok(()), result) => result,
    }
}

async fn close_connection<P: AcpConnectionProfile>(
    services: &AcpConnectionServices,
    profile: &P,
    timeout: Duration,
) -> agent_client_protocol::Result<()> {
    services.close_admission();
    services.extensions().cancel_all();
    let stage = (timeout / 3).max(Duration::from_millis(1));
    tokio::time::timeout(timeout, async {
        let (leaked_extensions, unsettled_runs) = tokio::join!(
            services.extensions().drain(stage),
            services.cancel_and_wait_runs_report(stage),
        );
        let settlements = tokio::time::timeout(stage, profile.wait_for_settlements(stage))
            .await
            .unwrap_or_else(|_| Err("profile settlement timed out".to_string()));
        let flush = profile.flush_before_agents();
        /*
        profile 持久化失败必须保持可见，但一旦所有调用和回执都已排空，
        就不能跳过显式的 Agent 关闭。
        */
        let agents = if leaked_extensions == 0 && unsettled_runs == 0 {
            tokio::time::timeout(stage, services.close_sessions())
                .await
                .unwrap_or_else(|_| {
                    Err(crate::error::ReactError::Other(
                        "ACP Session Agent close timed out".to_string(),
                    ))
                })
        } else {
            Err(crate::error::ReactError::Other(
                "ACP Session Agent close awaits unsettled callbacks or Runs".to_string(),
            ))
        };
        let mut failures = Vec::new();
        if leaked_extensions > 0 {
            failures.push(format!(
                "{leaked_extensions} extension invocations did not settle"
            ));
        }
        if unsettled_runs > 0 {
            failures.push(format!("{unsettled_runs} Runs lack framework receipts"));
        }
        if let Err(error) = settlements {
            failures.push(format!("profile settlement failed: {error}"));
        }
        if let Err(error) = flush {
            failures.push(format!("profile flush failed: {error}"));
        }
        if let Err(error) = agents {
            failures.push(format!("Session Agent close failed: {error}"));
        }
        if failures.is_empty() {
            profile.release_after_agents();
            Ok(())
        } else {
            Err(Error::internal_error().data(bounded(&format!(
                "ACP Session shutdown incomplete: {}",
                failures.join("; ")
            ))))
        }
    })
    .await
    .map_err(|_| Error::internal_error().data("ACP Session shutdown timed out".to_string()))?
}

struct PreparedPrompt {
    session_id: v1::SessionId,
    session: Arc<AcpSession>,
    active: ActiveTurnLease,
    turn: TurnRequest,
}

async fn prepare_prompt(
    registry: &SessionRegistry,
    config: &AcpAdapterConfig,
    request: v1::PromptRequest,
) -> std::result::Result<PreparedPrompt, Error> {
    let Some(session) = registry.get(&request.session_id).await else {
        return Err(Error::invalid_params().data("unknown ACP Session ID".to_string()));
    };
    let prompt = map_prompt(request.prompt, config.max_prompt_chars).map_err(invalid_params)?;
    let active = session.begin_turn().map_err(invalid_params)?;
    let identity = match EventIdentity::for_chat(
        Some(request.session_id.to_string()),
        active.turn.id(),
        active.turn.message_id.clone(),
        None,
    ) {
        Ok(identity) => identity,
        Err(error) => {
            drop(active);
            return Err(framework_error(error));
        }
    };
    let turn = match prompt {
        MappedPrompt::Text(text) => TurnRequest::new(identity, text),
        MappedPrompt::Structured(message) => TurnRequest::from_message(identity, message),
    }
    .mode(TurnMode::Chat)
    .cancel(active.turn.cancellation());
    Ok(PreparedPrompt {
        session_id: request.session_id,
        session,
        active,
        turn,
    })
}

async fn drive_prompt<P: AcpConnectionProfile>(
    prepared: PreparedPrompt,
    responder: Responder<v1::PromptResponse>,
    connection: ConnectionTo<Client>,
    request_cancellation: agent_client_protocol::RequestCancellation,
    services: Arc<AcpConnectionServices>,
    profile: Arc<P>,
) -> agent_client_protocol::Result<()> {
    let PreparedPrompt {
        session_id,
        session,
        active,
        turn,
        ..
    } = prepared;
    let projector = services.projector(session_id.clone(), connection.clone());
    let run_id = active.turn.id().to_string();
    let stream_id = turn.identity.stream_id.as_str().to_string();
    let observers = if services.is_extended().await {
        let session_str = session_id.to_string();
        profile.run_observers(RunObserverContext {
            session_id: session_str.as_str(),
            run_id: run_id.as_str(),
            stream_id: stream_id.as_str(),
            connection: connection.clone(),
        })
    } else {
        Vec::new()
    };
    let journal = if services.is_extended().await {
        match profile.run_journal(&run_id) {
            Ok(journal) => journal,
            Err(error) => {
                profile.rollback_run(&run_id, &stream_id);
                return responder.respond_with_error(framework_error(error));
            }
        }
    } else {
        None
    };
    let spec = RunStartSpec {
        session: session.clone(),
        active,
        run_id: run_id.clone(),
        stream_id: stream_id.clone(),
        turn,
        projector: Some(projector),
        journal,
        observers,
    };
    let (entry, task) = match services.prepare_run(spec).await {
        Ok(pair) => pair,
        Err(error) => {
            profile.rollback_run(&run_id, &stream_id);
            return responder.respond_with_error(framework_error(error));
        }
    };
    if services.is_extended().await
        && let Err(error) = profile.register_standard_run(&session_id.to_string(), entry.clone())
    {
        let _ = services.remove_run(&entry.run_id).await;
        profile.rollback_run(&entry.run_id, &entry.stream_id);
        drop(task);
        return responder.respond_with_error(framework_error(error));
    }
    if let Err(error) = connection.spawn(task) {
        let _ = services.remove_run(&entry.run_id).await;
        profile.rollback_run(&entry.run_id, &entry.stream_id);
        return responder.respond_with_error(error);
    }
    profile.run_spawned(entry.clone());
    let receipt = tokio::select! {
        receipt = entry.wait_receipt() => receipt,
        () = request_cancellation.cancelled() => {
            entry.cancel();
            entry.wait_receipt().await
        }
    };
    if let Err(error) = profile.persist_run_settled(&entry, &receipt) {
        return responder.respond_with_error(framework_error(error));
    }
    match (&receipt.outcome, &receipt.delivery) {
        (_, TurnDeliveryOutcome::Failed(failure)) => {
            return responder.respond_with_error(Error::internal_error().data(bounded(&format!(
                "turn delivery failed: {}",
                failure.message
            ))));
        }
        (TurnOutcome::Completed, TurnDeliveryOutcome::Delivered) => {}
        (TurnOutcome::Completed, delivery) => {
            return responder.respond_with_error(Error::internal_error().data(bounded(&format!(
                "turn completed with delivery status {}",
                delivery_status(delivery)
            ))));
        }
        (TurnOutcome::Cancelled, _) | (TurnOutcome::Failed(_), _) => {}
    }
    let mut response = match &receipt.outcome {
        TurnOutcome::Completed => v1::PromptResponse::new(v1::StopReason::EndTurn),
        TurnOutcome::Cancelled => v1::PromptResponse::new(v1::StopReason::Cancelled),
        TurnOutcome::Failed(failure) => {
            return responder
                .respond_with_error(Error::internal_error().data(bounded(&failure.message)));
        }
    };
    if services.is_extended().await {
        let session_str = session_id.to_string();
        if let Some(meta) = profile.annotate_standard(StandardBridgeOutcome::PromptStarted {
            session_id: session_str.as_str(),
            run_id: entry.run_id.as_str(),
            stream_id: entry.stream_id.as_str(),
        }) {
            response = response.meta(meta);
        }
    }
    responder.respond(response)
}

fn delivery_status(delivery: &TurnDeliveryOutcome) -> &'static str {
    match delivery {
        TurnDeliveryOutcome::NotAttempted => "not_attempted",
        TurnDeliveryOutcome::Delivered => "delivered",
        TurnDeliveryOutcome::Closed => "closed",
        TurnDeliveryOutcome::Failed(_) => "failed",
    }
}

fn invalid_params(error: crate::error::ReactError) -> Error {
    Error::invalid_params().data(bounded(&error.to_string()))
}

fn framework_error(error: impl std::fmt::Display) -> Error {
    Error::internal_error().data(bounded(&error.to_string()))
}

fn bounded(message: &str) -> String {
    message.chars().take(MAX_ERROR_CHARS).collect()
}
