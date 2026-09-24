use crate::agent::{Agent, CancellationToken};
use crate::error::{ReactError, Result};
use agent_client_protocol::schema::v1::{
    ClientCapabilities, McpServer, Meta, NewSessionRequest, SessionId,
};
use futures::future::BoxFuture;
use futures::future::join_all;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Weak};
use tokio::sync::{Mutex, Notify, RwLock};

/**
 * 用于创建一个独立框架 Agent 的完整 ACP 上下文。
 *
 * 具体实现可以将常规构造工作委托给
 * [`crate::agent::factory::AgentFactory`]。这个协议专属的钩子还携带了
 * 通用工厂配置未持有的异步 Session 建立输入，包括 MCP 声明与协商后的
 * Client 能力。
 */
#[derive(Debug, Clone)]
pub struct AcpSessionContext {
    /// 生成的稳定 ACP Session 身份标识。
    pub session_id: SessionId,
    /// 来自 `session/new` 的绝对主工作目录。
    pub cwd: PathBuf,
    /// Client 请求的额外绝对工作区根目录。
    pub additional_directories: Vec<PathBuf>,
    /// 在返回 Agent 之前必须准备好的 MCP 声明。
    pub mcp_servers: Vec<McpServer>,
    /// 在 `initialize` 期间捕获的 Client 能力快照。
    pub client_capabilities: ClientCapabilities,
    /// 为 Session 工厂保留的带命名空间的请求元数据。
    pub meta: Option<Meta>,
}

/**
 * 创建独立拥有一个 ACP Session 历史记录的框架 Agent。
 *
 * 稳定版 ACP 要求支持 ResourceLink 提示（prompt）。因此返回的 Agent
 * 在 Client 发送 ResourceLink 时必须实现结构化聊天方法；纯文本 Agent
 * 应显式让该 Prompt 失败，而不是退化为扁平化的私有文本约定。工厂在
 * 成功返回之前，还必须准备好 [`AcpSessionContext`] 中的每一条 MCP 声明。
 */
pub trait AcpSessionFactory: Send + Sync + 'static {
    /// 准备并返回一个独占拥有该 Session 的新 Agent。
    fn create_session(
        &self,
        context: AcpSessionContext,
    ) -> BoxFuture<'static, Result<Box<dyn Agent>>>;
}

impl<F, Fut> AcpSessionFactory for F
where
    F: Fn(AcpSessionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Box<dyn Agent>>> + Send + 'static,
{
    fn create_session(
        &self,
        context: AcpSessionContext,
    ) -> BoxFuture<'static, Result<Box<dyn Agent>>> {
        Box::pin((self)(context))
    }
}

/// 一次 Session 上正在进行的执行的取消权柄。
#[derive(Clone)]
pub struct ActiveTurn {
    /// 当前活跃执行的稳定运行身份（`<session>:turn:<n>`）。
    pub(crate) id: String,
    /// 该次执行对应的消息身份标识。
    pub(crate) message_id: String,
    /// 与框架驱动器共享的取消令牌。
    pub(crate) cancel: CancellationToken,
}

impl ActiveTurn {
    /// 当前活跃执行的稳定运行身份（`<session>:turn:<n>`）。
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 与框架驱动器共享的取消令牌。
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
}

/**
 * 对某个 Session 唯一活跃运行槽位的独占租约。释放（drop）该租约会
 * 释放该槽位；标准 Prompt 与扩展 Run 在其驱动器的整个生命周期内
 * 恰好持有一份租约。
 */
pub struct ActiveTurnLease {
    /// 所属 Session 的弱引用；释放租约时用于回写空闲槽位。
    pub(crate) session: Weak<AcpSession>,
    /// 本次租约对应的活跃运行。
    pub turn: ActiveTurn,
}

impl Drop for ActiveTurnLease {
    fn drop(&mut self) {
        let Some(session) = self.session.upgrade() else {
            return;
        };
        let mut slot = session
            .turn
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if slot
            .active
            .as_ref()
            .is_some_and(|active| active.id == self.turn.id)
        {
            slot.active = None;
            session.turn_settled.notify_waiters();
        }
    }
}

struct TurnSlot {
    next_sequence: u64,
    active: Option<ActiveTurn>,
}

#[derive(Debug, Clone)]
enum SessionClosePhase {
    Prepared,
    Running,
    Failed(String),
    Settled,
}

#[derive(Debug)]
struct SessionCloseAttempt {
    next_generation: u64,
    current: Option<Arc<SessionCloseReceipt>>,
}

#[derive(Debug)]
struct SessionCloseReceipt {
    generation: u64,
    phase: StdMutex<SessionClosePhase>,
    settled: Notify,
    #[cfg(test)]
    abort: StdMutex<Option<tokio::task::AbortHandle>>,
}

/**
 * 一次 ACP Session 关闭尝试的第二阶段所使用的类型化能力凭证。
 *
 * 该租约可克隆，使框架与门面（facade）层的等待方能够加入同一次尝试。
 * 失败的尝试在当前这一代（generation）内保持稳定；调用方必须再次调用
 * [`SessionRegistry::begin_close_session`] 才能获得一次重试。
 */
#[derive(Clone)]
pub struct SessionCloseLease {
    /// 签发该租约的注册表身份标识，用于校验租约归属。
    registry_id: String,
    /// 被关闭的 Session 标识。
    session_id: SessionId,
    /// 被关闭的 Session 本身。
    session: Arc<AcpSession>,
    /// 该次关闭尝试的共享回执。
    receipt: Arc<SessionCloseReceipt>,
}

impl SessionCloseLease {
    /// 该租约所关闭的 Session 标识。
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// 该租约对应的关闭尝试代数（generation）。
    pub fn generation(&self) -> u64 {
        self.receipt.generation
    }
}

impl std::fmt::Debug for SessionCloseLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionCloseLease")
            .field("session_id", &self.session_id)
            .field("generation", &self.receipt.generation)
            .finish_non_exhaustive()
    }
}

/**
 * 一个 ACP Session：一个独立的框架 Agent 加上该 Session 作用域内的运行
 * 槽位。标准 profile 与协商出的扩展 profile 共享同一份结构 —— 刻意
 * 没有维护第二份 Session 映射表。
 */
pub struct AcpSession {
    /// 创建该 Session 时使用的上下文（cwd、MCP 声明、Client 能力等）。
    pub context: AcpSessionContext,
    /// 该 Session 独立拥有的框架 Agent。
    pub agent: Arc<dyn Agent>,
    /// 该 Session 唯一活跃运行槽位的内部状态。
    turn: StdMutex<TurnSlot>,
    /// 活跃运行结算时用于唤醒等待方的通知器。
    turn_settled: Notify,
    /// 该 Session 是否已被标记为关闭。
    closed: AtomicBool,
    /// 当前正在进行或已完成的关闭尝试状态。
    close_attempt: StdMutex<SessionCloseAttempt>,
}

impl AcpSession {
    fn new(context: AcpSessionContext, agent: Box<dyn Agent>) -> Self {
        Self {
            context,
            agent: Arc::from(agent),
            turn: StdMutex::new(TurnSlot {
                next_sequence: 0,
                active: None,
            }),
            turn_settled: Notify::new(),
            closed: AtomicBool::new(false),
            close_attempt: StdMutex::new(SessionCloseAttempt {
                next_generation: 0,
                current: None,
            }),
        }
    }

    /**
     * 为一次新的 Prompt Turn 预留唯一的活跃运行槽位；若 Session 已关闭
     * 或已存在活跃运行，则返回错误。
     */
    pub fn begin_turn(self: &Arc<Self>) -> Result<ActiveTurnLease> {
        let mut slot = self.turn.lock().unwrap_or_else(|error| error.into_inner());
        if self.closed.load(Ordering::Acquire) {
            return Err(ReactError::Other("ACP Session is closed".to_string()));
        }
        if slot.active.is_some() {
            return Err(ReactError::Other(
                "ACP Session already has an active Prompt Turn".to_string(),
            ));
        }
        let next = slot
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| ReactError::Other("ACP Session turn sequence exhausted".to_string()))?;
        slot.next_sequence = next;
        let session_id = self.context.session_id.to_string();
        let active = ActiveTurn {
            id: format!("{session_id}:turn:{next}"),
            message_id: format!("{session_id}:message:{next}"),
            cancel: CancellationToken::new(),
        };
        slot.active = Some(active.clone());
        Ok(ActiveTurnLease {
            session: Arc::downgrade(self),
            turn: active,
        })
    }

    /// 取消当前活跃的运行（若存在）；返回是否确实取消了一次运行。
    pub fn cancel_active(&self) -> bool {
        let cancel = self
            .turn
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .active
            .as_ref()
            .map(|active| active.cancel.clone());
        if let Some(cancel) = cancel {
            cancel.cancel();
            true
        } else {
            false
        }
    }

    fn mark_closed(&self) {
        let _slot = self.turn.lock().unwrap_or_else(|error| error.into_inner());
        self.closed.store(true, Ordering::Release);
    }

    fn prepare_close(self: &Arc<Self>, registry_id: &str) -> Result<SessionCloseLease> {
        self.mark_closed();
        self.cancel_active();
        let mut attempt = self
            .close_attempt
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let current_phase = attempt.current.as_ref().map(|receipt| {
            receipt
                .phase
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        });
        let receipt = match current_phase {
            Some(
                SessionClosePhase::Prepared
                | SessionClosePhase::Running
                | SessionClosePhase::Settled,
            ) => attempt.current.as_ref().cloned().ok_or_else(|| {
                ReactError::Other("ACP Session close receipt disappeared".to_string())
            })?,
            Some(SessionClosePhase::Failed(_)) | None => {
                let next = attempt.next_generation.checked_add(1).ok_or_else(|| {
                    ReactError::Other("ACP Session close generation exhausted".to_string())
                })?;
                attempt.next_generation = next;
                let receipt = Arc::new(SessionCloseReceipt {
                    generation: next,
                    phase: StdMutex::new(SessionClosePhase::Prepared),
                    settled: Notify::new(),
                    #[cfg(test)]
                    abort: StdMutex::new(None),
                });
                attempt.current = Some(Arc::clone(&receipt));
                receipt
            }
        };
        Ok(SessionCloseLease {
            registry_id: registry_id.to_string(),
            session_id: self.context.session_id.clone(),
            session: Arc::clone(self),
            receipt,
        })
    }

    /// 等待，直到该 Session 当前没有活跃运行为止。
    pub async fn wait_until_idle(&self) {
        loop {
            let notified = self.turn_settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self
                .turn
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .active
                .is_none()
            {
                return;
            }
            notified.await;
        }
    }
}

/// 一个连接范围内所有 ACP Session 的注册表，负责创建、查找与关闭 Session。
pub struct SessionRegistry {
    /// 该注册表的稳定身份标识，用于校验关闭租约的归属。
    registry_id: String,
    /// 用于创建每个 Session 独立 Agent 的工厂。
    factory: Arc<dyn AcpSessionFactory>,
    /// 允许同时存在的最大 Session 数量。
    max_sessions: usize,
    /// 是否仍接受新的 Session 创建请求。
    admission_open: AtomicBool,
    /// 序列化 Session 创建与关闭之间的竞态窗口的互斥锁。
    creation_gate: Mutex<()>,
    /// 在 `initialize` 期间捕获的 Client 能力快照。
    client_capabilities: RwLock<Option<ClientCapabilities>>,
    /// 当前所有活跃的 Session，按 Session 标识索引。
    sessions: RwLock<HashMap<SessionId, Arc<AcpSession>>>,
}

enum FinishCloseAction {
    Start,
    Wait,
    Settled,
    Failed(String),
}

#[cfg(test)]
struct FinishCloseBarrier {
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

struct SessionCloseTaskOwner {
    receipt: Arc<SessionCloseReceipt>,
    armed: bool,
}

impl SessionCloseTaskOwner {
    fn new(receipt: Arc<SessionCloseReceipt>) -> Self {
        Self {
            receipt,
            armed: true,
        }
    }

    fn settle(&mut self, phase: SessionClosePhase) {
        *self
            .receipt
            .phase
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = phase;
        #[cfg(test)]
        {
            *self
                .receipt
                .abort
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
        }
        self.armed = false;
        self.receipt.settled.notify_waiters();
    }
}

impl Drop for SessionCloseTaskOwner {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut phase = self
            .receipt
            .phase
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if matches!(&*phase, SessionClosePhase::Running) {
            *phase = SessionClosePhase::Failed(
                "owned Agent close task ended before publishing settlement".to_string(),
            );
        }
        #[cfg(test)]
        {
            *self
                .receipt
                .abort
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = None;
        }
        drop(phase);
        self.receipt.settled.notify_waiters();
    }
}

async fn run_session_close_attempt(session: Arc<AcpSession>, mut owner: SessionCloseTaskOwner) {
    session.wait_until_idle().await;
    let result = session.agent.close().await;
    owner.settle(match result {
        Ok(()) => SessionClosePhase::Settled,
        Err(error) => SessionClosePhase::Failed(error.to_string()),
    });
}

impl SessionRegistry {
    /// 创建一个新的 Session 注册表，绑定给定的工厂与 Session 数量上限。
    pub fn new(factory: Arc<dyn AcpSessionFactory>, max_sessions: usize) -> Self {
        Self {
            registry_id: uuid::Uuid::new_v4().to_string(),
            factory,
            max_sessions,
            admission_open: AtomicBool::new(true),
            creation_gate: Mutex::new(()),
            client_capabilities: RwLock::new(None),
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// 记录 `initialize` 阶段协商出的 Client 能力，供后续创建 Session 时使用。
    pub async fn initialize(&self, capabilities: ClientCapabilities) {
        *self.client_capabilities.write().await = Some(capabilities);
    }

    /// 永久关闭新 Session 的准入；已存在的 Session 不受影响。
    pub fn close_admission(&self) {
        self.admission_open.store(false, Ordering::Release);
    }

    fn ensure_admission(&self) -> Result<()> {
        if self.admission_open.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(ReactError::Other(
                "ACP Session registry is closing".to_string(),
            ))
        }
    }

    /**
     * 处理标准的 `session/new` 请求：校验路径、生成新的 Session 身份
     * 标识，并将其插入注册表。
     */
    pub async fn create(&self, request: NewSessionRequest) -> Result<SessionId> {
        validate_session_paths(&request)?;
        let context = AcpSessionContext {
            session_id: SessionId::new(format!("sess_{}", uuid::Uuid::new_v4())),
            cwd: request.cwd,
            additional_directories: request.additional_directories,
            mcp_servers: request.mcp_servers,
            client_capabilities: self.capabilities().await?,
            meta: request.meta,
        };
        self.insert_session(context).await
    }

    /// 在 `initialize` 期间捕获的 Client 能力快照。
    pub async fn capabilities(&self) -> Result<ClientCapabilities> {
        self.client_capabilities
            .read()
            .await
            .clone()
            .ok_or_else(|| ReactError::Other("ACP connection is not initialized".to_string()))
    }

    /**
     * 插入一个已完整描述的 Session，通过该连接唯一的工厂创建它独立的
     * Agent。标准的 `session/new` 与扩展 profile 的 create/load 都会
     * 汇聚到这里，从而使 Session 数量上限与按 Session 划分的 Agent
     * 所有权保持权威一致。
     */
    pub async fn insert_session(&self, context: AcpSessionContext) -> Result<SessionId> {
        let _creation = self.creation_gate.lock().await;
        self.ensure_admission()?;
        if self.sessions.read().await.len() >= self.max_sessions {
            return Err(ReactError::Other(format!(
                "ACP Session limit {} reached",
                self.max_sessions
            )));
        }
        if self.sessions.read().await.contains_key(&context.session_id) {
            return Err(ReactError::Other(format!(
                "ACP Session {} is already registered",
                context.session_id
            )));
        }
        let agent = self.factory.create_session(context.clone()).await?;
        /* ACP Session 各自拥有独立的 Agent，因此把 Session 的 cwd 绑定为
         * Agent 的默认工作目录不会跨会话泄漏。这样做也能保持适配器与
         * 纯文本 Agent 实现的兼容性。 */
        agent.set_working_dir(Some(context.cwd.clone()));
        let session_id = context.session_id.clone();
        let session = Arc::new(AcpSession::new(context, agent));
        if !self.admission_open.load(Ordering::Acquire) {
            /* 与异步工厂发生竞态的关闭流程仍然拥有这个 Agent。创建关口
             * （creation gate）释放后，将其保留为一个已关闭的 Session，
             * 以便 close_all 能处理到它；调用方绝不能拿到可用的句柄。 */
            session.mark_closed();
            self.sessions.write().await.insert(session_id, session);
            return Err(ReactError::Other(
                "ACP Session registry is closing".to_string(),
            ));
        }
        self.sessions
            .write()
            .await
            .insert(session_id.clone(), session);
        Ok(session_id)
    }

    /// 按 Session 标识查找一个活跃的 Session；不存在时返回 `None`。
    pub async fn get(&self, session_id: &SessionId) -> Option<Arc<AcpSession>> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /**
     * 开始关闭一个 Session，但不等待其活跃运行结束。
     *
     * 返回的租约证明新回合（turn）已被拦截、且当前回合的取消已经发出。
     * 门面（facade）层可以在调用 [`Self::finish_close_session`] 之前，
     * 先耗尽自己持有的操作租约。
     */
    pub async fn begin_close_session(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionCloseLease>> {
        let _creation = self.creation_gate.lock().await;
        let session = self.sessions.read().await.get(session_id).cloned();
        let Some(session) = session else {
            return Ok(None);
        };
        session.prepare_close(&self.registry_id).map(Some)
    }

    /**
     * 在门面层持有的操作耗尽之后，完成一次关闭尝试。
     *
     * 持有同一份租约的并发调用方会加入同一次 Agent 关闭尝试。失败的
     * 尝试会保留该 Session，并把同样的失败结果返回给每一个等待方。
     * 需要重试时应再次调用 `begin_close_session` 获取新的租约。
     */
    pub async fn finish_close_session(&self, lease: SessionCloseLease) -> Result<()> {
        self.finish_close_session_inner(lease, None).await
    }

    async fn finish_close_session_inner(
        &self,
        lease: SessionCloseLease,
        #[cfg(test)] mut barrier: Option<FinishCloseBarrier>,
        #[cfg(not(test))] _barrier: Option<()>,
    ) -> Result<()> {
        if lease.registry_id != self.registry_id {
            return Err(ReactError::Other(format!(
                "ACP Session {} close lease belongs to another registry",
                lease.session_id
            )));
        }

        loop {
            let notified = lease.receipt.settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let action = {
                let mut phase = lease
                    .receipt
                    .phase
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                match phase.clone() {
                    SessionClosePhase::Prepared => {
                        *phase = SessionClosePhase::Running;
                        FinishCloseAction::Start
                    }
                    SessionClosePhase::Running => FinishCloseAction::Wait,
                    SessionClosePhase::Settled => FinishCloseAction::Settled,
                    SessionClosePhase::Failed(error) => FinishCloseAction::Failed(error),
                }
            };
            match action {
                FinishCloseAction::Start => {
                    let session = Arc::clone(&lease.session);
                    match tokio::runtime::Handle::try_current() {
                        Ok(handle) => {
                            let owner = SessionCloseTaskOwner::new(Arc::clone(&lease.receipt));
                            let close_task =
                                handle.spawn(run_session_close_attempt(session, owner));
                            #[cfg(test)]
                            {
                                *lease
                                    .receipt
                                    .abort
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner()) =
                                    Some(close_task.abort_handle());
                            }
                            #[cfg(not(test))]
                            let _close_task = close_task;
                        }
                        Err(error) => {
                            let message = format!(
                                "ACP Session close requires an active Tokio runtime: {error}"
                            );
                            *lease
                                .receipt
                                .phase
                                .lock()
                                .unwrap_or_else(|lock_error| lock_error.into_inner()) =
                                SessionClosePhase::Failed(message);
                            lease.receipt.settled.notify_waiters();
                        }
                    }
                }
                FinishCloseAction::Wait => {
                    #[cfg(test)]
                    if let Some(barrier) = barrier.take() {
                        barrier.entered.notify_waiters();
                        barrier.release.notified().await;
                        continue;
                    }
                    notified.await;
                }
                FinishCloseAction::Settled => {
                    let mut sessions = self.sessions.write().await;
                    if sessions
                        .get(&lease.session_id)
                        .is_some_and(|current| Arc::ptr_eq(current, &lease.session))
                    {
                        sessions.remove(&lease.session_id);
                    }
                    return Ok(());
                }
                FinishCloseAction::Failed(error) => {
                    return Err(ReactError::Other(format!(
                        "ACP Session {} close did not settle: {error}",
                        lease.session_id
                    )));
                }
            }
        }
    }

    /// 通过标准的两阶段生命周期关闭一个 Session。
    pub async fn close_session(&self, session_id: &SessionId) -> Result<bool> {
        let Some(lease) = self.begin_close_session(session_id).await? else {
            return Ok(false);
        };
        self.finish_close_session(lease).await?;
        Ok(true)
    }

    /// 取消指定 Session 当前活跃的运行；Session 不存在时返回 `false`。
    pub async fn cancel(&self, session_id: &SessionId) -> bool {
        let session = self.get(session_id).await;
        if let Some(session) = session {
            session.cancel_active()
        } else {
            false
        }
    }

    /**
     * 关闭注册表中所有 Session：先永久停止新的准入，再并发地对每个
     * Session 执行两阶段关闭，并汇总任何未能结算的失败信息。
     */
    pub async fn close_all(&self) -> Result<()> {
        self.close_admission();
        let leases = {
            let _creation = self.creation_gate.lock().await;
            let guard = self.sessions.read().await;
            guard
                .values()
                .map(|session| session.prepare_close(&self.registry_id))
                .collect::<Result<Vec<_>>>()?
        };
        let results = join_all(leases.into_iter().map(|lease| async move {
            let id = lease.session_id.clone();
            (id, self.finish_close_session(lease).await)
        }))
        .await;
        let mut failures = Vec::new();
        for (id, result) in results {
            if let Err(error) = result {
                failures.push(format!("{id}: {error}"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(ReactError::Other(format!(
                "ACP Session close_all failed to settle {} Session(s): {}",
                failures.len(),
                failures.join("; ")
            )))
        }
    }
}

pub(crate) fn validate_session_paths(request: &NewSessionRequest) -> Result<()> {
    if !request.cwd.is_absolute() {
        return Err(ReactError::Other(
            "ACP Session cwd must be absolute".to_string(),
        ));
    }
    if request
        .additional_directories
        .iter()
        .any(|path| !path.is_absolute())
    {
        return Err(ReactError::Other(
            "ACP Session additional directories must be absolute".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentEvent;
    use futures::stream::{self, BoxStream, StreamExt};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    struct CloseProbeAgent {
        hang: bool,
        close_started: Arc<AtomicBool>,
    }

    struct RetryCloseProbeAgent {
        failures_remaining: std::sync::atomic::AtomicUsize,
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    struct ControlledCloseProbeAgent {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl Agent for CloseProbeAgent {
        fn name(&self) -> &str {
            "close-probe"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
            Box::pin(async move {
                self.close_started.store(true, Ordering::Release);
                if self.hang {
                    futures::future::pending::<Result<()>>().await
                } else {
                    Ok(())
                }
            })
        }

        fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move { Ok(task.to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            let answer = task.to_string();
            Box::pin(
                async move { Ok(stream::iter(vec![Ok(AgentEvent::FinalAnswer(answer))]).boxed()) },
            )
        }
    }

    impl Agent for RetryCloseProbeAgent {
        fn name(&self) -> &str {
            "retry-close-probe"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
            let attempt = self
                .attempts
                .fetch_add(1, Ordering::AcqRel)
                .saturating_add(1);
            let fail = self
                .failures_remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok();
            Box::pin(async move {
                if fail {
                    Err(ReactError::Other(format!(
                        "injected Session close failure attempt {attempt}"
                    )))
                } else {
                    Ok(())
                }
            })
        }

        fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move { Ok(task.to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            let answer = task.to_string();
            Box::pin(
                async move { Ok(stream::iter(vec![Ok(AgentEvent::FinalAnswer(answer))]).boxed()) },
            )
        }
    }

    impl Agent for ControlledCloseProbeAgent {
        fn name(&self) -> &str {
            "controlled-close-probe"
        }

        fn model_name(&self) -> &str {
            "test"
        }

        fn system_prompt(&self) -> &str {
            "test"
        }

        fn close<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
            self.attempts.fetch_add(1, Ordering::AcqRel);
            let started = Arc::clone(&self.started);
            let release = Arc::clone(&self.release);
            Box::pin(async move {
                started.notify_waiters();
                release.notified().await;
                Ok(())
            })
        }

        fn execute<'a>(&'a self, task: &'a str) -> BoxFuture<'a, Result<String>> {
            Box::pin(async move { Ok(task.to_string()) })
        }

        fn execute_stream<'a>(
            &'a self,
            task: &'a str,
        ) -> BoxFuture<'a, Result<BoxStream<'a, Result<AgentEvent>>>> {
            let answer = task.to_string();
            Box::pin(
                async move { Ok(stream::iter(vec![Ok(AgentEvent::FinalAnswer(answer))]).boxed()) },
            )
        }
    }

    fn context(id: &str) -> AcpSessionContext {
        AcpSessionContext {
            session_id: SessionId::new(id.to_string()),
            cwd: std::env::temp_dir(),
            additional_directories: Vec::new(),
            mcp_servers: Vec::new(),
            client_capabilities: ClientCapabilities::default(),
            meta: None,
        }
    }

    async fn wait_for(predicate: impl Fn() -> bool) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(1), async {
            while !predicate() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| ReactError::Other("timed out waiting for close probe".to_string()))
    }

    #[tokio::test]
    async fn close_all_cancels_every_turn_before_concurrent_agent_close() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = Arc::new(SessionRegistry::new(unused_factory, 2));
        let first_close = Arc::new(AtomicBool::new(false));
        let second_close = Arc::new(AtomicBool::new(false));
        let first = Arc::new(AcpSession::new(
            context("first"),
            Box::new(CloseProbeAgent {
                hang: true,
                close_started: first_close.clone(),
            }),
        ));
        let second = Arc::new(AcpSession::new(
            context("second"),
            Box::new(CloseProbeAgent {
                hang: false,
                close_started: second_close.clone(),
            }),
        ));
        let first_turn = first.begin_turn()?;
        let second_turn = second.begin_turn()?;
        let first_cancel = first_turn.turn.cancel.clone();
        let second_cancel = second_turn.turn.cancel.clone();
        {
            let mut sessions = registry.sessions.write().await;
            sessions.insert(first.context.session_id.clone(), first);
            sessions.insert(second.context.session_id.clone(), second);
        }

        let close_task = tokio::spawn({
            let registry = registry.clone();
            async move { registry.close_all().await }
        });
        wait_for(|| first_cancel.is_cancelled() && second_cancel.is_cancelled()).await?;
        drop(first_turn);
        drop(second_turn);
        wait_for(|| first_close.load(Ordering::Acquire) && second_close.load(Ordering::Acquire))
            .await?;
        close_task.abort();
        let _ = close_task.await;
        Ok(())
    }

    #[tokio::test]
    async fn close_all_retains_failed_sessions_for_retry() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = SessionRegistry::new(unused_factory, 2);
        let failed_attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let settled_attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failed = Arc::new(AcpSession::new(
            context("failed"),
            Box::new(RetryCloseProbeAgent {
                failures_remaining: std::sync::atomic::AtomicUsize::new(1),
                attempts: Arc::clone(&failed_attempts),
            }),
        ));
        let settled = Arc::new(AcpSession::new(
            context("settled"),
            Box::new(RetryCloseProbeAgent {
                failures_remaining: std::sync::atomic::AtomicUsize::new(0),
                attempts: Arc::clone(&settled_attempts),
            }),
        ));
        {
            let mut sessions = registry.sessions.write().await;
            sessions.insert(failed.context.session_id.clone(), Arc::clone(&failed));
            sessions.insert(settled.context.session_id.clone(), settled);
        }

        assert!(registry.close_all().await.is_err());
        let sessions = registry.sessions.read().await;
        assert_eq!(sessions.len(), 1);
        assert!(
            sessions
                .get(&failed.context.session_id)
                .is_some_and(|current| Arc::ptr_eq(current, &failed))
        );
        drop(sessions);
        assert_eq!(failed_attempts.load(Ordering::Acquire), 1);
        assert_eq!(settled_attempts.load(Ordering::Acquire), 1);

        registry.close_all().await?;
        assert!(registry.sessions.read().await.is_empty());
        assert_eq!(failed_attempts.load(Ordering::Acquire), 2);
        assert_eq!(settled_attempts.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[tokio::test]
    async fn registry_close_permanently_fences_session_creation() -> Result<()> {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let factory: Arc<dyn AcpSessionFactory> = Arc::new({
            let attempts = Arc::clone(&attempts);
            move |_context| {
                let attempts = Arc::clone(&attempts);
                async move {
                    attempts.fetch_add(1, Ordering::AcqRel);
                    Ok(Box::new(RetryCloseProbeAgent {
                        failures_remaining: std::sync::atomic::AtomicUsize::new(0),
                        attempts: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                    }) as Box<dyn Agent>)
                }
            }
        });
        let registry = SessionRegistry::new(factory, 2);
        registry.close_all().await?;
        assert!(registry.insert_session(context("late")).await.is_err());
        assert_eq!(attempts.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[tokio::test]
    async fn begin_close_cancels_before_external_drain_and_finish() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = SessionRegistry::new(unused_factory, 1);
        let close_started = Arc::new(AtomicBool::new(false));
        let session = Arc::new(AcpSession::new(
            context("two-phase"),
            Box::new(CloseProbeAgent {
                hang: false,
                close_started: Arc::clone(&close_started),
            }),
        ));
        let turn = session.begin_turn()?;
        let cancellation = turn.turn.cancel.clone();
        registry
            .sessions
            .write()
            .await
            .insert(session.context.session_id.clone(), Arc::clone(&session));

        let lease = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("close lease missing".to_string()))?;
        assert!(cancellation.is_cancelled());
        assert!(session.begin_turn().is_err());
        assert!(registry.get(&session.context.session_id).await.is_some());
        assert!(!close_started.load(Ordering::Acquire));

        /* 门面（facade）层在这里耗尽自己的操作租约。框架回合（turn）会
         * 一直保持注册状态，直到这次外部耗尽释放其运行所有者为止。 */
        drop(turn);
        registry.finish_close_session(lease).await?;
        assert!(close_started.load(Ordering::Acquire));
        assert!(registry.get(&session.context.session_id).await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_finish_joins_one_agent_close_attempt() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = Arc::new(SessionRegistry::new(unused_factory, 1));
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let session = Arc::new(AcpSession::new(
            context("single-flight"),
            Box::new(ControlledCloseProbeAgent {
                attempts: Arc::clone(&attempts),
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
        ));
        registry
            .sessions
            .write()
            .await
            .insert(session.context.session_id.clone(), Arc::clone(&session));
        let lease = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("close lease missing".to_string()))?;

        let close_started = started.notified();
        let first = tokio::spawn({
            let registry = Arc::clone(&registry);
            let lease = lease.clone();
            async move { registry.finish_close_session(lease).await }
        });
        let second = tokio::spawn({
            let registry = Arc::clone(&registry);
            async move { registry.finish_close_session(lease).await }
        });
        tokio::time::timeout(Duration::from_secs(1), close_started)
            .await
            .map_err(|_| ReactError::Other("Agent close did not start".to_string()))?;
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        release.notify_one();
        first
            .await
            .map_err(|error| ReactError::Other(format!("first finish join failed: {error}")))??;
        second
            .await
            .map_err(|error| ReactError::Other(format!("second finish join failed: {error}")))??;
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        assert!(registry.get(&session.context.session_id).await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn aborted_owned_close_task_publishes_failure_and_allows_retry() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = Arc::new(SessionRegistry::new(unused_factory, 1));
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let session = Arc::new(AcpSession::new(
            context("aborted-owner"),
            Box::new(ControlledCloseProbeAgent {
                attempts: Arc::clone(&attempts),
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
        ));
        registry
            .sessions
            .write()
            .await
            .insert(session.context.session_id.clone(), Arc::clone(&session));
        let lease = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("close lease missing".to_string()))?;

        let close_started = started.notified();
        let finish = tokio::spawn({
            let registry = Arc::clone(&registry);
            let lease = lease.clone();
            async move { registry.finish_close_session(lease).await }
        });
        tokio::time::timeout(Duration::from_secs(1), close_started)
            .await
            .map_err(|_| ReactError::Other("Agent close did not start".to_string()))?;
        let abort = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let abort = lease
                    .receipt
                    .abort
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone();
                if let Some(abort) = abort {
                    return abort;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| ReactError::Other("owned close abort handle missing".to_string()))?;
        abort.abort();
        let error = tokio::time::timeout(Duration::from_secs(1), finish)
            .await
            .map_err(|_| ReactError::Other("finish waiter remained stuck".to_string()))?
            .map_err(|join_error| {
                ReactError::Other(format!("finish waiter join failed: {join_error}"))
            })?
            .err()
            .ok_or_else(|| ReactError::Other("aborted close unexpectedly settled".to_string()))?;
        assert!(error.to_string().contains("ended before publishing"));
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        assert!(registry.get(&session.context.session_id).await.is_some());

        let retry = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("retry close lease missing".to_string()))?;
        release.notify_one();
        registry.finish_close_session(retry).await?;
        assert_eq!(attempts.load(Ordering::Acquire), 2);
        assert!(registry.get(&session.context.session_id).await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn late_settled_lease_does_not_conflict_with_same_id_replacement() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = SessionRegistry::new(unused_factory, 1);
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let original = Arc::new(AcpSession::new(
            context("reused-id"),
            Box::new(RetryCloseProbeAgent {
                failures_remaining: std::sync::atomic::AtomicUsize::new(0),
                attempts,
            }),
        ));
        registry
            .sessions
            .write()
            .await
            .insert(original.context.session_id.clone(), Arc::clone(&original));
        let lease = registry
            .begin_close_session(&original.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("close lease missing".to_string()))?;
        let late = lease.clone();
        registry.finish_close_session(lease).await?;

        let replacement = Arc::new(AcpSession::new(
            context("reused-id"),
            Box::new(CloseProbeAgent {
                hang: false,
                close_started: Arc::new(AtomicBool::new(false)),
            }),
        ));
        registry.sessions.write().await.insert(
            replacement.context.session_id.clone(),
            Arc::clone(&replacement),
        );

        registry.finish_close_session(late).await?;
        let current = registry
            .get(&replacement.context.session_id)
            .await
            .ok_or_else(|| ReactError::Other("replacement Session was removed".to_string()))?;
        assert!(Arc::ptr_eq(&current, &replacement));
        Ok(())
    }

    #[tokio::test]
    async fn running_waiter_uses_receipt_after_same_id_replacement_race() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = Arc::new(SessionRegistry::new(unused_factory, 1));
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let close_started = Arc::new(Notify::new());
        let close_release = Arc::new(Notify::new());
        let original = Arc::new(AcpSession::new(
            context("running-reused-id"),
            Box::new(ControlledCloseProbeAgent {
                attempts,
                started: Arc::clone(&close_started),
                release: Arc::clone(&close_release),
            }),
        ));
        registry
            .sessions
            .write()
            .await
            .insert(original.context.session_id.clone(), Arc::clone(&original));
        let lease = registry
            .begin_close_session(&original.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("close lease missing".to_string()))?;

        let started = close_started.notified();
        let first = tokio::spawn({
            let registry = Arc::clone(&registry);
            let lease = lease.clone();
            async move { registry.finish_close_session(lease).await }
        });
        tokio::time::timeout(Duration::from_secs(1), started)
            .await
            .map_err(|_| ReactError::Other("Agent close did not start".to_string()))?;

        let waiter_entered = Arc::new(Notify::new());
        let waiter_release = Arc::new(Notify::new());
        let entered = waiter_entered.notified();
        let waiter_entered_for_late = Arc::clone(&waiter_entered);
        let waiter_release_for_late = Arc::clone(&waiter_release);
        let late = tokio::spawn({
            let registry = Arc::clone(&registry);
            async move {
                registry
                    .finish_close_session_inner(
                        lease,
                        Some(FinishCloseBarrier {
                            entered: waiter_entered_for_late,
                            release: waiter_release_for_late,
                        }),
                    )
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(1), entered)
            .await
            .map_err(|_| ReactError::Other("running waiter did not reach barrier".to_string()))?;

        close_release.notify_one();
        first
            .await
            .map_err(|error| ReactError::Other(format!("first finish join failed: {error}")))??;
        let replacement = Arc::new(AcpSession::new(
            context("running-reused-id"),
            Box::new(CloseProbeAgent {
                hang: false,
                close_started: Arc::new(AtomicBool::new(false)),
            }),
        ));
        registry.sessions.write().await.insert(
            replacement.context.session_id.clone(),
            Arc::clone(&replacement),
        );
        waiter_release.notify_one();
        late.await
            .map_err(|error| ReactError::Other(format!("late finish join failed: {error}")))??;

        let current = registry
            .get(&replacement.context.session_id)
            .await
            .ok_or_else(|| ReactError::Other("replacement Session was removed".to_string()))?;
        assert!(Arc::ptr_eq(&current, &replacement));
        Ok(())
    }

    #[tokio::test]
    async fn failed_finish_requires_new_begin_generation_for_retry() -> Result<()> {
        let unused_factory: Arc<dyn AcpSessionFactory> = Arc::new(|_context| async {
            Err(ReactError::Other(
                "factory is unused in this test".to_string(),
            ))
        });
        let registry = SessionRegistry::new(unused_factory, 1);
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let session = Arc::new(AcpSession::new(
            context("retry-generation"),
            Box::new(RetryCloseProbeAgent {
                failures_remaining: std::sync::atomic::AtomicUsize::new(2),
                attempts: Arc::clone(&attempts),
            }),
        ));
        registry
            .sessions
            .write()
            .await
            .insert(session.context.session_id.clone(), Arc::clone(&session));
        let first = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("first close lease missing".to_string()))?;
        let late_first = first.clone();
        let first_error = registry
            .finish_close_session(first)
            .await
            .err()
            .ok_or_else(|| ReactError::Other("first close unexpectedly settled".to_string()))?
            .to_string();
        assert!(first_error.contains("attempt 1"));
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        assert!(registry.get(&session.context.session_id).await.is_some());

        let second = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("second close lease missing".to_string()))?;
        assert_eq!(second.generation(), 2);
        let second_error = registry
            .finish_close_session(second)
            .await
            .err()
            .ok_or_else(|| ReactError::Other("second close unexpectedly settled".to_string()))?
            .to_string();
        assert!(second_error.contains("attempt 2"));
        assert_eq!(attempts.load(Ordering::Acquire), 2);

        let late_first_error = registry
            .finish_close_session(late_first)
            .await
            .err()
            .ok_or_else(|| ReactError::Other("late first lease unexpectedly settled".to_string()))?
            .to_string();
        assert_eq!(late_first_error, first_error);
        assert_eq!(attempts.load(Ordering::Acquire), 2);

        let retry = registry
            .begin_close_session(&session.context.session_id)
            .await?
            .ok_or_else(|| ReactError::Other("retry close lease missing".to_string()))?;
        assert_eq!(retry.generation(), 3);
        registry.finish_close_session(retry).await?;
        assert_eq!(attempts.load(Ordering::Acquire), 3);
        assert!(registry.get(&session.context.session_id).await.is_none());
        Ok(())
    }
}
