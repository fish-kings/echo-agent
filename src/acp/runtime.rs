/*!
协议无关的单连接运行时，由标准 ACP profile 与任何协商产生的扩展 profile
共享（supreme plan 05，todo `extract-shared-acp-runtime`）。

设计不变量（参见 `docs/supreme/specs/2026-09-04-source-first-
multilanguage-sdk-runtime/design.md` §10/§11）：

- 每个连接只有一个 Session map、一个 Run 权威、一个事件账本（ledger）
  和一个取消权威。`session/prompt` 与扩展的 `run/start` 都通过
  [`AcpConnectionServices::begin_run`] 进入，因此同一个 Session 永远
  不会同时携带两个并发的 run，且两个入口观察到的是同一批 run id。
- 每个被接受的 [`EventEnvelope`] 都会先提交到 run 账本（若配置了
  持久化日志钩子，也会先经过它），然后才会被渲染为标准的
  `session/update` 投影并转发给扩展观察者。这两种视图都只是
  同一个已提交事实的投影；两者都不能凭空产生序列号或终态。
- 状态、终态与回执（receipt）事实只能由框架的
  [`AgentTurnDriver`] 及其 [`TurnReceipt`] 推进——本模块只负责存储它们，
  从不自行推导。

根 crate 保持协议无关：扩展协商 payload 是不透明的
`serde_json::Value`，带类型的 `_echo_agent/*` handler 由 profile
实现者通过 [`AcpConnectionProfile::attach`] 使用官方的 builder 组合方式挂载。
*/

use crate::agent::EventEnvelope;
use crate::error::{ReactError, Result};
use crate::runtime::{AgentTurnDriver, EventSink, SinkControl, TurnReceipt, TurnRequest};
use agent_client_protocol::schema::v1::SessionId;
use agent_client_protocol::{Client, ConnectionTo};
use async_trait::async_trait;
use echo_state::journal::EventJournal;
use futures::future::BoxFuture;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::{Notify, RwLock};

use super::extension::ExtensionInvocationAuthority;
use super::projection::AcpEventProjector;
use super::session::{AcpSession, ActiveTurnLease, SessionRegistry};

/// 每个内存中 run 账本保留的信封（envelope）字节数的默认上限。
const DEFAULT_MAX_LEDGER_BYTES: usize = 8 * 1024 * 1024;
/// 每个内存中 run 账本保留的信封数量的默认上限。
const DEFAULT_MAX_LEDGER_EVENTS: usize = 10_000;

/**
协商后的连接模式。`Standard` 是无条件的默认模式；只有当某个 profile 的
hello 协商接受了该 Client 时，才会进入 `Extended`。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionMode {
    /// 标准 ACP 模式（默认）。
    #[default]
    Standard,
    /// 已通过某个 profile 的 hello 协商，进入扩展模式。
    Extended,
}

/**
单个 run 的内存事件账本的边界。对于没有配置日志钩子的 run，该账本是其
实时重放权威；处理较慢的消费者永远不会使账本超出这些边界——它们会观察到
一个带类型的缺口（gap），而不是无限增长的内存。
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcpLedgerLimits {
    /// 保留的最大事件数量。
    pub max_events: usize,
    /// 保留的最大总字节数。
    pub max_bytes: usize,
}

impl Default for AcpLedgerLimits {
    /// 返回默认的账本上限配置。
    fn default() -> Self {
        Self {
            max_events: DEFAULT_MAX_LEDGER_EVENTS,
            max_bytes: DEFAULT_MAX_LEDGER_BYTES,
        }
    }
}

/**
单个 run 的事件账本：以持久化优先、带边界的方式作为该 run 已提交信封的权威。
可选的日志钩子会在进入内存环形缓冲区之前先接收每个信封；日志失败会通过
驱动方的回执报告为投递失败。当生产者已经产生过一个执行终态时，本账本
不会再制造第二个终态。
*/
pub struct EventLedger {
    /// 账本的容量边界。
    limits: AcpLedgerLimits,
    /// 可选的持久化日志钩子。
    journal: Option<Arc<dyn EventJournal<EventEnvelope>>>,
    /// 内存态的账本内容，受互斥锁保护。
    inner: StdMutex<LedgerInner>,
}

/// 内存账本的可变内部状态。
struct LedgerInner {
    /// 已提交的信封，按顺序保存。
    events: VecDeque<EventEnvelope>,
    /// 当前保留信封的总字节数。
    total_bytes: usize,
    /// 已见过的最大序列号。
    last_sequence: u64,
}

impl EventLedger {
    /// 使用给定的容量边界与可选日志钩子创建一个新的事件账本。
    pub fn new(
        limits: AcpLedgerLimits,
        journal: Option<Arc<dyn EventJournal<EventEnvelope>>>,
    ) -> Self {
        Self {
            limits,
            journal,
            inner: StdMutex::new(LedgerInner {
                events: VecDeque::new(),
                total_bytes: 0,
                last_sequence: 0,
            }),
        }
    }

    /**
    提交一个信封：先写入可选的日志钩子，再追加到内存账本，
    并在超出容量边界时从队首淘汰最旧的信封。
    */
    pub fn commit(&self, envelope: EventEnvelope) -> Result<()> {
        if let Some(journal) = &self.journal {
            journal.append(envelope.clone()).map_err(|error| {
                ReactError::Other(format!("run event journal rejected the envelope: {error}"))
            })?;
        }
        let serialized = serde_json::to_vec(&envelope).map_err(|error| {
            ReactError::Other(format!("event envelope is not encodable: {error}"))
        })?;
        let bytes = serialized.len();
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| ReactError::Other("run ledger lock poisoned".to_string()))?;
        inner.last_sequence = inner.last_sequence.max(envelope.sequence);
        inner.events.push_back(envelope);
        inner.total_bytes = inner
            .total_bytes
            .checked_add(bytes)
            .ok_or_else(|| ReactError::Other("run ledger byte accounting overflow".to_string()))?;
        while inner.events.len() > self.limits.max_events
            || inner.total_bytes > self.limits.max_bytes
        {
            let Some(evicted) = inner.events.pop_front() else {
                break;
            };
            let evicted_bytes = serde_json::to_vec(&evicted)
                .map(|bytes| bytes.len())
                .unwrap_or_default();
            inner.total_bytes = inner.total_bytes.saturating_sub(evicted_bytes);
        }
        Ok(())
    }

    /// 严格晚于 `after_sequence` 的已提交信封，按顺序返回。
    pub fn events_after(&self, after_sequence: u64) -> Vec<EventEnvelope> {
        let Ok(inner) = self.inner.lock() else {
            return Vec::new();
        };
        inner
            .events
            .iter()
            .filter(|envelope| envelope.sequence > after_sequence)
            .cloned()
            .collect()
    }

    /// 内存中仍保留的最旧信封序列号（为空时返回 0）。
    pub fn retained_floor(&self) -> u64 {
        let Ok(inner) = self.inner.lock() else {
            return 0;
        };
        inner
            .events
            .front()
            .map(|envelope| envelope.sequence)
            .unwrap_or(0)
    }

    /// 已提交的最大序列号（为空时返回 0）。
    pub fn last_sequence(&self) -> u64 {
        let Ok(inner) = self.inner.lock() else {
            return 0;
        };
        inner.last_sequence
    }

    /**
    日志钩子（若存在）报告的持久化保留下限。
    重放请求若需要低于该下限的序列号，会返回一个带类型的缺口（gap），
    而不是悄悄跳过这些事实。
    */
    pub fn journal_retained_floor(&self) -> Option<u64> {
        self.journal
            .as_ref()
            .map(|journal| journal.retained_floor())
    }

    /// 日志钩子（若存在）的最大持久化序列号。
    pub fn journal_last_sequence(&self) -> Option<u64> {
        self.journal.as_ref().map(|journal| journal.last_sequence())
    }

    /// 若日志钩子暴露了同步操作，则刷新（flush）该日志钩子。
    pub fn flush(&self) -> Result<()> {
        /* `EventJournal` 本身没有 flush 方法；分段实现各自维护自己的
        持久化契约。钩子的职责在于日志本身——账本只保证每个信封
        都被传递给了它。 */
        Ok(())
    }
}

/**
提交事件的扩展侧消费者。观察者在账本提交之后、按信封顺序运行；
出错会在驱动方的回执上变为投递失败。实时投递的边界（ACK 窗口、缺口）
是观察者实现自身的契约，而不是账本的执行权威。
*/
#[async_trait]
pub trait RunEventObserver: Send + Sync {
    /// 处理一个刚刚提交到账本的信封。
    async fn on_committed_event(&self, envelope: &EventEnvelope) -> Result<()>;
}

/**
一个 run 启动时所需的一切。由入口点（标准 Prompt handler 或扩展 Run
handler）在取得该 Session 唯一的 run 名额之后构建。
*/
pub struct RunStartSpec {
    /// 该 run 所属的 Session。
    pub session: Arc<AcpSession>,
    /// 已占用的活跃 turn 名额租约。
    pub active: ActiveTurnLease,
    /// 该 run 的 id。
    pub run_id: String,
    /// 该 run 对应的流 id。
    pub stream_id: String,
    /// 要驱动的 turn 请求。
    pub turn: TurnRequest,
    /// 该 run 的标准 `session/update` 投影。
    pub projector: Option<AcpEventProjector>,
    /// 持久化日志钩子；为 `None` 时账本仅保存在内存中。
    pub journal: Option<Arc<dyn EventJournal<EventEnvelope>>>,
    /// 账本提交之后要调用的扩展观察者。
    pub observers: Vec<Arc<dyn RunEventObserver>>,
}

/**
一次已注册的执行。无论是标准入口还是扩展入口，`run_id` 都等于该
Session 的 turn id，因此两个入口共享同一个身份标识。
*/
pub struct RunEntry {
    /// 该 run 的 id。
    pub run_id: String,
    /// 该 run 所属的 Session id。
    pub session_id: SessionId,
    /// 该 run 对应的流 id。
    pub stream_id: String,
    /// 该 run 的事件账本。
    pub ledger: Arc<EventLedger>,
    /// 该 run 的取消令牌。
    cancel: crate::agent::CancellationToken,
    /// 该 run 当前的生命周期状态。
    lifecycle: StdMutex<RunLifecycle>,
    /// 用于通知等待者“已结算”的通知器。
    settled: Notify,
    /// 该 run 是否已被撤销（未启动即失败）。
    revoked: AtomicBool,
}

/// 一次 run 的生命周期状态。
#[derive(Clone)]
enum RunLifecycle {
    /// 仍在运行中。
    Running,
    /// 已结算，携带框架产生的最终回执。
    Settled(Arc<TurnReceipt>),
}

impl RunEntry {
    /// 用给定的标识信息与取消令牌构造一个新的运行中 RunEntry。
    pub(crate) fn new(
        run_id: String,
        session_id: SessionId,
        stream_id: String,
        ledger: Arc<EventLedger>,
        cancel: crate::agent::CancellationToken,
    ) -> Self {
        Self {
            run_id,
            session_id,
            stream_id,
            ledger,
            cancel,
            lifecycle: StdMutex::new(RunLifecycle::Running),
            settled: Notify::new(),
            revoked: AtomicBool::new(false),
        }
    }

    /// 该 run 的框架取消令牌。
    pub fn cancellation(&self) -> crate::agent::CancellationToken {
        self.cancel.clone()
    }

    /**
    请求取消。与自然完成之间的竞态由框架自身的终态语义解决——
    本方法只负责发出取消信号。
    */
    pub fn cancel(&self) -> bool {
        if self.is_running() {
            self.cancel.cancel();
            true
        } else {
            false
        }
    }

    /// 撤销该 run（用于尚未启动即失败的场景），并一并发出取消信号。
    pub(crate) fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
        self.cancel();
    }

    /// 该 run 是否已被撤销。
    pub(crate) fn is_revoked(&self) -> bool {
        self.revoked.load(Ordering::Acquire)
    }

    /// 该 run 当前是否仍在运行中。
    pub fn is_running(&self) -> bool {
        matches!(
            self.lifecycle
                .lock()
                .map(|lifecycle| lifecycle.clone())
                .unwrap_or(RunLifecycle::Running),
            RunLifecycle::Running
        )
    }

    /// 若框架已产生回执，则返回该已结算回执。
    pub fn receipt(&self) -> Option<Arc<TurnReceipt>> {
        self.lifecycle
            .lock()
            .ok()
            .and_then(|lifecycle| match lifecycle.clone() {
                RunLifecycle::Running => None,
                RunLifecycle::Settled(receipt) => Some(receipt),
            })
    }

    /// 用框架产生的回执结算该 run（仅在仍处于 Running 状态时生效一次）。
    pub(crate) fn settle(&self, receipt: TurnReceipt) {
        if let Ok(mut lifecycle) = self.lifecycle.lock()
            && matches!(*lifecycle, RunLifecycle::Running)
        {
            *lifecycle = RunLifecycle::Settled(Arc::new(receipt));
            self.settled.notify_waiters();
        }
    }

    /**
    等待并返回框架产生的唯一回执。绝不会凭空制造终态：
    如果驱动任务在结算前被中止，本方法会永远等待下去——
    关闭路径应改为取消 run 并等待其结算。
    */
    pub async fn wait_receipt(&self) -> Arc<TurnReceipt> {
        loop {
            let notified = self.settled.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(receipt) = self.receipt() {
                return receipt;
            }
            notified.await;
        }
    }
}

/// 一个连接的 run 注册表：唯一的 Run 权威。
#[derive(Default)]
pub(crate) struct RunRegistry {
    /// 按 run id 索引的已注册 run。
    runs: RwLock<HashMap<String, Arc<RunEntry>>>,
}

impl RunRegistry {
    /// 注册一个新 run；若该 run id 已存在则报错。
    async fn register(&self, entry: Arc<RunEntry>) -> Result<()> {
        let mut runs = self.runs.write().await;
        if runs.contains_key(&entry.run_id) {
            return Err(ReactError::Other(format!(
                "run id {} is already registered",
                entry.run_id
            )));
        }
        runs.insert(entry.run_id.clone(), entry);
        Ok(())
    }

    /// 按 run id 查询已注册的 run。
    pub async fn get(&self, run_id: &str) -> Option<Arc<RunEntry>> {
        self.runs.read().await.get(run_id).cloned()
    }

    /// 从注册表中移除并返回该 run。
    pub async fn remove(&self, run_id: &str) -> Option<Arc<RunEntry>> {
        self.runs.write().await.remove(run_id)
    }

    /**
    取消所有存活的 run，并（有限时地）等待它们结算。
    驱动任务被中止的条目会通过中止路径结算；等待中的 run 会在
    其回执出现、或该条目随连接一起被丢弃时立即解除等待。
    */
    async fn cancel_and_wait(&self, wait: Duration) -> usize {
        let entries: Vec<Arc<RunEntry>> = self.runs.read().await.values().cloned().collect();
        for entry in &entries {
            entry.cancel();
        }
        let _ = tokio::time::timeout(
            wait,
            futures::future::join_all(entries.iter().map(|entry| {
                let entry = entry.clone();
                async move {
                    if entry.is_running() {
                        /* 与驱动任务存在竞态：要么回执在这里到达，
                        要么连接的拆除逻辑中止该任务。 */
                        let _ = entry.wait_receipt().await;
                    }
                }
            })),
        )
        .await;
        let mut runs = self.runs.write().await;
        runs.retain(|_, entry| entry.is_running());
        runs.len()
    }

    /// 当前已注册的 run 数量。
    async fn run_count(&self) -> usize {
        self.runs.read().await.len()
    }
}

/**
以账本为先的组合 sink：先提交到账本（含日志），再渲染标准投影，
最后转发给扩展观察者。任何错误都会作为投递失败上报；驱动方在调用本
sink 之前已经观察到的生产者终态会被保留。
*/
struct SharedRunSink {
    /// 事件账本。
    ledger: Arc<EventLedger>,
    /// 标准 `session/update` 投影器。
    projector: Option<AcpEventProjector>,
    /// 需要转发的扩展观察者。
    observers: Vec<Arc<dyn RunEventObserver>>,
}

#[async_trait]
impl EventSink for SharedRunSink {
    /// 将事件提交到共享账本，并转发给下游。
    async fn on_event(&self, envelope: EventEnvelope) -> Result<SinkControl> {
        self.ledger.commit(envelope.clone())?;
        if let Some(projector) = &self.projector {
            projector.emit(&envelope).await?;
        }
        for observer in &self.observers {
            observer.on_committed_event(&envelope).await?;
        }
        Ok(SinkControl::Continue)
    }
}

/// 每个连接的共享运行时：两种 ACP profile 共用的唯一 Session/Run/事件权威。
pub struct AcpConnectionServices {
    /// 当前连接模式。
    mode: RwLock<ConnectionMode>,
    /// 唯一的 Session 权威。
    sessions: Arc<SessionRegistry>,
    /// 唯一的 Run 权威。
    runs: RunRegistry,
    /// 新建账本时使用的容量边界。
    ledger_limits: AcpLedgerLimits,
    /// 适配器配置（更新窗口边界等）。
    config: Arc<super::adapter::AcpAdapterConfig>,
    /// 是否仍接受新的 Session/Run 准入。
    admission_open: AtomicBool,
    /// 连接范围内的扩展调用权威。
    extensions: Arc<ExtensionInvocationAuthority>,
}

impl AcpConnectionServices {
    /// 基于给定的 Session 注册表与适配器配置创建一个新的连接服务实例。
    pub fn new(
        sessions: Arc<SessionRegistry>,
        config: Arc<super::adapter::AcpAdapterConfig>,
    ) -> Self {
        /* 适配器配置在连接被创建之前就已完成校验，所以非正的并发数
        理论上不会到达这里；万一发生，则降级为单个许可而不是 panic。 */
        let extensions = ExtensionInvocationAuthority::new(config.max_extension_concurrency)
            .unwrap_or_else(|error| {
                tracing::warn!(
                    "invalid extension concurrency {} ({}), falling back to 1",
                    config.max_extension_concurrency,
                    error
                );
                ExtensionInvocationAuthority::new_saturating(1)
            });
        Self {
            mode: RwLock::new(ConnectionMode::Standard),
            sessions,
            runs: RunRegistry::default(),
            ledger_limits: AcpLedgerLimits::default(),
            config,
            admission_open: AtomicBool::new(true),
            extensions,
        }
    }

    /**
    连接范围内的扩展调用权威（设计文档 §12.3）。
    由该连接上的每一个扩展代理共享；绝不会携带第二个
    run/session/终态权威。
    */
    pub fn extensions(&self) -> &Arc<ExtensionInvocationAuthority> {
        &self.extensions
    }

    /// 覆盖内存账本的容量边界（扩展 profile 会根据其协商出的限制推导该值）。
    pub fn with_ledger_limits(mut self, limits: AcpLedgerLimits) -> Self {
        self.ledger_limits = limits;
        self
    }

    /// 以可变引用形式覆盖内存账本的容量边界。
    pub fn with_ledger_limits_mut(&mut self, limits: AcpLedgerLimits) -> &mut Self {
        self.ledger_limits = limits;
        self
    }

    /// 当前的账本容量边界。
    pub fn ledger_limits(&self) -> AcpLedgerLimits {
        self.ledger_limits
    }

    /// 该连接唯一的 Session 权威。
    pub fn sessions(&self) -> &Arc<SessionRegistry> {
        &self.sessions
    }

    /// 当前的连接模式。
    pub async fn mode(&self) -> ConnectionMode {
        *self.mode.read().await
    }

    /// 该连接是否已进入扩展模式。
    pub async fn is_extended(&self) -> bool {
        self.mode().await == ConnectionMode::Extended
    }

    /// 该连接当前是否仍接受新的准入（Session/Run）。
    pub fn admission_open(&self) -> bool {
        self.admission_open.load(Ordering::Acquire)
    }

    /// 关闭该连接的准入：不再接受新的 Session、Run 或扩展调用。
    pub fn close_admission(&self) {
        self.admission_open.store(false, Ordering::Release);
        self.sessions.close_admission();
        self.extensions.close_admission();
    }

    /// 若准入已关闭则返回错误，否则放行。
    pub fn ensure_admission(&self) -> Result<()> {
        if self.admission_open() {
            Ok(())
        } else {
            Err(ReactError::Other("ACP Host is shutting down".to_string()))
        }
    }

    /// 创建一个新 Session；若在创建期间准入被关闭，则回滚该 Session。
    pub async fn create_session(
        &self,
        request: agent_client_protocol::schema::v1::NewSessionRequest,
    ) -> Result<SessionId> {
        self.ensure_admission()?;
        let session_id = self.sessions.create(request).await?;
        if !self.admission_open() {
            let _ = self.sessions.close_session(&session_id).await;
            return Err(ReactError::Other("ACP Host is shutting down".to_string()));
        }
        Ok(session_id)
    }

    /// 插入一个已构造好的 Session 上下文；若在插入期间准入被关闭，则回滚该 Session。
    pub async fn insert_session(
        &self,
        context: super::session::AcpSessionContext,
    ) -> Result<SessionId> {
        self.ensure_admission()?;
        let session_id = self.sessions.insert_session(context).await?;
        if !self.admission_open() {
            let _ = self.sessions.close_session(&session_id).await;
            return Err(ReactError::Other("ACP Host is shutting down".to_string()));
        }
        Ok(session_id)
    }

    /// 设置该连接的连接模式。
    pub(crate) async fn set_mode(&self, mode: ConnectionMode) {
        *self.mode.write().await = mode;
    }

    /// 按 id 解析一个 run。
    pub async fn run(&self, run_id: &str) -> Option<Arc<RunEntry>> {
        self.runs.get(run_id).await
    }

    /**
    返回某个 Session 所拥有的活跃 run 的取消令牌。
    一个 Session 至多允许一个活跃 run，因此对于那些公开方法不携带
    令牌的 trait 回调而言，这是一个无歧义的桥接方式。
    */
    pub async fn active_run_cancellation(
        &self,
        session_id: &str,
    ) -> Option<crate::agent::CancellationToken> {
        self.runs
            .runs
            .read()
            .await
            .values()
            .find(|entry| entry.session_id.to_string() == session_id && entry.is_running())
            .map(|entry| entry.cancellation())
    }

    /**
    在后续的准备工作或连接创建失败时，移除一个已准备好的 run。
    丢弃返回的任务 future 会释放该 Session 的租约；本方法只负责
    移除共享的 Run 注册记录。
    */
    pub async fn remove_run(&self, run_id: &str) -> Option<Arc<RunEntry>> {
        let entry = self.runs.remove(run_id).await;
        if let Some(entry) = &entry {
            entry.revoke();
        }
        entry
    }

    /// 使用适配器的边界配置，为一个 run 构建标准投影器。
    pub fn projector(
        &self,
        session_id: SessionId,
        connection: ConnectionTo<Client>,
    ) -> AcpEventProjector {
        AcpEventProjector::new(
            session_id,
            connection,
            self.config.max_update_chars,
            self.config.max_updates_per_turn,
            self.config.max_total_update_chars,
        )
    }

    /**
    注册一个 run，并返回供调用方在该连接上 spawn 的驱动任务。
    注册操作会预留该 run id；驱动只有在被 spawn 的任务真正运行后才会开始。
    重复的 run id 属于调用方的 bug，会直接失败且不会触碰该 Session 的
    run 名额。
    */
    pub async fn prepare_run(
        self: &Arc<Self>,
        spec: RunStartSpec,
    ) -> Result<(
        Arc<RunEntry>,
        BoxFuture<'static, agent_client_protocol::Result<()>>,
    )> {
        self.ensure_admission()?;
        let ledger = Arc::new(EventLedger::new(self.ledger_limits, spec.journal));
        let entry = Arc::new(RunEntry::new(
            spec.run_id.clone(),
            spec.session.context.session_id.clone(),
            spec.stream_id.clone(),
            ledger.clone(),
            spec.active.turn.cancellation(),
        ));
        self.runs.register(entry.clone()).await?;
        if !self.admission_open() {
            let _ = self.runs.remove(&entry.run_id).await;
            entry.cancel();
            return Err(ReactError::Other("ACP Host is shutting down".to_string()));
        }

        let sink = Arc::new(SharedRunSink {
            ledger,
            projector: spec.projector,
            observers: spec.observers,
        });
        let agent = spec.session.agent.clone();
        let turn = spec.turn;
        let drive_entry = entry.clone();
        let lease = spec.active;
        let task: BoxFuture<'static, agent_client_protocol::Result<()>> = Box::pin(async move {
            let receipt = if drive_entry.is_revoked() {
                match TurnReceipt::cancelled(drive_entry.run_id.clone()) {
                    Ok(receipt) => receipt,
                    Err(error) => {
                        return Err(
                            agent_client_protocol::Error::internal_error().data(error.to_string())
                        );
                    }
                }
            } else {
                AgentTurnDriver
                    .drive(agent.as_ref(), turn, sink.as_ref())
                    .await
            };
            drive_entry.settle(receipt);
            drop(lease);
            Ok(())
        });
        Ok((entry, task))
    }

    /**
    按 id 取消一个 run。对未知或已结算的 run 返回 false；
    唯一的终态仍然来自框架本身。
    */
    pub async fn cancel_run(&self, run_id: &str) -> bool {
        let Some(entry) = self.runs.get(run_id).await else {
            return false;
        };
        entry.cancel()
    }

    /// 该连接上已注册的 run 数量（包含存活与已结算的）。
    pub async fn run_count(&self) -> usize {
        self.runs.run_count().await
    }

    /**
    关闭扩展与 Run 的准入，取消这两类调用，等待它们的持有者，
    然后关闭 Session Agent。
    */
    pub async fn close(&self, timeout: Duration) -> Result<()> {
        self.close_admission();
        self.extensions.cancel_all();
        let stage = (timeout / 2).max(Duration::from_millis(1));
        tokio::time::timeout(timeout, async {
            let (extensions, runs) = tokio::join!(
                self.extensions.drain(stage),
                self.cancel_and_wait_runs_report(stage),
            );
            if extensions > 0 || runs > 0 {
                return Err(ReactError::Other(format!(
                    "ACP close retained {extensions} extension invocation(s) and {runs} Run(s) without framework receipts"
                )));
            }
            tokio::time::timeout(stage, self.close_sessions())
                .await
                .map_err(|_| ReactError::Other("ACP Session Agent close timed out".to_string()))?
        })
        .await
        .map_err(|_| ReactError::Other("ACP connection close timed out".to_string()))?
    }

    /// 取消所有存活的 run 并等待其结算（结果被丢弃）。
    pub async fn cancel_and_wait_runs(&self, timeout: Duration) {
        let _ = self.runs.cancel_and_wait(timeout).await;
    }

    /// 取消所有存活的 run 并等待其结算，返回超时后仍未结算的 run 数量。
    pub(crate) async fn cancel_and_wait_runs_report(&self, timeout: Duration) -> usize {
        self.runs.cancel_and_wait(timeout).await
    }

    /// 关闭该连接上的全部 Session。
    pub async fn close_sessions(&self) -> Result<()> {
        self.sessions.close_all().await
    }
}

/**
profile 可能为标准响应附加扩展句柄信息的场景
（`_meta` 桥接，设计文档 §10.4）。
*/
#[derive(Debug, Clone, Copy)]
pub enum StandardBridgeOutcome<'a> {
    /// 该 ACP Session id 上的 `session/new` 已成功。
    SessionCreated { session_id: &'a str },
    /// 在该 Session 上以给定的 run/stream id 启动了一个 Prompt run。
    PromptStarted {
        /// 所属的 Session id。
        session_id: &'a str,
        /// 该 run 的 id。
        run_id: &'a str,
        /// 该 run 对应的流 id。
        stream_id: &'a str,
    },
}

/// profile 在为一个 run 构建其扩展观察者时所获得的上下文。
pub struct RunObserverContext<'a> {
    /// 所属的 Session id。
    pub session_id: &'a str,
    /// 该 run 的 id。
    pub run_id: &'a str,
    /// 该 run 对应的流 id。
    pub stream_id: &'a str,
    /// 该连接的 Client 连接句柄。
    pub connection: ConnectionTo<Client>,
}

/**
构建在共享连接运行时之上、可组合的扩展 profile。

适配器只维护一个官方连接：`initialize`、stdio、读写循环、
标准 handler 以及关闭链都只会被构建一次。一个 profile 贡献：
(a) 在 `agentCapabilities._meta.<key>` 下的能力广播；
(b) 决定是否将连接提升为 [`ConnectionMode::Extended`] 的 hello 判定；
(c) 每个 run 的事件观察者；
(d) 标准响应上的 `_meta` 桥接；
(e) 通过 [`AcpConnectionProfile::attach`] 提供的带类型 handler ——
适配器会把它与官方的 `Builder::with_connection_builder` 合并。
*/
pub trait AcpConnectionProfile: Send + Sync + 'static {
    /**
    在每个 `initialize` 响应中发布的 `(_meta key, value)`。
    返回 `None` 表示永不广播（此时该 profile 表现为纯标准模式）。
    */
    fn advertisement_meta(&self) -> Option<(String, serde_json::Value)> {
        None
    }

    /**
    根据原始 Client hello 决定连接模式。`Ok(())` 进入扩展模式；
    `Err(reason)` 保持标准模式且不会使标准 `initialize` 失败。
    一个完全没有 `_meta` 条目的普通 Client 永远不会到达本方法。
    */
    fn negotiate_hello(&self, hello: &serde_json::Value) -> std::result::Result<(), String>;

    /**
    为扩展模式下的句柄桥接，给一次成功的标准响应附加元信息。
    返回 `Some(meta)` 会设置该响应的 `_meta`。
    */
    fn annotate_standard(
        &self,
        outcome: StandardBridgeOutcome<'_>,
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        let _ = outcome;
        None
    }

    /**
    注册一个通过标准 ACP `session/new` 路径创建的 Session。
    扩展 profile 借此让响应中的桥接句柄指向同一个 Session 注册表条目。
    */
    fn register_standard_session(
        &self,
        _session_id: &str,
        _cwd: &std::path::Path,
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /**
    在驱动任务被 spawn 之前，注册一个通过标准 ACP `session/prompt`
    创建的 Run。profile 可以借此持久化其启动记录，并签发与扩展
    请求相同的 Run/Stream 句柄。
    */
    fn register_standard_run(
        &self,
        _session_id: &str,
        _entry: Arc<RunEntry>,
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /**
    返回某个 Run 的持久化日志钩子。仅标准模式的 profile 返回
    `None`；核心 profile 会为两种 prompt 入口路径都使用它。
    */
    fn run_journal(
        &self,
        _run_id: &str,
    ) -> std::result::Result<Option<Arc<dyn EventJournal<EventEnvelope>>>, String> {
        Ok(None)
    }

    /**
    在一个 Run 结算之后，持久化框架产生的确切回执。默认 profile
    什么也不做；具备持久化能力的 profile 应存储该回执，且不产生
    第二个终态。
    */
    fn persist_run_settled(
        &self,
        _entry: &RunEntry,
        _receipt: &TurnReceipt,
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /**
    撤销共享运行时能够 spawn 驱动之前、profile 侧已经完成的注册，
    使失败的标准 Prompt 请求保持事务性。
    */
    fn rollback_run(&self, _run_id: &str, _stream_id: &str) {}

    /**
    通知 profile：驱动任务已被官方连接接受。该钩子不得持久化终态
    回执；[`Self::persist_run_settled`] 才是标准 Prompt 唯一的
    持久化结算权威。
    */
    fn run_spawned(&self, _entry: Arc<RunEntry>) {}

    /**
    在 Run 取消/等待完成之后、Session Agent/MCP 关闭之前，
    刷新 profile 自身持有的持久化状态。
    */
    fn flush_before_agents(&self) -> std::result::Result<(), String> {
        Ok(())
    }

    /**
    在 Session Agent/MCP 关闭之后，释放连接持有的 profile 资源
    （例如扩展注册）。这是设计文档 §12.3/§16 拆卸顺序中的最后一步。
    */
    fn release_after_agents(&self) {}

    /// 等待该 profile 自身的结算工作完成。
    fn wait_for_settlements(
        &self,
        _timeout: Duration,
    ) -> BoxFuture<'static, std::result::Result<(), String>> {
        Box::pin(async { Ok(()) })
    }

    /**
    扩展模式下、某个 run 的扩展观察者集合。在驱动 spawn 之前，
    每次 run 启动只调用一次。
    */
    fn run_observers(&self, context: RunObserverContext<'_>) -> Vec<Arc<dyn RunEventObserver>> {
        let _ = context;
        Vec::new()
    }

    /**
    构建该 profile 带类型的 handler builder。适配器会把它合并到标准
    handler 之后，因此始终只存在一条官方的分发链。扩展 handler 必须
    自行检查 [`AcpConnectionServices::is_extended`]，并在连接仍为
    标准模式时回答“方法未找到”。
    */
    fn attach(
        &self,
        services: Arc<AcpConnectionServices>,
    ) -> agent_client_protocol::Builder<
        agent_client_protocol::Agent,
        impl agent_client_protocol::HandleDispatchFrom<agent_client_protocol::Client>,
        impl agent_client_protocol::RunWithConnectionTo<agent_client_protocol::Client>,
        impl agent_client_protocol::HandleConnectionClose<agent_client_protocol::Client>,
        agent_client_protocol::RawConnectionContext,
    >;
}

/// 运行关闭流程的单元测试。
#[cfg(test)]
mod close_tests {
    use super::*;

    /// 验证运行关闭超时后，未结算的框架回执会被保留以便重试。
    #[tokio::test]
    async fn run_close_timeout_keeps_unsettled_framework_receipt_for_retry() -> Result<()> {
        let runs = RunRegistry::default();
        let cancellation = crate::agent::CancellationToken::new();
        let entry = Arc::new(RunEntry::new(
            "run-close-timeout".to_string(),
            SessionId::new("session-close-timeout"),
            "stream-close-timeout".to_string(),
            Arc::new(EventLedger::new(AcpLedgerLimits::default(), None)),
            cancellation.clone(),
        ));
        runs.register(Arc::clone(&entry)).await?;

        runs.cancel_and_wait(Duration::from_millis(20)).await;
        assert!(cancellation.is_cancelled());
        assert_eq!(runs.run_count().await, 1);

        entry.settle(TurnReceipt::cancelled(entry.run_id.clone())?);
        runs.cancel_and_wait(Duration::from_secs(1)).await;
        assert_eq!(runs.run_count().await, 0);
        Ok(())
    }
}
