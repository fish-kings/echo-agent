/*!
阻塞式文件操作的进程级准入控制与排序。

基于文件的异步 trait 会把自有闭包提交到这里，而不是在 Tokio 运行时线程上
直接调用 `std::fs`。一旦准入成功，操作即由内部任务持有，即使调用方被 drop
也会执行完毕。相同 key 的操作按提交顺序执行；互不相关的 key 可在进程级
上限内并发执行。
*/

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use futures::future::{BoxFuture, FutureExt, Shared, ready};
use tokio::sync::{Semaphore, oneshot};

/// 已接受但尚未结束的文件操作的最大数量。
pub const PROCESS_FILE_OPERATION_CAPACITY: usize = 64;
/// 同时占用阻塞线程的闭包的最大数量。
pub const PROCESS_FILE_OPERATION_CONCURRENCY: usize = 8;

/// 进程级准入信号量，限制在途（已接受未结束）操作数。
static ADMISSION: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(PROCESS_FILE_OPERATION_CAPACITY)));
/// 进程级执行信号量，限制同时运行的阻塞闭包数。
static EXECUTION: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(PROCESS_FILE_OPERATION_CONCURRENCY)));
/// 进程级的操作持有者运行时，保证操作不随调用方运行时关闭而中断。
static PROCESS_OWNER: LazyLock<Result<tokio::runtime::Runtime, std::io::Error>> =
    LazyLock::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("echo-file-owner")
            .build()
    });
/// 每个 key 当前队尾操作的注册表，用于实现同 key 顺序执行。
static KEY_TAILS: LazyLock<Mutex<HashMap<BlockingFileOperationKey, KeyTail>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// 可被多个等待者共享的完成信号 future。
type SharedCompletion = Shared<BoxFuture<'static, ()>>;

/// 某个 key 的队尾记录。
struct KeyTail {
    /// 队尾操作的代数（单调递增），用于判断是否可以清除该记录。
    generation: u64,
    /// 队尾操作完成时触发的共享信号。
    completion: SharedCompletion,
}

/// 阻塞式文件操作的带类型作用域。
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum BlockingFileOperationScope {
    /// 针对单条独立排序记录的操作。
    Entity(String),
    /// 针对整个存储范围的投影或扫描操作。
    Collection(String),
}

/// 基于文件的存储操作所用的无冲突排序 key。
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockingFileOperationKey {
    /// 调用方所属的命名空间（例如某个存储类型）。
    namespace: String,
    /// 规范化后的存储根目录。
    canonical_root: PathBuf,
    /// 操作的作用域。
    scope: BlockingFileOperationScope,
}

impl BlockingFileOperationKey {
    /**
    由各自独立参与哈希的组成部分构造一个 key。

    `canonical_root` 必须已是规范路径，或以其他方式保证身份稳定。
    由于本构造函数会在异步准入路径上被调用，因此不会访问文件系统。
    若为同一目录传入不同别名，将产生彼此独立的排序域。
    */
    pub fn new(
        namespace: impl Into<String>,
        canonical_root: impl Into<PathBuf>,
        scope: BlockingFileOperationScope,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            canonical_root: canonical_root.into(),
            scope,
        }
    }
}

/// 提交的文件闭包之外的基础设施故障。
#[derive(Debug, thiserror::Error)]
pub enum BlockingFileOperationError {
    /// 准入信号量已关闭。
    #[error("blocking file operation admission is closed")]
    AdmissionClosed,
    /// 执行信号量已关闭。
    #[error("blocking file operation execution is closed")]
    ExecutionClosed,
    /// 阻塞任务失败（例如闭包 panic），携带错误描述。
    #[error("blocking file operation task failed: {0}")]
    Join(String),
    /// 持有者任务在发布结果前就已结束。
    #[error("blocking file operation owner ended before publishing its result")]
    OwnerClosed,
    /// key 注册表的互斥锁已中毒。
    #[error("blocking file operation key registry is poisoned")]
    RegistryPoisoned,
    /// 当前不处于活跃的 Tokio 运行时中。
    #[error("blocking file operation requires an active Tokio runtime")]
    RuntimeUnavailable,
    /// 进程级操作持有者运行时构建失败，携带错误描述。
    #[error("process file operation owner is unavailable: {0}")]
    ProcessOwnerUnavailable(String),
    /// 某个 key 的代数计数已耗尽。
    #[error("blocking file operation key generation is exhausted")]
    KeyGenerationExhausted,
}

/**
在进程级上限约束下运行一个自有的阻塞式文件闭包。

该 future 可能需要等待准入。准入成功后、持有者任务被 spawn 之前不存在
取消点。在此之后 drop 返回的 future 只会丢弃结果接收端；闭包、key 排序
以及许可证都会一直被持有，直到操作结束。
*/
pub async fn run_keyed_file_operation<T, F>(
    key: BlockingFileOperationKey,
    operation: F,
) -> Result<T, BlockingFileOperationError>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::runtime::Handle::try_current()
        .map_err(|_| BlockingFileOperationError::RuntimeUnavailable)?;
    let owner = PROCESS_OWNER
        .as_ref()
        .map_err(|error| BlockingFileOperationError::ProcessOwnerUnavailable(error.to_string()))?;
    let admission = ADMISSION
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| BlockingFileOperationError::AdmissionClosed)?;
    let (previous, completion_tx, generation) = enqueue_key(&key)?;
    let (result_tx, result_rx) = oneshot::channel();
    let owner_key = key.clone();
    owner.spawn(async move {
        previous.await;
        let outcome = match EXECUTION.clone().acquire_owned().await {
            Ok(execution) => tokio::task::spawn_blocking(move || {
                let _execution = execution;
                operation()
            })
            .await
            .map_err(|error| BlockingFileOperationError::Join(error.to_string())),
            Err(_) => Err(BlockingFileOperationError::ExecutionClosed),
        };
        let _ignored = completion_tx.send(());
        finish_key(&owner_key, generation);
        drop(admission);
        let _ignored = result_tx.send(outcome);
    });
    result_rx
        .await
        .map_err(|_| BlockingFileOperationError::OwnerClosed)?
}

/// 将操作登记为该 key 的新队尾，返回前一个队尾的完成信号、本操作的完成发送端及代数。
fn enqueue_key(
    key: &BlockingFileOperationKey,
) -> Result<(SharedCompletion, oneshot::Sender<()>, u64), BlockingFileOperationError> {
    let mut tails = KEY_TAILS
        .lock()
        .map_err(|_| BlockingFileOperationError::RegistryPoisoned)?;
    let (previous, generation) = match tails.get(key) {
        Some(tail) => (
            tail.completion.clone(),
            tail.generation
                .checked_add(1)
                .ok_or(BlockingFileOperationError::KeyGenerationExhausted)?,
        ),
        None => (ready(()).boxed().shared(), 1),
    };
    let (completion_tx, completion_rx) = oneshot::channel();
    let completion = completion_rx.map(|_| ()).boxed().shared();
    tails.insert(
        key.clone(),
        KeyTail {
            generation,
            completion,
        },
    );
    Ok((previous, completion_tx, generation))
}

/// 操作结束后，若该 key 的队尾仍是本操作，则将其从注册表中移除。
fn finish_key(key: &BlockingFileOperationKey, generation: u64) {
    let Ok(mut tails) = KEY_TAILS.lock() else {
        return;
    };
    if tails
        .get(key)
        .is_some_and(|tail| tail.generation == generation)
    {
        tails.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    /// 串行化异步测试，避免共享的进程级信号量互相干扰。
    static ASYNC_TEST_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// 构造一个测试用的实体作用域 key。
    fn entity_key(scope: impl Into<String>) -> BlockingFileOperationKey {
        BlockingFileOperationKey::new(
            "blocking-test",
            PathBuf::from("process-owner"),
            BlockingFileOperationScope::Entity(scope.into()),
        )
    }

    #[test]
    fn missing_runtime_returns_typed_error_without_panicking() -> Result<(), String> {
        let outcome =
            futures::executor::block_on(run_keyed_file_operation(entity_key("no-runtime"), || ()));
        assert!(matches!(
            outcome,
            Err(BlockingFileOperationError::RuntimeUnavailable)
        ));
        Ok(())
    }

    #[test]
    fn exhausted_key_generation_fails_closed() -> Result<(), String> {
        let key = entity_key("generation-exhaustion");
        {
            let mut tails = KEY_TAILS.lock().map_err(|error| error.to_string())?;
            tails.insert(
                key.clone(),
                KeyTail {
                    generation: u64::MAX,
                    completion: ready(()).boxed().shared(),
                },
            );
        }
        let outcome = enqueue_key(&key);
        assert!(matches!(
            outcome,
            Err(BlockingFileOperationError::KeyGenerationExhausted)
        ));
        let mut tails = KEY_TAILS.lock().map_err(|error| error.to_string())?;
        tails.remove(&key);
        Ok(())
    }

    #[test]
    fn typed_keys_do_not_alias_delimiters_or_collection_names() {
        let entity = BlockingFileOperationKey::new(
            "conversation:store",
            PathBuf::from("root:one"),
            BlockingFileOperationScope::Entity("__list__".to_string()),
        );
        let collection = BlockingFileOperationKey::new(
            "conversation:store",
            PathBuf::from("root:one"),
            BlockingFileOperationScope::Collection("list".to_string()),
        );
        let split_differently = BlockingFileOperationKey::new(
            "conversation",
            PathBuf::from("store:root:one"),
            BlockingFileOperationScope::Entity("__list__".to_string()),
        );
        assert_ne!(entity, collection);
        assert_ne!(entity, split_differently);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_operation_does_not_stall_runtime_heartbeat() -> Result<(), String> {
        let _serial = ASYNC_TEST_SERIAL.lock().await;
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let operation = tokio::spawn(run_keyed_file_operation(
            entity_key("heartbeat"),
            move || {
                let _ignored = entered_tx.send(());
                release_rx
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|error| error.to_string())
            },
        ));
        entered_rx.await.map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_millis(250), async {
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "runtime heartbeat stalled".to_string())?;
        release_tx.send(()).map_err(|error| error.to_string())?;
        operation
            .await
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    #[tokio::test]
    async fn caller_abort_does_not_cancel_admitted_operation() -> Result<(), String> {
        let _serial = ASYNC_TEST_SERIAL.lock().await;
        let completed = Arc::new(AtomicBool::new(false));
        let completed_by_operation = Arc::clone(&completed);
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let caller = tokio::spawn(run_keyed_file_operation(entity_key("abort"), move || {
            let _ignored = entered_tx.send(());
            let _ignored = release_rx.recv_timeout(Duration::from_secs(2));
            completed_by_operation.store(true, Ordering::Release);
        }));
        entered_rx.await.map_err(|error| error.to_string())?;
        caller.abort();
        release_tx.send(()).map_err(|error| error.to_string())?;
        tokio::time::timeout(Duration::from_secs(1), async {
            while !completed.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "aborted caller cancelled owned operation".to_string())?;
        Ok(())
    }

    #[tokio::test]
    async fn same_key_is_ordered_and_different_keys_are_bounded() -> Result<(), String> {
        let _serial = ASYNC_TEST_SERIAL.lock().await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let first_finished = Arc::new(AtomicBool::new(false));
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let first_order = Arc::clone(&order);
        let first_finished_by_operation = Arc::clone(&first_finished);
        let first = tokio::spawn(run_keyed_file_operation(entity_key("ordered"), move || {
            if let Ok(mut order) = first_order.lock() {
                order.push(1);
            }
            let _ignored = release_rx.recv_timeout(Duration::from_secs(2));
            first_finished_by_operation.store(true, Ordering::Release);
        }));
        tokio::task::yield_now().await;
        let second_order = Arc::clone(&order);
        let first_finished_for_second = Arc::clone(&first_finished);
        let ordering_violation = Arc::new(AtomicBool::new(false));
        let ordering_violation_by_operation = Arc::clone(&ordering_violation);
        let second = tokio::spawn(run_keyed_file_operation(entity_key("ordered"), move || {
            if !first_finished_for_second.load(Ordering::Acquire) {
                ordering_violation_by_operation.store(true, Ordering::Release);
            }
            if let Ok(mut order) = second_order.lock() {
                order.push(2);
            }
        }));
        tokio::task::yield_now().await;
        release_tx.send(()).map_err(|error| error.to_string())?;
        first
            .await
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        second
            .await
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        assert_eq!(
            order.lock().map_err(|error| error.to_string())?.as_slice(),
            &[1, 2]
        );
        assert!(!ordering_violation.load(Ordering::Acquire));

        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let release = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let mut operations = Vec::new();
        for index in 0..PROCESS_FILE_OPERATION_CONCURRENCY.saturating_mul(2) {
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            let release = Arc::clone(&release);
            operations.push(tokio::spawn(run_keyed_file_operation(
                entity_key(format!("parallel-{index}")),
                move || {
                    let current = active.fetch_add(1, Ordering::AcqRel).saturating_add(1);
                    maximum.fetch_max(current, Ordering::AcqRel);
                    let (released, notification) = &*release;
                    if let Ok(mut released) = released.lock() {
                        while !*released {
                            match notification.wait(released) {
                                Ok(next) => released = next,
                                Err(_) => break,
                            }
                        }
                    }
                    active.fetch_sub(1, Ordering::AcqRel);
                },
            )));
        }
        tokio::time::timeout(Duration::from_secs(2), async {
            while maximum.load(Ordering::Acquire) < PROCESS_FILE_OPERATION_CONCURRENCY {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| "blocking operation concurrency did not fill its bound".to_string())?;
        let (released, notification) = &*release;
        {
            let mut released = released.lock().map_err(|error| error.to_string())?;
            *released = true;
            notification.notify_all();
        }
        for operation in operations {
            operation
                .await
                .map_err(|error| error.to_string())?
                .map_err(|error| error.to_string())?;
        }
        let observed = maximum.load(Ordering::Acquire);
        assert_eq!(observed, PROCESS_FILE_OPERATION_CONCURRENCY);
        Ok(())
    }

    #[test]
    fn caller_runtime_shutdown_does_not_release_same_key_tail() -> Result<(), String> {
        let _serial = ASYNC_TEST_SERIAL.blocking_lock();
        let key = entity_key("runtime-shutdown");
        let first_finished = Arc::new(AtomicBool::new(false));
        let first_finished_by_operation = Arc::clone(&first_finished);
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let runtime_a = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|error| error.to_string())?;
        let first_key = key.clone();
        let _first = runtime_a.spawn(run_keyed_file_operation(first_key, move || {
            let _ignored = entered_tx.send(());
            let _ignored = release_rx.recv_timeout(Duration::from_secs(2));
            first_finished_by_operation.store(true, Ordering::Release);
        }));
        runtime_a
            .block_on(entered_rx)
            .map_err(|error| error.to_string())?;
        drop(runtime_a);

        let runtime_b = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|error| error.to_string())?;
        let second_started = Arc::new(AtomicBool::new(false));
        let second_started_by_operation = Arc::clone(&second_started);
        let ordering_violation = Arc::new(AtomicBool::new(false));
        let ordering_violation_by_operation = Arc::clone(&ordering_violation);
        let second_key = key.clone();
        let second = runtime_b.spawn(run_keyed_file_operation(second_key, move || {
            if !first_finished.load(Ordering::Acquire) {
                ordering_violation_by_operation.store(true, Ordering::Release);
            }
            second_started_by_operation.store(true, Ordering::Release);
        }));
        runtime_b.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let generation = KEY_TAILS
                        .lock()
                        .map_err(|error| error.to_string())?
                        .get(&key)
                        .map(|tail| tail.generation);
                    if generation == Some(2) {
                        return Ok::<(), String>(());
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .map_err(|_| "second runtime did not enqueue its operation".to_string())??;
            assert!(!second_started.load(Ordering::Acquire));
            release_tx.send(()).map_err(|error| error.to_string())?;
            second
                .await
                .map_err(|error| error.to_string())?
                .map_err(|error| error.to_string())?;
            Ok::<(), String>(())
        })?;
        assert!(!ordering_violation.load(Ordering::Acquire));
        assert!(
            !KEY_TAILS
                .lock()
                .map_err(|error| error.to_string())?
                .contains_key(&key)
        );
        Ok(())
    }

    #[tokio::test]
    async fn panicking_operation_releases_key_and_permits() -> Result<(), String> {
        let _serial = ASYNC_TEST_SERIAL.lock().await;
        let key = entity_key("panic-release");
        let failed = run_keyed_file_operation(key.clone(), || -> () {
            // 故障注入：验证持有者能够正确收尾一个发生 panic 的阻塞闭包。
            std::panic::resume_unwind(Box::new("injected file operation panic"));
        })
        .await;
        assert!(matches!(failed, Err(BlockingFileOperationError::Join(_))));
        run_keyed_file_operation(key.clone(), || ())
            .await
            .map_err(|error| error.to_string())?;
        assert!(
            !KEY_TAILS
                .lock()
                .map_err(|error| error.to_string())?
                .contains_key(&key)
        );
        Ok(())
    }
}
