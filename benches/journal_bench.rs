/*! 序列化事件日志（journal）核心原语的 Criterion 基准测试。

运行方式：`cargo bench -p echo_agent --bench journal_bench`

这些数据支撑 Task-5 在原语层面的性能门槛：追加吞吐量、全量回放吞吐量，
以及检查点复合（checkpoint-compounded）恢复比率（`CheckpointedReducer`
存在的原因，就是让 10 万量级的恢复只重放尾部，而不是重放整个日志）。
*/

use criterion::{BenchmarkId, Criterion};
use echo_agent::state::journal::{
    CheckpointStore, CheckpointedReducer, EventJournal, EventReducer, FileCheckpointStore,
    FileEventJournal, MemoryCheckpointStore, MemoryEventJournal, PreparedJournalBatch,
    SegmentedFileEventJournal,
};
use echo_agent::utils::fs::FileDurability;
use std::fmt;
use std::sync::{Arc, Mutex};

/// 封装基准测试过程中出现的失败信息的错误类型。
#[derive(Clone, Debug)]
struct BenchFailure(String);

impl fmt::Display for BenchFailure {
    /// 将失败信息原样写出。
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BenchFailure {}

/// 基准测试专用的 `Result` 别名，错误类型固定为 `BenchFailure`。
type BenchResult<T> = std::result::Result<T, BenchFailure>;

/// 为 `Result` 附加上下文信息，便于基准测试失败时定位问题。
trait BenchContext<T> {
    /// 用给定的上下文描述包装错误，转换为 `BenchResult`。
    fn bench_context(self, context: &str) -> BenchResult<T>;
}

impl<T, E: fmt::Debug> BenchContext<T> for std::result::Result<T, E> {
    /// 将任意 `Result` 的错误格式化为带上下文的 `BenchFailure`。
    fn bench_context(self, context: &str) -> BenchResult<T> {
        match self {
            Ok(value) => Ok(value),
            Err(error) => Err(BenchFailure(format!("{context}: {error:?}"))),
        }
    }
}

/// 记录基准测试迭代过程中出现的第一个失败，供收尾时统一上报。
#[derive(Clone, Default)]
struct BenchFailures {
    /// 记录到的第一个失败（如果有的话）。
    first: Arc<Mutex<Option<BenchFailure>>>,
}

impl BenchFailures {
    /// 记录一次迭代的结果；若已存在失败则保留最早的那一个。
    fn record<T>(&self, result: BenchResult<T>) -> BenchResult<T> {
        if let Err(error) = &result {
            let mut first = match self.first.lock() {
                Ok(first) => first,
                Err(poisoned) => poisoned.into_inner(),
            };
            if first.is_none() {
                *first = Some(error.clone());
            }
        }
        result
    }

    /// 取出并返回记录到的第一个失败；没有失败则返回 `Ok(())`。
    fn finish(&self) -> BenchResult<()> {
        let mut first = match self.first.lock() {
            Ok(first) => first,
            Err(poisoned) => poisoned.into_inner(),
        };
        match first.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// 将计数事件折叠为一个小状态的归约器（reducer）。
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct BenchReducer {
    /// 已应用的事件数量。
    applied: u64,
    /// 所有事件值的累加校验和。
    checksum: u64,
}

impl EventReducer for BenchReducer {
    type Event = u64;

    /// 应用一个事件：累加计数并更新校验和。
    fn apply(&mut self, event: &u64) {
        self.applied = self.applied.saturating_add(1);
        self.checksum = self.checksum.wrapping_add(*event);
    }
}

/// 生成 `total` 个事件，并按 `batch_size` 切分为若干预备好的写入批次。
fn prepared_batches(
    total: usize,
    batch_size: usize,
) -> BenchResult<Vec<PreparedJournalBatch<u64>>> {
    if batch_size == 0 {
        return Err(BenchFailure(
            "benchmark batch size must be positive".to_string(),
        ));
    }
    (0..total)
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .map(|chunk| {
            let events = chunk
                .iter()
                .map(|value| u64::try_from(*value).unwrap_or(u64::MAX))
                .collect::<Vec<_>>();
            PreparedJournalBatch::new(events).bench_context("prepare benchmark batch")
        })
        .collect()
}

/// 依次将若干批次追加到日志中，返回追加完成后的最新序列号。
fn append_batches<J: EventJournal<u64>>(
    journal: &J,
    batches: Vec<PreparedJournalBatch<u64>>,
    context: &str,
) -> BenchResult<u64> {
    for batch in batches {
        journal.append_batch(batch).bench_context(context)?;
    }
    Ok(journal.last_sequence())
}

/// 构建一个内存日志，并按给定的检查点节奏写入 `size` 个事件。
fn build_memory_journal(
    size: u64,
    checkpoint_cadence: u64,
) -> BenchResult<Arc<MemoryEventJournal<u64>>> {
    let journal = Arc::new(MemoryEventJournal::<u64>::new());
    let writer = CheckpointedReducer::<_, BenchReducer>::new(
        Arc::clone(&journal),
        Arc::new(MemoryCheckpointStore::new()),
        checkpoint_cadence,
    );
    for value in 0..size {
        writer.apply(value).bench_context("build memory journal")?;
    }
    Ok(journal)
}

/// 注册并运行本文件中的全部日志基准测试用例。
fn try_bench_journal(c: &mut Criterion) -> BenchResult<()> {
    const RECOVERY_EVENTS: u64 = 105_321;
    const CHECKPOINT_CADENCE: u64 = 10_000;
    const BENCH_EVENTS: usize = 128;
    let failures = BenchFailures::default();
    let mut group = c.benchmark_group("journal");

    for batch_size in [1_usize, 2, 8, 32, 128] {
        let case_failures = failures.clone();
        group.bench_with_input(
            BenchmarkId::new("memory_append_total_128", batch_size),
            &batch_size,
            move |b, size| {
                let iteration_failures = case_failures.clone();
                b.iter_batched(
                    || -> BenchResult<_> {
                        Ok((
                            MemoryEventJournal::new(),
                            prepared_batches(BENCH_EVENTS, *size)?,
                        ))
                    },
                    move |setup| {
                        let result = (|| -> BenchResult<u64> {
                            let (journal, batches) = setup?;
                            append_batches(&journal, batches, "memory batch")
                        })();
                        iteration_failures.record(result)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
        let case_failures = failures.clone();
        group.bench_with_input(
            BenchmarkId::new("file_append_total_128_flush", batch_size),
            &batch_size,
            move |b, size| {
                let iteration_failures = case_failures.clone();
                b.iter_batched(
                    || -> BenchResult<_> {
                        let directory = tempfile::tempdir().bench_context("batch tempdir")?;
                        let journal = FileEventJournal::open(
                            directory.path().join("events.jsonl"),
                            FileDurability::Flush,
                        )
                        .bench_context("open file batch journal")?;
                        let batches = prepared_batches(BENCH_EVENTS, *size)?;
                        Ok((directory, journal, batches))
                    },
                    move |setup| {
                        let result = (|| -> BenchResult<u64> {
                            let (_directory, journal, batches) = setup?;
                            append_batches(&journal, batches, "file batch")
                        })();
                        iteration_failures.record(result)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
        let case_failures = failures.clone();
        group.bench_with_input(
            BenchmarkId::new("segmented_append_total_128_flush", batch_size),
            &batch_size,
            move |b, size| {
                let iteration_failures = case_failures.clone();
                b.iter_batched(
                    || -> BenchResult<_> {
                        let directory =
                            tempfile::tempdir().bench_context("segmented batch tempdir")?;
                        let journal = SegmentedFileEventJournal::open(
                            directory.path().join("segments"),
                            64 * 1024,
                            FileDurability::Flush,
                        )
                        .bench_context("open segmented batch journal")?;
                        let batches = prepared_batches(BENCH_EVENTS, *size)?;
                        Ok((directory, journal, batches))
                    },
                    move |setup| {
                        let result = (|| -> BenchResult<u64> {
                            let (_directory, journal, batches) = setup?;
                            append_batches(&journal, batches, "segmented batch")
                        })();
                        iteration_failures.record(result)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    let case_failures = failures.clone();
    group.bench_function("memory_replay_after_zero_10k", |b| {
        let iteration_failures = case_failures.clone();
        b.iter_batched(
            || -> BenchResult<_> {
                let journal = MemoryEventJournal::<u64>::new();
                for value in 0..10_000u64 {
                    journal
                        .append(value)
                        .bench_context("seed memory replay journal")?;
                }
                Ok(journal)
            },
            move |setup| {
                let result = (|| -> BenchResult<usize> {
                    let journal = setup?;
                    Ok(journal
                        .replay_after(0, usize::MAX)
                        .bench_context("memory replay")?
                        .len())
                })();
                iteration_failures.record(result)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    let case_failures = failures.clone();
    group.bench_function("file_replay_after_zero_10k", |b| {
        let iteration_failures = case_failures.clone();
        b.iter_batched(
            || -> BenchResult<_> {
                let dir = tempfile::tempdir().bench_context("file replay tempdir")?;
                let journal = FileEventJournal::<u64>::open(
                    dir.path().join("events.jsonl"),
                    FileDurability::Flush,
                )
                .bench_context("open file replay journal")?;
                for value in 0..10_000u64 {
                    journal
                        .append(value)
                        .bench_context("seed file replay journal")?;
                }
                Ok((dir, journal))
            },
            move |setup| {
                let result = (|| -> BenchResult<usize> {
                    let (_dir, journal) = setup?;
                    Ok(journal
                        .replay_after(0, usize::MAX)
                        .bench_context("file replay")?
                        .len())
                })();
                iteration_failures.record(result)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    let case_failures = failures.clone();
    group.bench_function("segmented_replay_after_zero_10k", |b| {
        let iteration_failures = case_failures.clone();
        b.iter_batched(
            || -> BenchResult<_> {
                let dir = tempfile::tempdir().bench_context("segmented replay tempdir")?;
                let journal = SegmentedFileEventJournal::<u64>::open(
                    dir.path().join("segments"),
                    64 * 1024,
                    FileDurability::Flush,
                )
                .bench_context("open segmented replay journal")?;
                for value in 0..10_000u64 {
                    journal
                        .append(value)
                        .bench_context("seed segmented replay journal")?;
                }
                Ok((dir, journal))
            },
            move |setup| {
                let result = (|| -> BenchResult<usize> {
                    let (_dir, journal) = setup?;
                    Ok(journal
                        .replay_after(0, usize::MAX)
                        .bench_context("segmented replay")?
                        .len())
                })();
                iteration_failures.record(result)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    let case_failures = failures.clone();
    group.bench_function("recover_full_replay_105k", |b| {
        let iteration_failures = case_failures.clone();
        b.iter_batched(
            || build_memory_journal(RECOVERY_EVENTS, CHECKPOINT_CADENCE),
            move |setup| {
                let result = (|| -> BenchResult<u64> {
                    let journal = setup?;
                    let reducer = CheckpointedReducer::<_, BenchReducer>::new(
                        journal,
                        Arc::new(MemoryCheckpointStore::new()),
                        CHECKPOINT_CADENCE,
                    );
                    Ok(reducer
                        .recover()
                        .bench_context("recover full memory journal")?
                        .last_applied_sequence)
                })();
                iteration_failures.record(result)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    let case_failures = failures.clone();
    group.bench_function("recover_compounded_105k_tail_5321", |b| {
        let iteration_failures = case_failures.clone();
        b.iter_batched(
            || -> BenchResult<_> {
                /* 真实场景路径：写入方每 10k 个事件压缩一次检查点；
                恢复时只加载最新的快照帧，并只重放其后的尾部事件。 */
                let journal = Arc::new(MemoryEventJournal::<u64>::new());
                let checkpoints: Arc<dyn CheckpointStore<BenchReducer>> =
                    Arc::new(MemoryCheckpointStore::new());
                let writer = CheckpointedReducer::<_, BenchReducer>::new(
                    Arc::clone(&journal),
                    Arc::clone(&checkpoints),
                    CHECKPOINT_CADENCE,
                );
                for value in 0..RECOVERY_EVENTS {
                    writer
                        .apply(value)
                        .bench_context("build compounded memory journal")?;
                }
                Ok((journal, checkpoints))
            },
            move |setup| {
                let result = (|| -> BenchResult<u64> {
                    let (journal, checkpoints) = setup?;
                    let reducer = CheckpointedReducer::<_, BenchReducer>::new(
                        journal,
                        checkpoints,
                        CHECKPOINT_CADENCE,
                    );
                    let recovered = reducer
                        .recover()
                        .bench_context("recover compounded memory journal")?
                        .last_applied_sequence;
                    let applied = reducer.with_state(|state| state.applied);
                    if applied != recovered {
                        return Err(BenchFailure(format!(
                            "compounded recovery applied {applied} events through sequence {recovered}"
                        )));
                    }
                    Ok(recovered)
                })();
                iteration_failures.record(result)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    /* 持久化恢复是生产环境中真正相关的性能门槛。FileEventJournal 会在
    经过校验的打开过程中记录字节偏移量，从而直接跳转到检查点之后的
    尾部数据，而不必扫描/反序列化完整的 JSONL 文件。 */
    let persistent_dir = tempfile::tempdir().bench_context("persistent tempdir")?;
    let persistent_journal = Arc::new(
        FileEventJournal::<u64>::open(
            persistent_dir.path().join("events.jsonl"),
            FileDurability::Flush,
        )
        .bench_context("open persistent journal")?,
    );
    let persistent_checkpoints: Arc<dyn CheckpointStore<BenchReducer>> = Arc::new(
        FileCheckpointStore::open(persistent_dir.path().join("checkpoint.json")),
    );
    let persistent_writer = CheckpointedReducer::<_, BenchReducer>::new(
        Arc::clone(&persistent_journal),
        Arc::clone(&persistent_checkpoints),
        CHECKPOINT_CADENCE,
    );
    for value in 0..RECOVERY_EVENTS {
        persistent_writer
            .apply(value)
            .bench_context("build persistent journal")?;
    }
    let case_failures = failures.clone();
    group.bench_function("file_recover_compounded_105k_tail_5321", |b| {
        let iteration_failures = case_failures.clone();
        b.iter(|| {
            let reducer = CheckpointedReducer::<_, BenchReducer>::new(
                Arc::clone(&persistent_journal),
                Arc::clone(&persistent_checkpoints),
                CHECKPOINT_CADENCE,
            );
            iteration_failures.record(
                reducer
                    .recover()
                    .bench_context("persistent recover")
                    .map(|receipt| receipt.last_applied_sequence),
            )
        })
    });

    group.finish();
    failures.finish()
}

/// 基准测试入口：运行全部日志用例，并在完成后汇报首个失败（如有）。
fn main() -> BenchResult<()> {
    let mut criterion = Criterion::default().configure_from_args();
    try_bench_journal(&mut criterion)?;
    criterion.final_summary();
    Ok(())
}
