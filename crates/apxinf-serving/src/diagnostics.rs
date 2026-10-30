//! Bounded, best-effort diagnostics outside the service lifecycle.
use std::{
    fmt::{self, Write as _},
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, OnceLock,
    },
    time::Duration,
};
use tokio::{io::AsyncReadExt, process::ChildStderr, task::JoinHandle};

const QUEUE_CAPACITY: usize = 256;
const RECORD_LIMIT: usize = 4096;
static DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();

#[derive(Default)]
struct Counters {
    written: AtomicU64,
    full: AtomicU64,
    oversized: AtomicU64,
    unavailable: AtomicU64,
    write_errors: AtomicU64,
    worker_read_errors: AtomicU64,
    pending: AtomicU64,
    available: AtomicBool,
}

enum Disposition {
    Written,
    Full,
    Unavailable,
}

struct Record {
    text: String,
    counters: Arc<Counters>,
    disposition: Disposition,
}

impl Drop for Record {
    fn drop(&mut self) {
        let counter = match self.disposition {
            Disposition::Written => &self.counters.written,
            Disposition::Full => &self.counters.full,
            Disposition::Unavailable => &self.counters.unavailable,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.counters.pending.fetch_sub(1, Ordering::Release);
    }
}

struct BoundedLine(String);

impl fmt::Write for BoundedLine {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if text.len() > RECORD_LIMIT - 1 - self.0.len() {
            return Err(fmt::Error);
        }
        self.0.push_str(text);
        Ok(())
    }
}

struct WriterState(Arc<Counters>);

impl Drop for WriterState {
    fn drop(&mut self) {
        self.0.available.store(false, Ordering::Release);
    }
}

struct Diagnostics {
    sender: Option<SyncSender<Record>>,
    counters: Arc<Counters>,
}

impl Diagnostics {
    fn with_writer<W: Write + Send + 'static>(writer: W, capacity: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Record>(capacity);
        let counters = Arc::new(Counters::default());
        counters.available.store(true, Ordering::Release);
        let thread_counters = counters.clone();
        let started = std::thread::Builder::new()
            .name("apxinf-diagnostics".into())
            .spawn(move || {
                let _state = WriterState(thread_counters.clone());
                let mut writer = writer;
                for mut record in receiver {
                    if writer.write_all(record.text.as_bytes()).is_err() {
                        thread_counters.write_errors.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    record.disposition = Disposition::Written;
                }
            });
        // Dropping the handle detaches the thread. A blocked sink cannot hold shutdown.
        if started.is_err() {
            counters.available.store(false, Ordering::Release);
        }
        Self {
            sender: started.ok().map(|_| sender),
            counters,
        }
    }

    fn emit(&self, arguments: fmt::Arguments<'_>) {
        let mut line = BoundedLine(String::with_capacity(RECORD_LIMIT));
        if line.write_fmt(arguments).is_err() {
            self.counters.oversized.fetch_add(1, Ordering::Relaxed);
            return;
        }
        line.0.push('\n');
        self.counters.pending.fetch_add(1, Ordering::Relaxed);
        let record = Record {
            text: line.0,
            counters: self.counters.clone(),
            disposition: Disposition::Unavailable,
        };
        let Some(sender) = &self.sender else {
            return;
        };
        match sender.try_send(record) {
            Ok(()) => {}
            Err(TrySendError::Full(mut record)) => record.disposition = Disposition::Full,
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    fn append_metrics(&self, output: &mut String) {
        let c = &self.counters;
        let get = |value: &AtomicU64| value.load(Ordering::Relaxed);
        let _ = writeln!(
            output,
            "apxinf_diagnostic_written_total {}",
            get(&c.written)
        );
        for (reason, count) in [
            ("full", get(&c.full)),
            ("oversized", get(&c.oversized)),
            ("unavailable", get(&c.unavailable)),
        ] {
            let _ = writeln!(
                output,
                "apxinf_diagnostic_dropped_total{{reason=\"{reason}\"}} {count}"
            );
        }
        let _ = writeln!(
            output,
            "apxinf_diagnostic_write_errors_total {}",
            get(&c.write_errors)
        );
        let _ = writeln!(
            output,
            "apxinf_diagnostic_worker_read_errors_total {}",
            get(&c.worker_read_errors)
        );
        let _ = writeln!(
            output,
            "apxinf_diagnostic_writer_available {}",
            u8::from(c.available.load(Ordering::Acquire))
        );
    }
}

fn global() -> &'static Diagnostics {
    DIAGNOSTICS.get_or_init(|| Diagnostics::with_writer(io::stderr(), QUEUE_CAPACITY))
}

/// Create the diagnostic thread before starting a worker process.
pub fn initialize() -> Result<(), &'static str> {
    if global().sender.is_some() {
        Ok(())
    } else {
        Err("Cannot start the diagnostic writer thread.")
    }
}

/// Discard the record if its size or queue capacity exceeds the fixed limit.
pub fn emit(arguments: fmt::Arguments<'_>) {
    global().emit(arguments);
}

pub(crate) fn append_metrics(output: &mut String) {
    global().append_metrics(output);
}

/// Allow a bounded delivery interval after service shutdown, without joining the writer.
pub async fn finish() {
    let Some(logger) = DIAGNOSTICS.get() else {
        return;
    };
    let _ = tokio::time::timeout(Duration::from_millis(100), async {
        while logger.counters.available.load(Ordering::Acquire)
            && logger.counters.pending.load(Ordering::Acquire) != 0
        {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await;
}

pub(crate) struct WorkerDiagnostics {
    task: JoinHandle<()>,
}

impl WorkerDiagnostics {
    pub(crate) fn start(mut stderr: ChildStderr, epoch: String) -> Self {
        let task = tokio::spawn(async move {
            let mut bytes = [0_u8; 512];
            loop {
                match stderr.read(&mut bytes).await {
                    Ok(0) => break,
                    Ok(count) => emit(format_args!(
                        "{}",
                        serde_json::json!({
                            "event": "worker_stderr", "worker_epoch": epoch,
                            "text": String::from_utf8_lossy(&bytes[..count]),
                        })
                    )),
                    Err(_) => {
                        global()
                            .counters
                            .worker_read_errors
                            .fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
                tokio::task::yield_now().await;
            }
        });
        Self { task }
    }
}

impl Drop for WorkerDiagnostics {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Condvar, Mutex};

    struct Gate {
        open: Mutex<bool>,
        changed: Condvar,
        bytes: Mutex<Vec<u8>>,
    }

    impl Gate {
        fn release(&self) {
            *self.open.lock().unwrap() = true;
            self.changed.notify_all();
        }
    }

    struct ReleaseOnDrop(Arc<Gate>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    struct GatedWriter {
        gate: Arc<Gate>,
        entered: Option<mpsc::Sender<()>>,
        fail: bool,
    }

    impl Write for GatedWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if let Some(entered) = self.entered.take() {
                let _ = entered.send(());
            }
            let mut open = self.gate.open.lock().unwrap();
            while !*open {
                open = self.gate.changed.wait(open).unwrap();
            }
            if self.fail {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "The sink closed.",
                ));
            }
            self.gate.bytes.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn logger(fail: bool) -> (Diagnostics, ReleaseOnDrop, mpsc::Receiver<()>) {
        let gate = Arc::new(Gate {
            open: Mutex::new(false),
            changed: Condvar::new(),
            bytes: Mutex::new(Vec::new()),
        });
        let (entered, receiver) = mpsc::channel();
        let logger = Diagnostics::with_writer(
            GatedWriter {
                gate: gate.clone(),
                entered: Some(entered),
                fail,
            },
            2,
        );
        (logger, ReleaseOnDrop(gate), receiver)
    }

    fn wait_empty(logger: &Diagnostics) {
        let started = std::time::Instant::now();
        while logger.counters.pending.load(Ordering::Acquire) != 0 {
            assert!(started.elapsed() < Duration::from_secs(2));
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn blocked_writer_bounds_the_queue_and_preserves_record_order() {
        let (logger, gate, entered) = logger(false);
        logger.emit(format_args!("first"));
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        logger.emit(format_args!("second"));
        logger.emit(format_args!("third"));
        for _ in 0..1000 {
            logger.emit(format_args!("discarded"));
        }
        assert_eq!(logger.counters.pending.load(Ordering::Relaxed), 3);
        assert_eq!(logger.counters.full.load(Ordering::Relaxed), 1000);
        gate.0.release();
        wait_empty(&logger);
        assert_eq!(logger.counters.written.load(Ordering::Relaxed), 3);
        assert_eq!(&*gate.0.bytes.lock().unwrap(), b"first\nsecond\nthird\n");
    }

    #[test]
    fn failed_writer_accounts_for_current_queued_and_later_records() {
        let (logger, gate, entered) = logger(true);
        logger.emit(format_args!("first"));
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        logger.emit(format_args!("second"));
        logger.emit(format_args!("third"));
        gate.0.release();
        wait_empty(&logger);
        logger.emit(format_args!("after failure"));
        wait_empty(&logger);
        let mut metrics = String::new();
        logger.append_metrics(&mut metrics);
        assert!(metrics.contains("apxinf_diagnostic_write_errors_total 1\n"));
        assert!(metrics.contains("apxinf_diagnostic_dropped_total{reason=\"unavailable\"} 4\n"));
        assert!(metrics.contains("apxinf_diagnostic_writer_available 0\n"));
        assert!(gate.0.bytes.lock().unwrap().is_empty());
    }

    #[test]
    fn byte_limit_keeps_complete_records_and_discards_oversized_utf8() {
        let (logger, gate, entered) = logger(false);
        let exact = "字".repeat(1365);
        logger.emit(format_args!("{exact}"));
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        logger.emit(format_args!("{exact}x"));
        logger.emit(format_args!("prefix {}", "x".repeat(RECORD_LIMIT)));
        assert_eq!(logger.counters.oversized.load(Ordering::Relaxed), 2);
        gate.0.release();
        wait_empty(&logger);
        let bytes = gate.0.bytes.lock().unwrap();
        assert_eq!(bytes.len(), RECORD_LIMIT);
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), format!("{exact}\n"));
    }
}
