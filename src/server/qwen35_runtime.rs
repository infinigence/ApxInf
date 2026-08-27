//! Incremental, single-owner transport for a checkpoint-backed Qwen3.5 executor.
//!
//! This module does not implement model math or supply fallback tokens. The
//! production CUDA executor must implement [`Qwen35StepExecutor`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use apxinf_model::runtime::{RequestPermit, RuntimeAdmission};
use apxinf_model::{RuntimeCapabilities, RuntimeError, RuntimeRequest};

use super::service::{ProtocolRuntime, TokenStream};

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
use apxinf_model::qwen35::{Qwen35CudaModel, Qwen35CudaSession};

pub trait Qwen35StepSession: Send {
    /// Advance the request by at most one generated token.
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError>;
}

pub trait Qwen35StepExecutor: Send + Sync + 'static {
    /// Create request-local CUDA state after admission has succeeded.
    fn open(&self, request: RuntimeRequest) -> Result<Box<dyn Qwen35StepSession>, RuntimeError>;
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
pub struct Qwen35CudaStepExecutor {
    model: std::sync::Arc<Qwen35CudaModel>,
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
impl Qwen35CudaStepExecutor {
    pub fn new(model: std::sync::Arc<Qwen35CudaModel>) -> Self {
        Self { model }
    }
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
struct Qwen35CudaStepSession {
    session: Qwen35CudaSession,
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
impl Qwen35StepSession for Qwen35CudaStepSession {
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
        self.session.next_token().map_err(RuntimeError::Execution)
    }
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
impl Qwen35StepExecutor for Qwen35CudaStepExecutor {
    fn open(&self, request: RuntimeRequest) -> Result<Box<dyn Qwen35StepSession>, RuntimeError> {
        self.model
            .open_with_cancel_multimodal(
                &request.input_ids,
                request.max_new_tokens,
                &request.cancel,
                request.multimodal.as_ref(),
            )
            .map(|session| {
                Box::new(Qwen35CudaStepSession { session }) as Box<dyn Qwen35StepSession>
            })
            .map_err(|error| {
                // A client disconnect aborts prefill mid-open; that is a
                // request-level cancellation, not a fatal executor failure,
                // so it must not poison service readiness.
                if request.cancel.is_cancelled() {
                    RuntimeError::Cancelled
                } else {
                    RuntimeError::Execution(error)
                }
            })
    }
}

enum SessionCommand {
    Next(mpsc::Sender<Result<Option<u32>, RuntimeError>>),
    Cancel(mpsc::Sender<()>),
}

struct WorkerJob {
    request: RuntimeRequest,
    permit: RequestPermit,
    response: mpsc::Sender<Result<OpenedSession, RuntimeError>>,
}

struct OpenedSession {
    commands: mpsc::Sender<SessionCommand>,
}

pub struct Qwen35ProtocolRuntime {
    capabilities: RuntimeCapabilities,
    admission: Arc<RuntimeAdmission>,
    sender: mpsc::SyncSender<WorkerJob>,
    stopped: Arc<AtomicBool>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Qwen35ProtocolRuntime {
    pub fn new(
        capabilities: RuntimeCapabilities,
        queue_capacity: usize,
        executor: Arc<dyn Qwen35StepExecutor>,
    ) -> Result<Self, RuntimeError> {
        if queue_capacity == 0 {
            return Err(RuntimeError::QueueFull);
        }
        let (sender, receiver) = mpsc::sync_channel(queue_capacity);
        let admission = Arc::new(RuntimeAdmission::new(capabilities));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = Arc::clone(&stopped);
        let join = thread::Builder::new()
            .name("apxinf-qwen35-worker".into())
            .spawn(move || worker_loop(receiver, executor, worker_stopped))
            .map_err(|error| RuntimeError::Execution(error.to_string()))?;
        Ok(Self {
            capabilities,
            admission,
            sender,
            stopped,
            join: Mutex::new(Some(join)),
        })
    }

    pub fn active_requests(&self) -> usize {
        self.admission.active_requests()
    }
}

impl ProtocolRuntime for Qwen35ProtocolRuntime {
    fn capabilities(&self) -> RuntimeCapabilities {
        self.capabilities
    }

    fn start(&self, request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(RuntimeError::WorkerStopped);
        }
        let cancel = request.cancel.clone();
        let (_, permit) = self.admission.admit(&request)?;
        let (response, opened) = mpsc::channel();
        let job = WorkerJob {
            request,
            permit,
            response,
        };
        match self.sender.try_send(job) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => return Err(RuntimeError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(RuntimeError::WorkerStopped),
        }
        let opened = opened.recv().unwrap_or(Err(RuntimeError::WorkerStopped))?;
        Ok(Box::new(Qwen35ProtocolTokenStream {
            commands: opened.commands,
            request_cancel: cancel,
            finished: AtomicBool::new(false),
        }))
    }

    fn warmup(&self) -> Result<(), RuntimeError> {
        let mut stream = self.start(RuntimeRequest::new(vec![1], 2))?;
        stream.next_token()?;
        stream
            .next_token()?
            .map(|_| ())
            .ok_or_else(|| RuntimeError::Execution("warmup produced no decode token".into()))
    }
}

impl Drop for Qwen35ProtocolRuntime {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        let (replacement, receiver) = mpsc::sync_channel(1);
        let old = std::mem::replace(&mut self.sender, replacement);
        drop(old);
        drop(receiver);
        if let Ok(mut join) = self.join.lock() {
            if let Some(join) = join.take() {
                let _ = join.join();
            }
        }
    }
}

fn worker_loop(
    receiver: mpsc::Receiver<WorkerJob>,
    executor: Arc<dyn Qwen35StepExecutor>,
    stopped: Arc<AtomicBool>,
) {
    while !stopped.load(Ordering::Acquire) {
        let job = match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(job) => job,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if job.request.cancel.is_cancelled() {
            let _ = job.response.send(Err(RuntimeError::Cancelled));
            continue;
        }
        let request_cancel = job.request.cancel.clone();
        let session = match executor.open(job.request) {
            Ok(session) => session,
            Err(error) => {
                let _ = job.response.send(Err(error));
                continue;
            }
        };
        let (commands, session_commands) = mpsc::channel();
        if job.response.send(Ok(OpenedSession { commands })).is_err() {
            continue;
        }
        run_session(
            session,
            session_commands,
            &stopped,
            request_cancel,
            job.permit,
        );
    }
}

fn run_session(
    mut session: Box<dyn Qwen35StepSession>,
    commands: mpsc::Receiver<SessionCommand>,
    stopped: &AtomicBool,
    request_cancel: apxinf_model::CancellationToken,
    _permit: RequestPermit,
) {
    while !stopped.load(Ordering::Acquire) {
        if request_cancel.is_cancelled() {
            return;
        }
        match commands.recv_timeout(Duration::from_millis(10)) {
            Ok(SessionCommand::Next(response)) => {
                let result = session.next_token();
                let terminal = matches!(result, Ok(None) | Err(_));
                if terminal {
                    drop(session);
                    let _ = response.send(result);
                    return;
                }
                let _ = response.send(result);
            }
            Ok(SessionCommand::Cancel(response)) => {
                drop(session);
                let _ = response.send(());
                return;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

struct Qwen35ProtocolTokenStream {
    commands: mpsc::Sender<SessionCommand>,
    request_cancel: apxinf_model::CancellationToken,
    finished: AtomicBool,
}

impl Qwen35ProtocolTokenStream {
    fn cancel_worker(&self) {
        let (response, acknowledgement) = mpsc::channel();
        if self.commands.send(SessionCommand::Cancel(response)).is_ok() {
            let _ = acknowledgement.recv();
        }
    }
}

impl TokenStream for Qwen35ProtocolTokenStream {
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
        if self.finished.load(Ordering::Acquire) {
            return Ok(None);
        }
        if self.request_cancel.is_cancelled() {
            self.finished.store(true, Ordering::Release);
            self.cancel_worker();
            return Err(RuntimeError::Cancelled);
        }
        // The session worker exits as soon as it observes cancellation, which
        // can race with a `Next` command already in flight; a disconnected
        // channel on a cancelled request is a cancellation, not a stopped
        // worker, and must not be reported as a fatal error.
        let disconnected = |cancelled: bool| {
            if cancelled {
                RuntimeError::Cancelled
            } else {
                RuntimeError::WorkerStopped
            }
        };
        let (response, result) = mpsc::channel();
        self.commands
            .send(SessionCommand::Next(response))
            .map_err(|_| disconnected(self.request_cancel.is_cancelled()))?;
        let result = result
            .recv()
            .unwrap_or_else(|_| Err(disconnected(self.request_cancel.is_cancelled())));
        if matches!(result, Ok(None) | Err(_)) {
            self.finished.store(true, Ordering::Release);
        }
        result
    }

    fn cancel(&self) {
        self.request_cancel.cancel();
        if !self.finished.swap(true, Ordering::AcqRel) {
            self.cancel_worker();
        }
    }
}

impl Drop for Qwen35ProtocolTokenStream {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::{Qwen35ProtocolRuntime, Qwen35StepExecutor, Qwen35StepSession};
    use crate::server::service::ProtocolRuntime;
    use apxinf_model::{RuntimeCapabilities, RuntimeError, RuntimeRequest};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct FakeExecutor {
        sessions: Mutex<VecDeque<Result<VecDeque<Result<u32, RuntimeError>>, RuntimeError>>>,
    }

    struct FakeSession {
        steps: VecDeque<Result<u32, RuntimeError>>,
    }

    impl Qwen35StepSession for FakeSession {
        fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
            self.steps.pop_front().transpose()
        }
    }

    impl Qwen35StepExecutor for FakeExecutor {
        fn open(
            &self,
            _request: RuntimeRequest,
        ) -> Result<Box<dyn Qwen35StepSession>, RuntimeError> {
            self.sessions
                .lock()
                .unwrap()
                .pop_front()
                .unwrap()
                .map(|steps| Box::new(FakeSession { steps }) as Box<dyn Qwen35StepSession>)
        }
    }

    fn runtime(
        sessions: Vec<Result<Vec<Result<u32, RuntimeError>>, RuntimeError>>,
    ) -> Qwen35ProtocolRuntime {
        Qwen35ProtocolRuntime::new(
            RuntimeCapabilities::frozen_qwen35(32, 0),
            1,
            Arc::new(FakeExecutor {
                sessions: Mutex::new(
                    sessions
                        .into_iter()
                        .map(|session| session.map(Into::into))
                        .collect(),
                ),
            }),
        )
        .unwrap()
    }

    #[test]
    fn stream_advances_exactly_one_model_step_per_call() {
        let runtime = runtime(vec![Ok(vec![Ok(7), Ok(8)])]);
        let mut stream = runtime.start(RuntimeRequest::new(vec![1], 2)).unwrap();
        assert_eq!(runtime.active_requests(), 1);
        assert_eq!(stream.next_token().unwrap(), Some(7));
        assert_eq!(runtime.active_requests(), 1);
        assert_eq!(stream.next_token().unwrap(), Some(8));
        assert_eq!(stream.next_token().unwrap(), None);
        assert_eq!(runtime.active_requests(), 0);
    }

    #[test]
    fn stream_and_request_cancellation_release_capacity() {
        let runtime = runtime(vec![Ok(vec![Ok(7), Ok(8)]), Ok(vec![Ok(9)])]);
        let request = RuntimeRequest::new(vec![1], 2);
        let request_cancel = request.cancel.clone();
        let mut stream = runtime.start(request).unwrap();
        assert_eq!(stream.next_token().unwrap(), Some(7));
        request_cancel.cancel();
        for _ in 0..100 {
            if runtime.active_requests() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(runtime.active_requests(), 0);
        assert_eq!(stream.next_token(), Err(RuntimeError::Cancelled));

        let stream = runtime.start(RuntimeRequest::new(vec![1], 1)).unwrap();
        stream.cancel();
        assert_eq!(runtime.active_requests(), 0);
    }

    #[test]
    fn execution_error_does_not_poison_the_next_request() {
        let runtime = runtime(vec![
            Ok(vec![Err(RuntimeError::Execution("boom".into()))]),
            Ok(vec![Ok(9)]),
        ]);
        let mut failed = runtime.start(RuntimeRequest::new(vec![1], 1)).unwrap();
        assert!(matches!(
            failed.next_token(),
            Err(RuntimeError::Execution(_))
        ));
        assert_eq!(runtime.active_requests(), 0);
        let mut recovered = runtime.start(RuntimeRequest::new(vec![1], 1)).unwrap();
        assert_eq!(recovered.next_token().unwrap(), Some(9));
    }

    #[test]
    fn active_stream_rejects_excess_capacity() {
        let runtime = runtime(vec![Ok(vec![Ok(7)])]);
        let _stream = runtime.start(RuntimeRequest::new(vec![1], 1)).unwrap();
        assert!(matches!(
            runtime.start(RuntimeRequest::new(vec![1], 1)),
            Err(RuntimeError::Capacity)
        ));
    }

    #[test]
    fn cancelling_during_open_releases_capacity_and_maps_to_cancelled() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// First open mimics the production executor's cancellable prefill:
        /// it blocks until the request token is cancelled, then reports the
        /// abort. Subsequent opens return a working one-token session.
        struct SlowThenReadyExecutor {
            opens: AtomicUsize,
        }

        impl Qwen35StepExecutor for SlowThenReadyExecutor {
            fn open(
                &self,
                request: RuntimeRequest,
            ) -> Result<Box<dyn Qwen35StepSession>, RuntimeError> {
                if self.opens.fetch_add(1, Ordering::SeqCst) == 0 {
                    let deadline = std::time::Instant::now() + Duration::from_secs(5);
                    while std::time::Instant::now() < deadline {
                        if request.cancel.is_cancelled() {
                            return Err(RuntimeError::Cancelled);
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    return Err(RuntimeError::Execution("open was never cancelled".into()));
                }
                Ok(Box::new(FakeSession {
                    steps: vec![Ok(9)].into(),
                }))
            }
        }

        let runtime = Arc::new(
            Qwen35ProtocolRuntime::new(
                RuntimeCapabilities::frozen_qwen35(32, 0),
                1,
                Arc::new(SlowThenReadyExecutor {
                    opens: AtomicUsize::new(0),
                }),
            )
            .unwrap(),
        );
        let request = RuntimeRequest::new(vec![1], 2);
        let cancel = request.cancel.clone();
        let starter = {
            let runtime = Arc::clone(&runtime);
            std::thread::spawn(move || runtime.start(request).map(|_| ()))
        };
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(runtime.active_requests(), 1);
        cancel.cancel();
        assert_eq!(starter.join().unwrap(), Err(RuntimeError::Cancelled));
        for _ in 0..500 {
            if runtime.active_requests() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(runtime.active_requests(), 0);
        let mut recovered = runtime.start(RuntimeRequest::new(vec![1], 1)).unwrap();
        assert_eq!(recovered.next_token().unwrap(), Some(9));
    }

    #[test]
    fn warmup_runs_through_the_serial_worker() {
        let runtime_instance = runtime(vec![Ok(vec![Ok(7), Ok(8)])]);
        assert_eq!(runtime_instance.warmup(), Ok(()));

        let runtime_instance = runtime(vec![Ok(vec![Err(RuntimeError::Execution(
            "warmup boom".into(),
        ))])]);
        assert_eq!(
            runtime_instance.warmup(),
            Err(RuntimeError::Execution("warmup boom".into()))
        );
    }
}
