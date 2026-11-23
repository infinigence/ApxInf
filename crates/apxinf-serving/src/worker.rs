use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    mpsc::{sync_channel, SyncSender as JobSender, TrySendError},
    Arc, Mutex,
};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::engine::{ChatEngine, EngineOutput, EngineRequest};
use crate::error::ApiError;

const LOADING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

enum JobResponse {
    Complete(oneshot::Sender<Result<EngineOutput, ApiError>>),
    Stream(mpsc::Sender<Result<EngineStreamEvent, ApiError>>),
}

impl JobResponse {
    fn is_closed(&self) -> bool {
        match self {
            Self::Complete(sender) => sender.is_closed(),
            Self::Stream(sender) => sender.is_closed(),
        }
    }
}

struct Job {
    request: EngineRequest,
    response: JobResponse,
    timing: Option<(u64, Instant)>,
}

#[derive(Clone, Debug)]
pub enum EngineStreamEvent {
    TextDelta(String),
    Complete(EngineOutput),
}

struct WorkerStatus {
    state: AtomicU8,
    error: Mutex<Option<String>>,
    vision: Mutex<Option<apxinf_model::vision_preprocess::VisionContract>>,
}

impl WorkerStatus {
    fn fail(&self, message: String) {
        *self.error.lock().unwrap_or_else(|error| error.into_inner()) = Some(message);
        self.state.store(FAILED, Ordering::Release);
    }

    /// Publish failure before the response sender can close. A panic poisons
    /// the model: the worker must exit after reporting this error, never retry.
    fn run<T>(&self, operation: impl FnOnce() -> Result<T, ApiError>) -> Result<T, ApiError> {
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(result) => result,
            Err(_) => {
                let message = "model worker panicked; restart required";
                self.fail(message.into());
                Err(ApiError::unavailable(message))
            }
        }
    }
}

#[derive(Clone)]
pub struct WorkerClient {
    sender: JobSender<Job>,
    status: Arc<WorkerStatus>,
    log_timing: bool,
}

impl WorkerClient {
    /// Start a batch-one worker. `factory` runs inside the worker thread so a
    /// thread-bound CUDA/model object is never constructed and then moved.
    pub fn vision_contract(&self) -> Option<apxinf_model::vision_preprocess::VisionContract> {
        self.status
            .vision
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn spawn<F>(queue_capacity: usize, factory: F) -> Self
    where
        F: FnOnce() -> Result<Box<dyn ChatEngine>, ApiError> + Send + 'static,
    {
        Self::spawn_with_timing(queue_capacity, false, factory)
    }

    pub fn spawn_with_timing<F>(queue_capacity: usize, log_timing: bool, factory: F) -> Self
    where
        F: FnOnce() -> Result<Box<dyn ChatEngine>, ApiError> + Send + 'static,
    {
        let (sender, receiver) = sync_channel::<Job>(queue_capacity.max(1));
        let status = Arc::new(WorkerStatus {
            state: AtomicU8::new(LOADING),
            error: Mutex::new(None),
            vision: Mutex::new(None),
        });
        let thread_status = status.clone();
        std::thread::Builder::new()
            .name("apxinf-inference".into())
            .spawn(move || {
                let mut engine = match thread_status.run(factory) {
                    Ok(engine) => {
                        *thread_status
                            .vision
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = engine.vision_contract().cloned();
                        thread_status.state.store(READY, Ordering::Release);
                        engine
                    }
                    Err(error) => {
                        thread_status.fail(error.message);
                        return;
                    }
                };
                while let Ok(job) = receiver.recv() {
                    // A queued HTTP request may have been canceled before it
                    // reached the batch-one worker. Do not spend an inference
                    // slot on work whose receiver no longer exists.
                    if job.response.is_closed() {
                        continue;
                    }
                    let started = job.timing.map(|(sequence, enqueued)| {
                        let started = Instant::now();
                        eprintln!(
                            "aios worker start: sequence={sequence:016x} queue_ms={:.6}",
                            (started - enqueued).as_secs_f64() * 1_000.0
                        );
                        (sequence, started)
                    });
                    match job.response {
                        JobResponse::Complete(response) => {
                            let result = thread_status.run(|| engine.complete(job.request));
                            let _ = response.send(result);
                        }
                        JobResponse::Stream(response) => {
                            let mut connected = true;
                            let mut first_delta_sent = false;
                            let mut pending = String::new();
                            let mut pending_deltas = 0usize;
                            let result = thread_status.run(|| {
                                engine.complete_stream(job.request, &mut |delta| {
                                    if !connected {
                                        return false;
                                    }
                                    // Preserve first-token streaming latency, then
                                    // coalesce the small UTF-8 fragments emitted by
                                    // an MTP round. This keeps byte-for-byte output
                                    // while avoiding one cross-thread wakeup and
                                    // one SSE frame per generated token.
                                    if !first_delta_sent {
                                        first_delta_sent = true;
                                        connected = response
                                            .blocking_send(Ok(EngineStreamEvent::TextDelta(
                                                delta.to_owned(),
                                            )))
                                            .is_ok();
                                    } else {
                                        pending.push_str(delta);
                                        pending_deltas += 1;
                                        if pending_deltas >= 8 {
                                            connected = response
                                                .blocking_send(Ok(EngineStreamEvent::TextDelta(
                                                    std::mem::take(&mut pending),
                                                )))
                                                .is_ok();
                                            pending_deltas = 0;
                                        }
                                    }
                                    connected
                                })
                            });
                            if connected {
                                if !pending.is_empty() {
                                    connected = response
                                        .blocking_send(Ok(EngineStreamEvent::TextDelta(pending)))
                                        .is_ok();
                                }
                            }
                            if connected {
                                let terminal = result.map(EngineStreamEvent::Complete);
                                let _ = response.blocking_send(terminal);
                            }
                        }
                    }
                    if thread_status.state.load(Ordering::Acquire) == FAILED {
                        return;
                    }
                    if let Some((sequence, started)) = started {
                        eprintln!(
                            "aios worker end: sequence={sequence:016x} worker_ms={:.6}",
                            started.elapsed().as_secs_f64() * 1_000.0
                        );
                    }
                }
            })
            .expect("create ApxInf inference worker thread");
        Self {
            sender,
            status,
            log_timing,
        }
    }

    pub fn ready(&self) -> bool {
        self.status.state.load(Ordering::Acquire) == READY
    }

    pub fn status_message(&self) -> &'static str {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => "loading",
            READY => "ready",
            FAILED => "failed",
            _ => "unknown",
        }
    }

    pub fn startup_error(&self) -> Option<String> {
        self.status
            .error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }

    pub fn submit(
        &self,
        request: EngineRequest,
    ) -> Result<oneshot::Receiver<Result<EngineOutput, ApiError>>, ApiError> {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => return Err(ApiError::unavailable("model is still loading")),
            FAILED => {
                return Err(ApiError::unavailable(
                    self.startup_error()
                        .unwrap_or_else(|| "model worker failed to start".into()),
                ))
            }
            READY => {}
            _ => return Err(ApiError::unavailable("model worker is unavailable")),
        }
        let (response, receiver) = oneshot::channel();
        match self.sender.try_send(Job {
            request,
            response: JobResponse::Complete(response),
            timing: None,
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(ApiError::queue_full()),
            Err(TrySendError::Disconnected(_)) => {
                return Err(ApiError::unavailable("model worker stopped"))
            }
        }
        Ok(receiver)
    }

    pub fn submit_stream(
        &self,
        request: EngineRequest,
    ) -> Result<mpsc::Receiver<Result<EngineStreamEvent, ApiError>>, ApiError> {
        self.submit_stream_for_request(request, None)
    }

    pub(crate) fn submit_stream_for_request(
        &self,
        request: EngineRequest,
        sequence: Option<u64>,
    ) -> Result<mpsc::Receiver<Result<EngineStreamEvent, ApiError>>, ApiError> {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => return Err(ApiError::unavailable("model is still loading")),
            FAILED => {
                return Err(ApiError::unavailable(
                    self.startup_error()
                        .unwrap_or_else(|| "model worker failed to start".into()),
                ))
            }
            READY => {}
            _ => return Err(ApiError::unavailable("model worker is unavailable")),
        }
        // This is separate from the inference queue capacity. It bounds
        // decoded text waiting for the HTTP task and applies backpressure to
        // the generation callback when a client reads slowly.
        // One MTP round can publish several deltas back-to-back. Keep a full
        // short completion buffered so HTTP/SSE scheduling never stalls the
        // GPU worker between speculative rounds. The channel remains bounded
        // and disconnect still closes it immediately.
        let (response, receiver) = mpsc::channel(128);
        match self.sender.try_send(Job {
            request,
            response: JobResponse::Stream(response),
            timing: sequence
                .filter(|_| self.log_timing)
                .map(|sequence| (sequence, Instant::now())),
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(ApiError::queue_full()),
            Err(TrySendError::Disconnected(_)) => {
                return Err(ApiError::unavailable("model worker stopped"))
            }
        }
        Ok(receiver)
    }

    pub async fn complete(&self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        let receiver = self.submit(request)?;
        receiver
            .await
            .map_err(|_| ApiError::unavailable("model worker stopped before responding"))?
    }
}
