use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use thiserror::Error;

use crate::qwen35::admission::{
    validate_input_ids_with_vocab, validate_total_budget, AdmissionError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub vocab_size: usize,
    pub max_model_len: usize,
    pub parallel_requests: usize,
    pub device_budget_bytes: usize,
    pub per_request_bytes: usize,
}

impl RuntimeCapabilities {
    pub fn frozen_qwen35(max_model_len: usize, device_budget_bytes: usize) -> Self {
        Self {
            vocab_size: crate::qwen35::MODEL_VOCAB_SIZE,
            max_model_len,
            parallel_requests: 1,
            device_budget_bytes,
            per_request_bytes: 0,
        }
    }
}

/// Preprocessed image input for a multimodal request. `pixel_values` is the
/// processor output (`[t*h*w, 1536]` row-major, BF16) and `grid` is the
/// `[t, h, w]` patch grid. The vision tower forward runs on the GPU worker
/// thread so image and text requests stay serialized on the single device.
#[derive(Debug, Clone)]
pub struct MultimodalPayload {
    pub pixel_values: Vec<half::bf16>,
    pub grid: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct RuntimeRequest {
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub cancel: CancellationToken,
    pub multimodal: Option<MultimodalPayload>,
}

impl RuntimeRequest {
    pub fn new(input_ids: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            input_ids,
            max_new_tokens,
            cancel: CancellationToken::new(),
            multimodal: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeResult {
    pub output_ids: Vec<u32>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    #[error("runtime admission failed: {0}")]
    Admission(#[from] AdmissionError),
    #[error("runtime capacity is exhausted")]
    Capacity,
    #[error("runtime request queue is full")]
    QueueFull,
    #[error("runtime request was cancelled")]
    Cancelled,
    #[error("runtime worker stopped")]
    WorkerStopped,
    #[error("runtime service is unhealthy")]
    Unhealthy,
    #[error("runtime execution failed: {0}")]
    Execution(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmissionDecision {
    pub reserved_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Usage {
    active_requests: usize,
    reserved_bytes: usize,
}

pub struct RuntimeAdmission {
    capabilities: RuntimeCapabilities,
    usage: Arc<Mutex<Usage>>,
    request_sizer: Option<Arc<dyn Fn(&RuntimeRequest) -> usize + Send + Sync>>,
}

impl std::fmt::Debug for RuntimeAdmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeAdmission")
            .field("capabilities", &self.capabilities)
            .field("usage", &self.usage)
            .field("request_sizer", &self.request_sizer.is_some())
            .finish()
    }
}

impl RuntimeAdmission {
    pub fn new(capabilities: RuntimeCapabilities) -> Self {
        Self {
            capabilities,
            usage: Arc::new(Mutex::new(Usage {
                active_requests: 0,
                reserved_bytes: 0,
            })),
            request_sizer: None,
        }
    }

    /// Admission that reserves memory per request from its actual token
    /// capacity instead of the worst-case `per_request_bytes`. Used by the
    /// concurrent batched runtime, where reserving `max_model_len` KV for
    /// every slot would forbid any useful concurrency; the serial runtime
    /// keeps the conservative prompt-independent accounting unchanged.
    pub fn with_request_sizer(
        capabilities: RuntimeCapabilities,
        sizer: Arc<dyn Fn(&RuntimeRequest) -> usize + Send + Sync>,
    ) -> Self {
        Self {
            capabilities,
            usage: Arc::new(Mutex::new(Usage {
                active_requests: 0,
                reserved_bytes: 0,
            })),
            request_sizer: Some(sizer),
        }
    }

    pub fn capabilities(&self) -> RuntimeCapabilities {
        self.capabilities
    }

    pub fn admit(
        &self,
        request: &RuntimeRequest,
    ) -> Result<(AdmissionDecision, RequestPermit), RuntimeError> {
        validate_input_ids_with_vocab(&request.input_ids, self.capabilities.vocab_size)?;
        validate_total_budget(
            request.input_ids.len(),
            request.max_new_tokens,
            self.capabilities.max_model_len,
        )?;
        let reserved_bytes = match &self.request_sizer {
            Some(sizer) => sizer(request),
            None => self.capabilities.per_request_bytes,
        };
        let mut usage = self.usage.lock().map_err(|_| RuntimeError::WorkerStopped)?;
        let active_ok = usage.active_requests < self.capabilities.parallel_requests;
        let bytes_ok = usage
            .reserved_bytes
            .checked_add(reserved_bytes)
            .is_some_and(|total| total <= self.capabilities.device_budget_bytes);
        if !active_ok || !bytes_ok {
            if self.request_sizer.is_some() {
                eprintln!(
                    "admission capacity: active={}/{} reserved={}+{}/{}",
                    usage.active_requests,
                    self.capabilities.parallel_requests,
                    usage.reserved_bytes,
                    reserved_bytes,
                    self.capabilities.device_budget_bytes
                );
            }
            return Err(RuntimeError::Capacity);
        }
        usage.active_requests += 1;
        usage.reserved_bytes += reserved_bytes;
        Ok((
            AdmissionDecision { reserved_bytes },
            RequestPermit {
                usage: Arc::clone(&self.usage),
                reserved_bytes,
            },
        ))
    }

    pub fn active_requests(&self) -> usize {
        self.usage
            .lock()
            .map(|usage| usage.active_requests)
            .unwrap_or(usize::MAX)
    }
}

#[derive(Debug)]
pub struct RequestPermit {
    usage: Arc<Mutex<Usage>>,
    reserved_bytes: usize,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        if let Ok(mut usage) = self.usage.lock() {
            usage.active_requests = usage.active_requests.saturating_sub(1);
            usage.reserved_bytes = usage.reserved_bytes.saturating_sub(self.reserved_bytes);
        }
    }
}

struct Job {
    request: RuntimeRequest,
    permit: RequestPermit,
    result: mpsc::Sender<Result<RuntimeResult, RuntimeError>>,
}

pub struct RuntimeHandle {
    sender: mpsc::SyncSender<Job>,
    admission: Arc<RuntimeAdmission>,
    stopped: Arc<AtomicBool>,
}

impl Clone for RuntimeHandle {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            admission: Arc::clone(&self.admission),
            stopped: Arc::clone(&self.stopped),
        }
    }
}

impl RuntimeHandle {
    pub fn submit(&self, request: RuntimeRequest) -> Result<RuntimeTicket, RuntimeError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(RuntimeError::WorkerStopped);
        }
        let cancel = request.cancel.clone();
        let (decision, permit) = self.admission.admit(&request)?;
        let (result, receiver) = mpsc::channel();
        let job = Job {
            request,
            permit,
            result,
        };
        match self.sender.try_send(job) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => return Err(RuntimeError::QueueFull),
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(RuntimeError::WorkerStopped),
        }
        Ok(RuntimeTicket {
            receiver,
            cancel,
            decision,
        })
    }
}

pub struct RuntimeTicket {
    receiver: mpsc::Receiver<Result<RuntimeResult, RuntimeError>>,
    cancel: CancellationToken,
    pub decision: AdmissionDecision,
}

impl RuntimeTicket {
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn recv(self) -> Result<RuntimeResult, RuntimeError> {
        self.receiver
            .recv()
            .unwrap_or(Err(RuntimeError::WorkerStopped))
    }
}

pub struct RuntimeWorker {
    handle: RuntimeHandle,
    join: Option<JoinHandle<()>>,
}

impl RuntimeWorker {
    pub fn start<F>(
        capabilities: RuntimeCapabilities,
        queue_capacity: usize,
        executor: F,
    ) -> Result<Self, RuntimeError>
    where
        F: Fn(RuntimeRequest) -> Result<RuntimeResult, RuntimeError> + Send + 'static,
    {
        if queue_capacity == 0 {
            return Err(RuntimeError::QueueFull);
        }
        let (sender, receiver) = mpsc::sync_channel::<Job>(queue_capacity);
        let admission = Arc::new(RuntimeAdmission::new(capabilities));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_stopped = Arc::clone(&stopped);
        let join = thread::Builder::new()
            .name("apxinf-gpu-worker".into())
            .spawn(move || loop {
                if worker_stopped.load(Ordering::Acquire) {
                    break;
                }
                let job = match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(job) => job,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                let result = if job.request.cancel.is_cancelled() {
                    Err(RuntimeError::Cancelled)
                } else {
                    executor(job.request.clone())
                };
                let _ = job.result.send(result);
                drop(job.permit);
            })
            .map_err(|error| RuntimeError::Execution(error.to_string()))?;
        Ok(Self {
            handle: RuntimeHandle {
                sender,
                admission,
                stopped,
            },
            join: Some(join),
        })
    }

    pub fn handle(&self) -> &RuntimeHandle {
        &self.handle
    }

    pub fn active_requests(&self) -> usize {
        self.handle.admission.active_requests()
    }
}

impl Drop for RuntimeWorker {
    fn drop(&mut self) {
        self.handle.stopped.store(true, Ordering::Release);
        let (replacement, _receiver) = mpsc::sync_channel(0);
        let old = std::mem::replace(&mut self.handle.sender, replacement);
        drop(old);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn caps() -> RuntimeCapabilities {
        RuntimeCapabilities {
            vocab_size: 248_320,
            max_model_len: 64,
            parallel_requests: 3,
            device_budget_bytes: 300,
            per_request_bytes: 100,
        }
    }

    #[test]
    fn permit_is_released_after_execution() {
        let worker = RuntimeWorker::start(caps(), 2, |_| {
            Ok(RuntimeResult {
                output_ids: vec![7],
            })
        })
        .unwrap();
        let ticket = worker
            .handle()
            .submit(RuntimeRequest::new(vec![1], 1))
            .unwrap();
        assert_eq!(ticket.recv().unwrap().output_ids, vec![7]);
        assert_eq!(worker.active_requests(), 0);
    }

    #[test]
    fn cancelled_queued_request_does_not_execute() {
        let executions = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&executions);
        let worker = RuntimeWorker::start(caps(), 2, move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(20));
            Ok(RuntimeResult { output_ids: vec![] })
        })
        .unwrap();
        let first = worker
            .handle()
            .submit(RuntimeRequest::new(vec![1], 1))
            .unwrap();
        let ticket = worker
            .handle()
            .submit(RuntimeRequest::new(vec![1], 1))
            .unwrap();
        ticket.cancel();
        assert_eq!(ticket.recv(), Err(RuntimeError::Cancelled));
        let _ = first.recv();
        assert_eq!(executions.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn queue_full_releases_capacity_permit() {
        let mut limited = caps();
        limited.parallel_requests = 3;
        let started = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&started);
        let worker = RuntimeWorker::start(limited, 1, move |_| {
            signal.store(true, Ordering::Release);
            thread::sleep(Duration::from_millis(100));
            Ok(RuntimeResult { output_ids: vec![] })
        })
        .unwrap();
        let handle = worker.handle();
        let first = handle.submit(RuntimeRequest::new(vec![1], 1)).unwrap();
        while !started.load(Ordering::Acquire) {
            thread::yield_now();
        }
        let second = handle.submit(RuntimeRequest::new(vec![1], 1)).unwrap();
        assert!(matches!(
            handle.submit(RuntimeRequest::new(vec![1], 1)),
            Err(RuntimeError::QueueFull)
        ));
        assert_eq!(worker.active_requests(), 2);
        let _ = first.recv();
        let _ = second.recv();
    }

    #[test]
    fn worker_drop_does_not_wait_for_cloned_handle() {
        let worker =
            RuntimeWorker::start(caps(), 1, |_| Ok(RuntimeResult { output_ids: vec![] })).unwrap();
        let cloned_handle = worker.handle().clone();
        drop(worker);
        assert!(matches!(
            cloned_handle.submit(RuntimeRequest::new(vec![1], 1)),
            Err(RuntimeError::WorkerStopped)
        ));
    }
}
