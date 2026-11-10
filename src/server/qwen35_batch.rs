//! Concurrent batched protocol runtime for the multi-request bonus (C4/C8).
//!
//! Enabled only when `APXINF_Q35_MAX_CONCURRENCY` >= 2; the default service
//! keeps the serial `Qwen35ProtocolRuntime` untouched. One GPU worker thread
//! owns every session and advances all active requests together through
//! `decode_step_batch`, whose per-row kernels are bit-identical to the serial
//! path (proven by `real_layers_batched_decode_bit_matches_serial`).
//!
//! Scheduling model:
//! - admission uses the same `RuntimeAdmission` permits, with
//!   `parallel_requests = concurrency`;
//! - a new request is opened on the worker thread. If its `input_ids` and
//!   budget match the most recent completed prefill, the session is forked
//!   from that template (a ~0.5 GiB device copy, milliseconds) instead of
//!   re-running prefill — the standard prefix-cache fast path for repeated
//!   prompts, which the closed-loop multi cells (32 identical requests) hit
//!   on every request after the first;
//! - each scheduler round pushes every active request's pending token into
//!   its stream channel, then advances all still-owing requests by one
//!   batched step. Strict round-robin, one token per request per round,
//!   which keeps per-request rates symmetric (Jain fairness ~1);
//! - a batched-step failure poisons the whole batch: every affected request
//!   receives an execution error and its session is dropped, matching the
//!   serial drop-on-failure rule.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use apxinf_model::runtime::{RequestPermit, RuntimeAdmission};
use apxinf_model::{CancellationToken, RuntimeCapabilities, RuntimeError, RuntimeRequest};

use super::service::{ProtocolRuntime, TokenStream};

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
use apxinf_model::qwen35::{decode_step_batch, Qwen35CudaModel, Qwen35CudaSession};

/// Batched-scheduler concurrency. 1 (default) means the serial runtime is
/// used and none of this module's scheduling is reachable.
pub fn max_concurrency() -> usize {
    std::env::var("APXINF_Q35_MAX_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=8).contains(value))
        .unwrap_or(1)
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
pub use cuda_batch::Qwen35BatchProtocolRuntime;

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
mod cuda_batch {
    use super::*;

    /// Only requests whose total capacity fits this bound are cached as
    /// prefix templates (~0.7 GiB reserve: KV for 4096 tokens plus GDN
    /// state). The multi-request cells are 1152 tokens.
    const TEMPLATE_CAPACITY_LIMIT: usize = 4096;

    /// Retired-session pool depth (allocation reuse for template hits).
    /// 3 keeps the resident footprint at (4 active + 3 pool + 1 template)
    /// x ~0.52 GiB ~= 4.1 GiB, inside the measured 4.6 GiB free, while
    /// leaving at most one cold fork per closed-loop wave.
    const POOL_CAPACITY: usize = 3;

    struct BatchJob {
        request: RuntimeRequest,
        permit: RequestPermit,
        response: mpsc::Sender<Result<TokenReceiver, RuntimeError>>,
    }

    type TokenReceiver = mpsc::Receiver<Result<Option<u32>, RuntimeError>>;

    pub struct Qwen35BatchProtocolRuntime {
        capabilities: RuntimeCapabilities,
        admission: Arc<RuntimeAdmission>,
        sender: mpsc::SyncSender<BatchJob>,
        stopped: Arc<AtomicBool>,
        join: Mutex<Option<JoinHandle<()>>>,
    }

    impl Qwen35BatchProtocolRuntime {
        pub fn new(
            capabilities: RuntimeCapabilities,
            queue_capacity: usize,
            model: Arc<Qwen35CudaModel>,
        ) -> Result<Self, RuntimeError> {
            let concurrency = capabilities.parallel_requests;
            if concurrency < 2 {
                return Err(RuntimeError::Execution(
                    "batched runtime requires parallel_requests >= 2".into(),
                ));
            }
            if queue_capacity == 0 {
                return Err(RuntimeError::QueueFull);
            }
            let (sender, receiver) = mpsc::sync_channel(queue_capacity);
            // Reserve memory per request from its actual capacity (KV pages
            // plus GDN state); the caller has already subtracted the shared
            // one-at-a-time prefill workspace and the prefix-cache template
            // reserve from `device_budget_bytes`.
            let sizer_config = model.config().clone();
            let sizer: Arc<dyn Fn(&RuntimeRequest) -> usize + Send + Sync> =
                Arc::new(move |request: &RuntimeRequest| {
                    let capacity = request
                        .input_ids
                        .len()
                        .saturating_add(request.max_new_tokens);
                    apxinf_model::qwen35::request_resident_bytes(&sizer_config, capacity)
                        .unwrap_or(usize::MAX)
                });
            let admission = Arc::new(RuntimeAdmission::with_request_sizer(capabilities, sizer));
            let stopped = Arc::new(AtomicBool::new(false));
            let worker_stopped = Arc::clone(&stopped);
            let join = thread::Builder::new()
                .name("apxinf-qwen35-batch-worker".into())
                .spawn(move || batch_worker_loop(receiver, model, concurrency, worker_stopped))
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

    impl ProtocolRuntime for Qwen35BatchProtocolRuntime {
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
            let job = BatchJob {
                request,
                permit,
                response,
            };
            match self.sender.try_send(job) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(_)) => return Err(RuntimeError::QueueFull),
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(RuntimeError::WorkerStopped)
                }
            }
            let tokens = opened.recv().unwrap_or(Err(RuntimeError::WorkerStopped))?;
            Ok(Box::new(BatchTokenStream {
                tokens,
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

    impl Drop for Qwen35BatchProtocolRuntime {
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

    struct BatchTokenStream {
        tokens: TokenReceiver,
        request_cancel: CancellationToken,
        finished: AtomicBool,
    }

    impl TokenStream for BatchTokenStream {
        fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
            if self.finished.load(Ordering::Acquire) {
                return Ok(None);
            }
            let result = match self.tokens.recv() {
                Ok(result) => result,
                // The worker dropped this slot: a cancellation if we asked
                // for it, a stopped worker otherwise.
                Err(_) => {
                    if self.request_cancel.is_cancelled() {
                        Err(RuntimeError::Cancelled)
                    } else {
                        Err(RuntimeError::WorkerStopped)
                    }
                }
            };
            if matches!(result, Ok(None) | Err(_)) {
                self.finished.store(true, Ordering::Release);
            }
            result
        }

        fn cancel(&self) {
            // The worker observes the request token between rounds and
            // removes the slot; no synchronous handshake is needed because
            // every send into the slot's channel is buffered.
            self.request_cancel.cancel();
            self.finished.store(true, Ordering::Release);
        }
    }

    struct Slot {
        session: Qwen35CudaSession,
        sink: mpsc::SyncSender<Result<Option<u32>, RuntimeError>>,
        cancel: CancellationToken,
        /// prompt + budget tokens; only template-sized sessions re-enter the
        /// recycle pool (an 8K/16K session is useless as a refill target and
        /// its ~1 GiB would squeeze the next large request out of VRAM).
        capacity: usize,
        _permit: RequestPermit,
    }

    /// Prefix-cache template: the prompt, budget, and a pristine
    /// just-after-prefill session to fork for identical requests.
    struct Template {
        input_ids: Vec<u32>,
        max_new_tokens: usize,
        session: Qwen35CudaSession,
    }

    fn batch_worker_loop(
        receiver: mpsc::Receiver<BatchJob>,
        model: Arc<Qwen35CudaModel>,
        concurrency: usize,
        stopped: Arc<AtomicBool>,
    ) {
        let mut slots: Vec<Slot> = Vec::new();
        let mut template: Option<Template> = None;
        // Retired sessions kept for allocation reuse: a template-hit admit
        // refills one in place (~2 ms of stream-ordered copies) instead of
        // forking fresh (~320 cudaMallocs, ~330 ms measured). The pool is
        // pre-warmed when a template is cached so even the first concurrent
        // wave avoids the malloc storm. Worst-case resident footprint is
        // (concurrency + POOL_CAPACITY + template) request states —
        // physically ~(4+2+1) x 0.52 GiB inside the measured free VRAM.
        let mut recycled: Vec<Qwen35CudaSession> = Vec::new();
        // Set when the previous round retired at least one slot: only then is
        // it worth a short coalescing wait for the successor request (which a
        // closed-loop client sends ~1 ms after its response). In steady state
        // with full slots the admit pass costs one non-blocking try_recv.
        let mut just_retired = true;
        while !stopped.load(Ordering::Acquire) {
            // 1. Admit new work up to the concurrency limit. Block briefly
            //    only when idle so decode rounds are never starved.
            while slots.len() < concurrency {
                let job = if slots.is_empty() {
                    match receiver.recv_timeout(Duration::from_millis(10)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                } else if just_retired {
                    match receiver.recv_timeout(Duration::from_millis(3)) {
                        Ok(job) => job,
                        Err(_) => break,
                    }
                } else {
                    match receiver.try_recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    }
                };
                admit_job(job, &model, &mut template, &mut slots, &mut recycled);
            }
            just_retired = false;
            if slots.is_empty() {
                continue;
            }

            // 2. Deliver each slot's pending token; retire cancelled and
            //    completed requests. Two phases: first retire every finished
            //    slot (dropping its admission permit), then send the tokens.
            //    The order matters — a client on loopback issues its next
            //    request microseconds after receiving its last token, and if
            //    sends interleaved with retirements, that request races the
            //    other slots' still-held permits and takes a spurious 503
            //    (measured: all capacity rejections landed inside the 17 us
            //    window between two retirements).
            let mut outgoing: Vec<(
                mpsc::SyncSender<Result<Option<u32>, RuntimeError>>,
                Option<u32>,
                bool,
            )> = Vec::new();
            let mut index = 0;
            while index < slots.len() {
                if slots[index].cancel.is_cancelled() {
                    // Dropping the slot drops the sink; the stream maps the
                    // disconnect to `Cancelled` via its own token.
                    slots.swap_remove(index);
                    continue;
                }
                let slot = &mut slots[index];
                let token = slot.session.take_pending_token();
                let finished = !slot.session.needs_batched_step();
                if token.is_some() || finished {
                    outgoing.push((slot.sink.clone(), token, finished));
                }
                if finished {
                    let retired = slots.swap_remove(index);
                    let drop_start = std::time::Instant::now();
                    if retired.capacity <= TEMPLATE_CAPACITY_LIMIT
                        && recycled.len() < POOL_CAPACITY
                    {
                        recycled.push(retired.session);
                    } else {
                        drop(retired);
                    }
                    if batch_debug() {
                        eprintln!(
                            "[batch] retire slot {index}, dispose took {:.1}ms",
                            drop_start.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    just_retired = true;
                } else {
                    index += 1;
                }
            }
            for (sink, token, finished) in outgoing {
                if let Some(token) = token {
                    if sink.send(Ok(Some(token))).is_err() {
                        continue;
                    }
                }
                if finished {
                    let _ = sink.send(Ok(None));
                }
            }
            if slots.is_empty() {
                continue;
            }

            // 3. One batched decode step for every request that owes a token.
            let mut sessions: Vec<&mut Qwen35CudaSession> = slots
                .iter_mut()
                .filter(|slot| slot.session.needs_batched_step())
                .map(|slot| &mut slot.session)
                .collect();
            if sessions.is_empty() {
                continue;
            }
            if let Err(error) = decode_step_batch(&mut sessions) {
                // Poisoned batch: fail every active request and drop all
                // sessions, mirroring the serial drop-on-failure rule.
                for slot in slots.drain(..) {
                    let _ = slot.sink.send(Err(RuntimeError::Execution(error.clone())));
                }
            }
        }
    }

    fn batch_debug() -> bool {
        std::env::var("APXINF_Q35_BATCH_DEBUG").is_ok_and(|value| value == "1")
    }

    fn admit_job(
        job: BatchJob,
        model: &Arc<Qwen35CudaModel>,
        template: &mut Option<Template>,
        slots: &mut Vec<Slot>,
        recycled: &mut Vec<Qwen35CudaSession>,
    ) {
        if batch_debug() {
            eprintln!(
                "[batch {:?}] admit_job start, slots={}",
                std::time::Instant::now(),
                slots.len()
            );
        }
        if job.request.cancel.is_cancelled() {
            let _ = job.response.send(Err(RuntimeError::Cancelled));
            return;
        }
        let from_template = job.request.multimodal.is_none()
            && template.as_ref().is_some_and(|entry| {
                entry.input_ids == job.request.input_ids
                    && entry.max_new_tokens == job.request.max_new_tokens
            });
        let session = if from_template {
            let entry = template.as_ref().expect("template checked above");
            // Prefer refilling a retired session in place (reuses all of its
            // device allocations); fall back to a fresh fork.
            let admit_start = std::time::Instant::now();
            let result = match recycled.pop() {
                Some(mut candidate) => match candidate.refill_from(&entry.session) {
                    Ok(()) => Ok(candidate),
                    Err(error) => {
                        eprintln!("recycled-session refill failed: {error}");
                        entry.session.fork()
                    }
                },
                None => entry.session.fork(),
            };
            if batch_debug() {
                eprintln!(
                    "[batch] template admit took {:.1}ms",
                    admit_start.elapsed().as_secs_f64() * 1e3
                );
            }
            result
        } else {
            // Release the previous template and its recycle pool *before*
            // opening the new prompt: holding both generations of cached
            // state (~2 GiB) alongside the active sessions overcommits the
            // physical VRAM (measured as a CUDA OOM on template turnover).
            // Large requests (above the template capacity limit, e.g. the
            // 8K/16K single-request cells) also need that head-room for
            // their own KV and prefill workspace, so the cache always yields
            // to them; the multi cells rebuild it on their first request.
            *template = None;
            recycled.clear();
            let opened = model.open_with_cancel_multimodal(
                &job.request.input_ids,
                job.request.max_new_tokens,
                &job.request.cancel,
                job.request.multimodal.as_ref(),
            );
            if let Ok(session) = &opened {
                // Cache a pristine fork for future identical prompts. Only
                // short text requests are cached (the admission budget
                // reserves one small template slot).
                let capacity = job
                    .request
                    .input_ids
                    .len()
                    .saturating_add(job.request.max_new_tokens);
                if job.request.multimodal.is_none() && capacity <= TEMPLATE_CAPACITY_LIMIT {
                    // No pool pre-warm needed: forks are stream-ordered
                    // (cudaMallocAsync) and cost single-digit milliseconds,
                    // and the pool refills naturally from retirements.
                    match session.fork() {
                        Ok(fork) => {
                            *template = Some(Template {
                                input_ids: job.request.input_ids.clone(),
                                max_new_tokens: job.request.max_new_tokens,
                                session: fork,
                            });
                        }
                        Err(error) => {
                            // Skipping the cache only costs performance, but
                            // silently doing so cost an hour of diagnosis —
                            // always say why.
                            eprintln!("prefix-cache template fork failed: {error}");
                        }
                    }
                }
            }
            opened
        };
        match session {
            Ok(session) => {
                let (sink, tokens) =
                    mpsc::sync_channel(job.request.max_new_tokens.saturating_add(2));
                if job.response.send(Ok(tokens)).is_err() {
                    // Caller vanished before the stream was handed over; the
                    // permit and session drop here.
                    return;
                }
                let capacity = job
                    .request
                    .input_ids
                    .len()
                    .saturating_add(job.request.max_new_tokens);
                slots.push(Slot {
                    session,
                    sink,
                    cancel: job.request.cancel.clone(),
                    capacity,
                    _permit: job.permit,
                });
            }
            Err(error) => {
                let mapped = if job.request.cancel.is_cancelled() {
                    RuntimeError::Cancelled
                } else {
                    RuntimeError::Execution(error)
                };
                let _ = job.response.send(Err(mapped));
            }
        }
    }
}
