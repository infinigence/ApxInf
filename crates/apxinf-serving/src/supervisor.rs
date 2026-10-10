//! One worker, a bounded queue, and explicit attempt settlement.
mod generation;
mod pending_queue;

#[cfg(test)]
mod diagnostic_regressions;
#[cfg(test)]
mod generation_control_regressions;
#[cfg(test)]
mod generation_deadline_regressions;
#[cfg(test)]
mod lifecycle_regressions;
#[cfg(test)]
mod memory_regressions;
#[cfg(test)]
mod rotation_pressure_regressions;
#[cfg(test)]
mod submit_regressions;
use crate::{
    contracts,
    diagnostics::{self, WorkerDiagnostics},
    host_pressure::{HostPressure, HostPressurePolicy},
    metrics::{ObservationMetrics, Outcome, RequestObservation},
    output::{OutputParser, Part, StopMatcher},
};
use generation::{wait_for_generation_control, GenerationPublic};
use pending_queue::PendingQueue;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::future::Future;
use std::io::Read;
use std::{
    path::PathBuf,
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{mpsc, oneshot, watch},
};

pub const PROTOCOL: &str = "apxinf-worker/2.0";
const FRAME_LIMIT: usize = 1_048_576;
const OUTPUT_PART_CAPACITY: usize = 64;
const REQUEST_COMMAND_ALLOWANCE: u64 = 4;
const SETTLEMENT_GRACE: Duration = Duration::from_secs(20);
const QUEUE_SWEEP_INTERVAL: Duration = Duration::from_millis(20);
type StopResult = Result<(), String>;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub code: String,
    pub message: String,
}
impl ApiError {
    pub fn new(status: u16, code: &str, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
        }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request", message)
    }
    fn worker(message: impl Into<String>) -> Self {
        Self::new(503, "worker_lost", message)
    }
}

#[derive(Clone)]
pub struct Config {
    pub python: PathBuf,
    pub worker: PathBuf,
    pub model: PathBuf,
    pub model_id: String,
    pub max_context: usize,
    pub max_output: usize,
    pub prefill_step_size: usize,
    pub output_batch_tokens: usize,
    pub queue_capacity: usize,
    pub timeout: Duration,
    pub memory_budget: u64,
    pub sequence_reservation: u64,
    pub host_pressure_policy: HostPressurePolicy,
}

impl Config {
    pub fn validate_memory(&self) -> Result<(), String> {
        if !(1..=contracts::MAX_SAFE_INTEGER).contains(&self.memory_budget)
            || !(1..=contracts::MAX_SAFE_INTEGER).contains(&self.sequence_reservation)
        {
            return Err("Memory limits must be between 1 and 9007199254740991 bytes.".into());
        }
        if self.sequence_reservation > self.memory_budget {
            return Err("The sequence reservation exceeds the memory budget.".into());
        }
        Ok(())
    }

    fn check_resident_capacity(&self, resident: u64) -> Result<(), String> {
        let required = resident
            .checked_add(self.sequence_reservation)
            .ok_or("The model and sequence reservation overflow the byte count.")?;
        if required > self.memory_budget {
            return Err("The model and sequence reservation exceed the memory budget.".into());
        }
        Ok(())
    }
}

fn ready_memory_totals(ready: &Value) -> Result<(u64, u64), String> {
    let memory = &ready["memory"];
    let count = |key: &str| {
        memory[key]
            .as_u64()
            .ok_or_else(|| format!("The worker memory report has no valid {key}."))
    };
    let resident = count("active_bytes")?
        .checked_add(count("cache_bytes")?)
        .ok_or("The worker resident memory exceeds the byte count.")?;
    Ok((resident, resident.max(count("peak_bytes")?)))
}

#[derive(Clone)]
pub struct Request {
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub max_tokens: usize,
    pub stops: Vec<String>,
    pub count_only: bool,
}

pub struct Begin {
    pub request_id: String,
    pub input_tokens: u64,
}
pub enum Event {
    Part(Part),
    Done {
        cause: String,
        input_tokens: u64,
        output_tokens: u64,
        matched_stop: Option<String>,
    },
    Error(ApiError),
}

struct Job {
    observation: Arc<RequestObservation>,
    request: Request,
    id: String,
    ingress: Instant,
    begin: oneshot::Sender<Result<Begin, ApiError>>,
    output: mpsc::Sender<Event>,
    terminal_output: Option<mpsc::OwnedPermit<Event>>,
    cancel: Arc<AtomicBool>,
}

pub struct Ticket {
    pub begin: oneshot::Receiver<Result<Begin, ApiError>>,
    pub output: mpsc::Receiver<Event>,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Metrics {
    observations: ObservationMetrics,
    pub queued: AtomicU64,
    pub active: AtomicU64,
    pub completed: AtomicU64,
    pub cancelled: AtomicU64,
    pub expired: AtomicU64,
    pub failed: AtomicU64,
    pub input_tokens: AtomicU64,
    pub output_tokens: AtomicU64,
    pub queue_ns: AtomicU64,
    pub elapsed_ns: AtomicU64,
    pub reserved_bytes: AtomicU64,
    pub peak_bytes: AtomicU64,
    pub worker_rotations: AtomicU64,
    pub worker_faults: AtomicU64,
}

pub struct Service {
    queue: PendingQueue,
    host_pressure: Arc<HostPressure>,
    pub ready: Value,
    ready_state: RwLock<Value>,
    stopped: watch::Receiver<Option<StopResult>>,
    pub model_id: String,
    pub available: AtomicBool,
    rotating: AtomicBool,
    pub metrics: Metrics,
    pub timeout: Duration,
    pub memory_budget: u64,
}

impl Service {
    pub fn enqueue(&self, request: Request) -> Result<Ticket, ApiError> {
        if !self.available.load(Ordering::Acquire) {
            if self.rotating.load(Ordering::Acquire) {
                return Err(ApiError::new(
                    503,
                    "model_unavailable",
                    "The model worker is rotating.",
                ));
            }
            return Err(ApiError::worker("The model worker is unavailable."));
        }
        self.check_host_pressure(false)?;
        let (begin, begin_rx) = oneshot::channel();
        let (output, output_rx) = mpsc::channel(OUTPUT_PART_CAPACITY + 1);
        // Hold one slot until settlement, even when all content slots are full.
        let terminal_output = output.clone().try_reserve_owned().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        let ingress = Instant::now();
        let job = Job {
            observation: Arc::new(RequestObservation::new(ingress, request.count_only)),
            request,
            id: uuid::Uuid::new_v4().to_string(),
            ingress,
            begin,
            output,
            terminal_output: Some(terminal_output),
            cancel: cancel.clone(),
        };
        self.queue.push(job, &self.metrics, self.timeout)?;
        Ok(Ticket {
            begin: begin_rx,
            output: output_rx,
            cancel,
        })
    }
    pub fn ready_snapshot(&self) -> Value {
        self.ready_state
            .read()
            .expect("The worker readiness lock is poisoned.")
            .clone()
    }
    pub fn admission_snapshot(&self) -> (bool, Value) {
        let snapshot = self.host_pressure.snapshot(Instant::now());
        (snapshot.allowed, snapshot.json())
    }
    fn check_host_pressure(&self, dispatch: bool) -> Result<(), ApiError> {
        self.host_pressure.check(dispatch).map_err(|reason| {
            ApiError::new(
                503,
                "capacity_unavailable",
                format!("Host memory pressure blocks admission: {reason}."),
            )
        })
    }
    pub fn shutdown(&self) {
        self.queue.close(&self.metrics, self.timeout);
        self.available.store(false, Ordering::Release);
    }
    pub async fn wait_stopped(&self) -> StopResult {
        let mut stopped = self.stopped.clone();
        loop {
            if let Some(result) = stopped.borrow_and_update().clone() {
                return result;
            }
            stopped.changed().await.map_err(|_| {
                "The coordinator stopped without confirming worker process reaping.".to_owned()
            })?;
        }
    }
    pub fn metrics_text(&self) -> String {
        let m = &self.metrics;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut observations = String::new();
        let outcomes = m.observations.append(&mut observations);
        let mut text = format!(concat!(
            "apxinf_queued_requests {}\napxinf_active_requests {}\n",
            "apxinf_requests_total{{status=\"completed\"}} {}\napxinf_requests_total{{status=\"cancelled\"}} {}\napxinf_requests_total{{status=\"failed\"}} {}\napxinf_requests_total{{status=\"expired\"}} {}\n",
            "apxinf_input_tokens_total {}\napxinf_output_tokens_total {}\napxinf_queue_seconds_sum {}\napxinf_request_seconds_sum {}\n",
            "apxinf_capacity_reserved_bytes {}\napxinf_worker_peak_bytes {}\napxinf_memory_budget_bytes {}\n",
            "apxinf_worker_rotations_total {}\napxinf_worker_faults_total {}\n"),
            get(&m.queued), get(&m.active), outcomes[0], outcomes[1], outcomes[2], outcomes[3],
            get(&m.input_tokens), get(&m.output_tokens), get(&m.queue_ns) as f64 / 1e9, get(&m.elapsed_ns) as f64 / 1e9,
            get(&m.reserved_bytes), get(&m.peak_bytes), self.memory_budget,
            get(&m.worker_rotations), get(&m.worker_faults));
        text.push_str(&observations);
        self.host_pressure.append_metrics(&mut text);
        diagnostics::append_metrics(&mut text);
        text
    }
}

struct Worker {
    child: Child,
    _diagnostics: Option<WorkerDiagnostics>,
    input: Option<ChildStdin>,
    frames: mpsc::Receiver<Result<Value, String>>,
    epoch: String,
    controls: std::collections::HashSet<String>,
    sent_commands: u64,
}

impl Worker {
    async fn send(&mut self, value: &Value) -> Result<(), ApiError> {
        let bytes = contracts::encode_frame(value)
            .map_err(|e| ApiError::new(500, "protocol_fault", e.to_string()))?;
        if self.sent_commands >= contracts::MAX_COMMANDS {
            return Err(ApiError::worker("The worker command budget is exhausted."));
        }
        self.sent_commands += 1;
        let input = self
            .input
            .as_mut()
            .ok_or_else(|| ApiError::worker("The worker input is closed."))?;
        tokio::time::timeout(SETTLEMENT_GRACE, async {
            input.write_all(&bytes).await?;
            input.flush().await
        })
        .await
        .map_err(|_| {
            ApiError::worker("The worker input pipe did not accept a command within 20 seconds.")
        })?
        .map_err(|_| ApiError::worker("The worker input pipe closed."))
    }
    fn requires_rotation(&self) -> bool {
        self.sent_commands > contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE
    }
    fn command(&self, kind: &str) -> Value {
        json!({"protocol":PROTOCOL,"kind":kind,"worker_epoch":self.epoch,"command_id":uuid::Uuid::new_v4().to_string()})
    }
    async fn receive(&mut self, timeout: Duration) -> Result<Value, ApiError> {
        let value = tokio::time::timeout(timeout, self.frames.recv())
            .await
            .map_err(|_| ApiError::worker("The worker response timed out."))?
            .ok_or_else(|| ApiError::worker("The worker output pipe closed."))?
            .map_err(ApiError::worker)?;
        if value["worker_epoch"] != self.epoch {
            return Err(ApiError::new(
                500,
                "protocol_fault",
                "The worker epoch does not match.",
            ));
        }
        Ok(value)
    }
    async fn control(
        &mut self,
        job: &Job,
        kind: &str,
        reason: &str,
        output_tokens: u64,
        cutoff: usize,
    ) -> Result<(), ApiError> {
        let mut v = self.command(kind);
        v["request_id"] = json!(job.id);
        v["attempt"] = json!(1);
        if kind == "cancel_request" {
            v["reason"] = json!(reason);
        } else {
            v["cause"] = json!(reason);
            v["output_token_count"] = json!(output_tokens);
            v["text_byte_cutoff"] = json!(cutoff);
        }
        self.send(&v).await?;
        self.controls
            .insert(v["command_id"].as_str().unwrap().to_owned());
        Ok(())
    }
    fn consume_control_reply(&mut self, value: &Value) -> Result<bool, ApiError> {
        if value["kind"] != "command_result" {
            return Ok(false);
        }
        if self
            .controls
            .remove(value["command_id"].as_str().unwrap_or(""))
        {
            if value["status"] == "error" {
                return Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The worker rejected a coordinator control command.",
                ));
            }
            return Ok(true);
        }
        Ok(false)
    }
    fn consume_idle_frame(&mut self, value: &Value) -> Result<(), ApiError> {
        if value["worker_epoch"] != self.epoch {
            return Err(ApiError::new(
                500,
                "protocol_fault",
                "The idle worker epoch does not match.",
            ));
        }
        if self.consume_control_reply(value)? {
            return Ok(());
        }
        Err(ApiError::new(
            500,
            "protocol_fault",
            "The idle worker emitted an unexpected frame.",
        ))
    }
}

fn read_worker_frames(
    stdout: tokio::process::ChildStdout,
) -> mpsc::Receiver<Result<Value, String>> {
    let (sender, frames) = mpsc::channel(64);
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut bytes = Vec::new();
            let parsed = loop {
                match reader.fill_buf().await {
                    Err(e) => break Err(format!("Worker read failed: {e}")),
                    Ok([]) => break Err("Worker stdout ended.".into()),
                    Ok(chunk) => {
                        let end = chunk.iter().position(|b| *b == b'\n').map(|p| p + 1);
                        let n = end.unwrap_or(chunk.len());
                        if bytes.len() + n > FRAME_LIMIT {
                            break Err("Worker frame exceeds its byte limit.".into());
                        }
                        bytes.extend_from_slice(&chunk[..n]);
                        reader.consume(n);
                        if end.is_some() {
                            break contracts::decode_frame(&bytes).map_err(|e| e.to_string());
                        }
                    }
                }
            };
            let failed = parsed.is_err();
            if sender.send(parsed).await.is_err() || failed {
                break;
            }
        }
    });
    frames
}

#[derive(Debug)]
enum LaunchError {
    Pressure(String),
    Worker(String),
    Stopped,
}

impl From<String> for LaunchError {
    fn from(message: String) -> Self {
        Self::Worker(message)
    }
}

impl LaunchError {
    fn message(self) -> String {
        match self {
            Self::Pressure(message) | Self::Worker(message) => message,
            Self::Stopped => "The service stopped before replacement loading.".into(),
        }
    }
}

async fn launch_worker(
    config: &Config,
    pressure: &HostPressure,
) -> Result<(Worker, Value), LaunchError> {
    pressure.require_load().map_err(LaunchError::Pressure)?;
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut command = Command::new(&config.python);
    command
        .arg("-u")
        .arg(&config.worker)
        .arg("--protocol")
        .arg(PROTOCOL)
        .arg("--worker-epoch")
        .arg(&epoch)
        .arg("--model")
        .arg(&config.model)
        .arg("--max-context")
        .arg(config.max_context.to_string())
        .arg("--max-output-tokens")
        .arg(config.max_output.to_string())
        .arg("--prefill-step-size")
        .arg(config.prefill_step_size.to_string())
        .arg("--output-batch-tokens")
        .arg(config.output_batch_tokens.to_string())
        .arg("--memory-limit-bytes")
        .arg(config.memory_budget.to_string())
        .env_clear()
        .env("PATH", "/opt/homebrew/bin:/usr/bin:/bin")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("TOKENIZERS_PARALLELISM", "false")
        .env("PYTHONNOUSERSITE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .map_err(|e| format!("Cannot start worker: {e}"))?;
    let Some(stderr) = child.stderr.take() else {
        let cleanup = stop_child(&mut child).await;
        return Err(startup_failure("Worker stderr is absent.".into(), cleanup).into());
    };
    let diagnostic_reader = WorkerDiagnostics::start(stderr, epoch.clone());
    let Some(input) = child.stdin.take() else {
        let cleanup = stop_child(&mut child).await;
        return Err(startup_failure("Worker stdin is absent.".into(), cleanup).into());
    };
    let Some(stdout) = child.stdout.take() else {
        let cleanup = stop_child(&mut child).await;
        return Err(startup_failure("Worker stdout is absent.".into(), cleanup).into());
    };
    let frames = read_worker_frames(stdout);
    let mut worker = Worker {
        child,
        _diagnostics: Some(diagnostic_reader),
        input: Some(input),
        frames,
        epoch,
        controls: Default::default(),
        sent_commands: 0,
    };
    let startup: Result<Value, String> = async {
        let ready = worker
            .receive(Duration::from_secs(180))
            .await
            .map_err(|e| e.message)?;
        if ready["kind"] != "ready" {
            return Err(format!("Worker startup failed: {ready}"));
        }
        if ready["model_path"].as_str() != config.model.to_str() {
            return Err("Worker model path does not match the approved path.".into());
        }
        if ready["capabilities"]["max_context"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > config.max_context as u64)
            || ready["capabilities"]["max_output_tokens"].as_u64() != Some(config.max_output as u64)
        {
            return Err(
                "The worker capability limits differ from the launch configuration.".into(),
            );
        }
        verify_manifest(&ready["model_manifest"], &config)?;
        let (resident, _) = ready_memory_totals(&ready)?;
        config.check_resident_capacity(resident)?;
        Ok(ready)
    }
    .await;
    match startup {
        Ok(ready) => {
            check_loaded_pressure(&mut worker, pressure).await?;
            Ok((worker, ready))
        }
        Err(error) => Err(startup_failure(error, stop_child(&mut worker.child).await).into()),
    }
}

async fn check_loaded_pressure(
    worker: &mut Worker,
    pressure: &HostPressure,
) -> Result<(), LaunchError> {
    if let Err(message) = pressure.require_load() {
        return match stop_child(&mut worker.child).await {
            Ok(()) => Err(LaunchError::Pressure(message)),
            Err(cleanup) => Err(LaunchError::Worker(startup_failure(message, Err(cleanup)))),
        };
    }
    Ok(())
}

async fn launch_replacement<F, Fut>(
    service: std::sync::Weak<Service>,
    mut launch: F,
) -> Result<(Worker, Value), LaunchError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(Worker, Value), LaunchError>>,
{
    loop {
        let Some(shared) = service.upgrade() else {
            return Err(LaunchError::Stopped);
        };
        if shared.queue.is_closed() {
            return Err(LaunchError::Stopped);
        }
        let allowed = shared.host_pressure.snapshot(Instant::now()).allowed;
        drop(shared);
        if !allowed {
            tokio::time::sleep(QUEUE_SWEEP_INTERVAL).await;
            continue;
        }
        match launch().await {
            Err(LaunchError::Pressure(_)) => {
                tokio::time::sleep(QUEUE_SWEEP_INTERVAL).await;
            }
            result => return result,
        }
    }
}

fn startup_failure(error: String, cleanup: StopResult) -> String {
    match cleanup {
        Ok(()) => error,
        Err(cleanup) => format!("{error} {cleanup}"),
    }
}

async fn stop_child(child: &mut Child) -> StopResult {
    let _ = child.start_kill();
    tokio::time::timeout(SETTLEMENT_GRACE, child.wait())
        .await
        .map_err(|_| "Worker exit was not confirmed within 20 seconds.".to_owned())?
        .map_err(|error| format!("Worker process reaping failed: {error}"))?;
    Ok(())
}

pub async fn start(config: Config) -> Result<Arc<Service>, String> {
    config.validate_memory()?;
    diagnostics::initialize()?;
    let pressure = Arc::new(HostPressure::new(config.host_pressure_policy));
    pressure.require_load()?;
    HostPressure::monitor(&pressure);
    let (worker, ready) = launch_worker(&config, &pressure)
        .await
        .map_err(LaunchError::message)?;
    let (stopped_sender, stopped) = watch::channel(None);
    let (_, peak) = ready_memory_totals(&ready)?;
    let service = Arc::new(Service {
        queue: PendingQueue::new(config.queue_capacity),
        host_pressure: pressure.clone(),
        ready_state: RwLock::new(ready.clone()),
        stopped,
        ready,
        model_id: config.model_id.clone(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout: config.timeout,
        memory_budget: config.memory_budget,
    });
    service.metrics.peak_bytes.store(peak, Ordering::Relaxed);
    start_queue_sweeper(&service);
    let weak_service = Arc::downgrade(&service);
    tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        config.sequence_reservation,
        move || {
            let config = config.clone();
            let pressure = pressure.clone();
            let weak_service = weak_service.clone();
            async move { launch_replacement(weak_service, || launch_worker(&config, &pressure)).await }
        },
        stopped_sender,
    ));
    Ok(service)
}

fn start_queue_sweeper(service: &Arc<Service>) {
    let service = Arc::downgrade(service);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(QUEUE_SWEEP_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let Some(service) = service.upgrade() else {
                break;
            };
            if service.queue.is_closed() {
                break;
            }
            service.queue.sweep(&service.metrics, service.timeout);
        }
    });
}

async fn fence_worker(worker: &mut Worker, service: &Service, error: ApiError) {
    service.available.store(false, Ordering::Release);
    service.rotating.store(false, Ordering::Release);
    service.host_pressure.stop();
    service
        .metrics
        .worker_faults
        .fetch_add(1, Ordering::Relaxed);
    service.queue.close(&service.metrics, service.timeout);
    diagnostics::emit(format_args!("Worker fenced: {}", error.message));
    if stop_child(&mut worker.child).await.is_ok() {
        service.metrics.active.store(0, Ordering::Relaxed);
        service.metrics.reserved_bytes.store(0, Ordering::Relaxed);
    } else {
        diagnostics::emit(format_args!(
            "Worker exit is unconfirmed. Capacity remains reserved."
        ));
    }
}

async fn coordinate_and_report<F, Fut, E>(
    worker: Worker,
    service: Arc<Service>,
    sequence_reservation: u64,
    launch: F,
    stopped: watch::Sender<Option<StopResult>>,
) where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(Worker, Value), E>>,
    E: Into<LaunchError>,
{
    let pressure = service.host_pressure.clone();
    let result = coordinate(worker, service, sequence_reservation, launch).await;
    pressure.stop();
    stopped.send_replace(Some(result));
}

async fn coordinate<F, Fut, E>(
    mut worker: Worker,
    shared: Arc<Service>,
    sequence_reservation: u64,
    mut launch: F,
) -> StopResult
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(Worker, Value), E>>,
    E: Into<LaunchError>,
{
    let mut fault = None;
    let mut pending_recovery: Option<Arc<RequestObservation>> = None;
    loop {
        if shared.queue.is_closed() {
            break;
        }
        if worker.requires_rotation() {
            match rotate_worker(&mut worker, &shared, &mut launch).await {
                Ok(Rotation::Ready) => {}
                Ok(Rotation::Stopped) => break,
                Err(error) => {
                    fault = Some(error.message.clone());
                    fence_worker(&mut worker, &shared, error).await;
                    break;
                }
            }
        }
        let job = tokio::select! {
            biased;
            frame = worker.frames.recv() => {
                let result = match frame {
                    Some(Ok(frame)) => worker.consume_idle_frame(&frame),
                    Some(Err(message)) => Err(ApiError::worker(message)),
                    None => Err(ApiError::worker("The idle worker output pipe closed.")),
                };
                if let Err(error) = result {
                    fault = Some(error.message.clone());
                    fence_worker(&mut worker, &shared, error).await;
                    break;
                }
                continue;
            }
            job = shared.queue.recv(&shared.metrics) => match job {
                Some(job) => job,
                None => break,
            },
        };
        let mut job = job;
        // Only this coordinator sends commands; rotation precedes every queue receive.
        debug_assert!(!worker.requires_rotation());
        if let Some(error) = unstarted_error(&job, shared.timeout) {
            job.observation
                .settled(&shared.metrics.observations, Instant::now());
            finish_before_execution(&mut job, &shared, error);
            continue;
        }
        if let Err(error) = shared.check_host_pressure(true) {
            job.observation
                .settled(&shared.metrics.observations, Instant::now());
            finish_before_execution(&mut job, &shared, error);
            continue;
        }
        shared
            .metrics
            .queue_ns
            .fetch_add(job.ingress.elapsed().as_nanos() as u64, Ordering::Relaxed);
        shared.metrics.active.store(1, Ordering::Relaxed);
        shared
            .metrics
            .reserved_bytes
            .store(sequence_reservation, Ordering::Relaxed);
        let observation = job.observation.clone();
        let result = execute(&mut worker, &shared, job, shared.timeout).await;
        if let Err(failure) = result {
            fault = Some(failure.error.message.clone());
            if !failure.reported {
                shared.metrics.failed.fetch_add(1, Ordering::Relaxed);
                observation.public_terminal(
                    &shared.metrics.observations,
                    Outcome::Failed,
                    Instant::now(),
                );
            }
            pending_recovery = Some(observation);
            fence_worker(&mut worker, &shared, failure.error).await;
            break;
        }
        shared.metrics.active.store(0, Ordering::Relaxed);
        shared.metrics.reserved_bytes.store(0, Ordering::Relaxed);
        observation.settled(&shared.metrics.observations, Instant::now());
    }
    // A closed coordinator must not leave its owned child running.
    stop_child(&mut worker.child).await?;
    if let Some(observation) = pending_recovery {
        observation.settled(&shared.metrics.observations, Instant::now());
    }
    shared.metrics.active.store(0, Ordering::Relaxed);
    shared.metrics.reserved_bytes.store(0, Ordering::Relaxed);
    fault.map_or(Ok(()), Err)
}

#[derive(Debug, PartialEq, Eq)]
enum Rotation {
    Ready,
    Stopped,
}

async fn rotate_worker<F, Fut, E>(
    worker: &mut Worker,
    service: &Service,
    launch: &mut F,
) -> Result<Rotation, ApiError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<(Worker, Value), E>>,
    E: Into<LaunchError>,
{
    service.rotating.store(true, Ordering::Release);
    service.available.store(false, Ordering::Release);
    let old_epoch = worker.epoch.clone();
    let shutdown = worker.command("shutdown");
    worker.send(&shutdown).await?;
    worker
        .controls
        .insert(shutdown["command_id"].as_str().unwrap().to_owned());
    // No command can follow shutdown. Closing the pipe also releases the input reader.
    drop(worker.input.take());
    let started = Instant::now();
    loop {
        let frame = worker
            .receive(SETTLEMENT_GRACE.saturating_sub(started.elapsed()))
            .await?;
        if worker.consume_control_reply(&frame)? {
            continue;
        }
        if frame["kind"] == "drained" && worker.controls.is_empty() {
            break;
        }
        return Err(ApiError::new(
            500,
            "protocol_fault",
            "The worker did not drain in protocol order.",
        ));
    }
    let status = tokio::time::timeout(
        SETTLEMENT_GRACE.saturating_sub(started.elapsed()),
        worker.child.wait(),
    )
    .await
    .map_err(|_| ApiError::worker("The drained worker did not exit."))?
    .map_err(|_| ApiError::worker("The drained worker could not be reaped."))?;
    if !status.success() {
        return Err(ApiError::worker("The drained worker exited with an error."));
    }
    if service.queue.is_closed() {
        service.rotating.store(false, Ordering::Release);
        return Ok(Rotation::Stopped);
    }
    let (mut replacement, ready) = match launch().await.map_err(Into::into) {
        Ok(loaded) => loaded,
        Err(LaunchError::Stopped) => {
            service.rotating.store(false, Ordering::Release);
            return Ok(Rotation::Stopped);
        }
        Err(error) => return Err(ApiError::worker(error.message())),
    };
    if replacement.epoch == old_epoch
        || ready["worker_epoch"] != replacement.epoch
        || ready["model_revision"] != service.ready["model_revision"]
        || ready["capability_revision"] != service.ready["capability_revision"]
    {
        let message = startup_failure(
            "The replacement worker identity differs from the approved profile.".into(),
            stop_child(&mut replacement.child).await,
        );
        return Err(ApiError::worker(message));
    }
    let (_, peak) = match ready_memory_totals(&ready) {
        Ok(memory) => memory,
        Err(error) => {
            return Err(ApiError::worker(startup_failure(
                error,
                stop_child(&mut replacement.child).await,
            )));
        }
    };
    service
        .metrics
        .peak_bytes
        .fetch_max(peak, Ordering::Relaxed);
    *service
        .ready_state
        .write()
        .expect("The worker readiness lock is poisoned.") = ready;
    *worker = replacement;
    service
        .metrics
        .worker_rotations
        .fetch_add(1, Ordering::Relaxed);
    service.rotating.store(false, Ordering::Release);
    service.queue.publish_availability(&service.available);
    diagnostics::emit(format_args!(
        "{}",
        json!({"event":"worker_rotated", "previous_epoch":old_epoch, "worker_epoch":worker.epoch})
    ));
    Ok(Rotation::Ready)
}

#[derive(Debug)]
struct ExecutionFailure {
    error: ApiError,
    reported: bool,
}

impl From<ApiError> for ExecutionFailure {
    fn from(error: ApiError) -> Self {
        Self {
            error,
            reported: false,
        }
    }
}

fn unstarted_error(job: &Job, timeout: Duration) -> Option<ApiError> {
    if job.ingress.elapsed() >= timeout {
        Some(ApiError::new(
            504,
            "deadline_exceeded",
            "The request expired before execution.",
        ))
    } else if job.cancel.load(Ordering::Acquire) || job.begin.is_closed() || job.output.is_closed()
    {
        Some(ApiError::new(
            499,
            "cancelled",
            "The request was cancelled before execution.",
        ))
    } else {
        None
    }
}

fn finish_before_execution(job: &mut Job, service: &Service, error: ApiError) {
    let outcome = match error.code.as_str() {
        "deadline_exceeded" => Outcome::Expired,
        "cancelled" => Outcome::Cancelled,
        _ => Outcome::Failed,
    };
    job.observation
        .public_terminal(&service.metrics.observations, outcome, Instant::now());
    match error.code.as_str() {
        "deadline_exceeded" => {
            service.metrics.expired.fetch_add(1, Ordering::Relaxed);
        }
        "cancelled" => {
            service.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        _ => {
            service.metrics.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
    let (unused, _) = oneshot::channel();
    let _ = std::mem::replace(&mut job.begin, unused).send(Err(error));
}

async fn prepare_request(
    worker: &mut Worker,
    service: &Service,
    job: &mut Job,
    timeout: Duration,
) -> Result<Option<Value>, ExecutionFailure> {
    job.observation.preparation_start(Instant::now());
    let mut prepare = worker.command("prepare_input");
    prepare["model_revision"] = service.ready["model_revision"].clone();
    prepare["messages"] = json!(job.request.messages);
    prepare["tools"] = json!(job.request.tools);
    prepare["template_options"] = json!({"enable_thinking":false});
    let mut stopped: Option<Instant> = None;
    wait_for_preparation(worker.send(&prepare), service, job, timeout, &mut stopped).await?;
    loop {
        let event = wait_for_preparation(
            async {
                worker
                    .frames
                    .recv()
                    .await
                    .ok_or_else(|| {
                        ApiError::worker("The worker output closed during preparation.")
                    })?
                    .map_err(ApiError::worker)
            },
            service,
            job,
            timeout,
            &mut stopped,
        )
        .await?;
        let result = (|| {
            if event["worker_epoch"] != worker.epoch {
                return Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The worker epoch changed during preparation.",
                ));
            }
            if worker.consume_control_reply(&event)? {
                return Ok(false);
            }
            if event["command_id"] != prepare["command_id"] {
                return Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The preparation response has a different command ID.",
                ));
            }
            match event["kind"].as_str() {
                Some("prepared_input")
                    if event["model_revision"] == service.ready["model_revision"] =>
                {
                    Ok(true)
                }
                Some("command_result") if event["status"] == "error" => Ok(true),
                _ => Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The worker returned an unexpected preparation event.",
                )),
            }
        })()
        .map_err(|error| ExecutionFailure {
            error,
            reported: stopped.is_some(),
        })?;
        if !result {
            continue;
        }
        job.observation
            .preparation_end(&service.metrics.observations, Instant::now());
        // A ready response and the public deadline can become observable together.
        if stopped.is_none() {
            if let Some(error) = unstarted_error(job, timeout) {
                finish_before_execution(job, service, error);
                stopped = Some(Instant::now());
            }
        }
        if stopped.is_some() {
            return Ok(None);
        }
        if event["kind"] == "command_result" {
            finish_before_execution(job, service, error_from_frame(&event));
            return Ok(None);
        }
        return Ok(Some(event));
    }
}

async fn wait_for_preparation<F, T>(
    operation: F,
    service: &Service,
    job: &mut Job,
    timeout: Duration,
    stopped: &mut Option<Instant>,
) -> Result<T, ExecutionFailure>
where
    F: Future<Output = Result<T, ApiError>>,
{
    tokio::pin!(operation);
    loop {
        if stopped.is_none() {
            if let Some(error) = unstarted_error(job, timeout) {
                finish_before_execution(job, service, error);
                *stopped = Some(Instant::now());
            }
        }
        if stopped.is_some_and(|at| at.elapsed() >= SETTLEMENT_GRACE) {
            return Err(ExecutionFailure {
                error: ApiError::worker("Preparation did not settle within 20 seconds."),
                reported: true,
            });
        }
        let wait = stopped
            .map_or_else(
                || timeout.saturating_sub(job.ingress.elapsed()),
                |at| SETTLEMENT_GRACE.saturating_sub(at.elapsed()),
            )
            .min(QUEUE_SWEEP_INTERVAL);
        tokio::select! {
            result = &mut operation => return result.map_err(|error| ExecutionFailure { error, reported: stopped.is_some() }),
            _ = tokio::time::sleep(wait) => {},
        }
    }
}

#[derive(Clone, Copy)]
struct StoppedSubmission {
    at: Instant,
    reason: &'static str,
}

async fn send_submission(
    worker: &mut Worker,
    service: &Service,
    job: &mut Job,
    timeout: Duration,
    submit: &Value,
) -> Result<Option<StoppedSubmission>, ExecutionFailure> {
    let operation = worker.send(submit);
    tokio::pin!(operation);
    let mut stopped: Option<StoppedSubmission> = None;
    loop {
        if stopped.is_none() {
            stopped = stop_submission_if_needed(service, job, timeout);
        }
        if stopped.is_some_and(|stop| stop.at.elapsed() >= SETTLEMENT_GRACE) {
            return Err(ExecutionFailure {
                error: ApiError::worker("Submission did not settle within 20 seconds."),
                reported: true,
            });
        }
        let wait = stopped
            .map_or_else(
                || timeout.saturating_sub(job.ingress.elapsed()),
                |stop| SETTLEMENT_GRACE.saturating_sub(stop.at.elapsed()),
            )
            .min(QUEUE_SWEEP_INTERVAL);
        tokio::select! {
            result = &mut operation => {
                // Completion of the write can coincide with the public deadline.
                if stopped.is_none() {
                    stopped = stop_submission_if_needed(service, job, timeout);
                }
                result.map_err(|error| ExecutionFailure { error, reported: stopped.is_some() })?;
                return Ok(stopped);
            }
            _ = tokio::time::sleep(wait) => {}
        }
    }
}

fn stop_submission_if_needed(
    service: &Service,
    job: &mut Job,
    timeout: Duration,
) -> Option<StoppedSubmission> {
    let error = unstarted_error(job, timeout)?;
    let reason = if error.code == "deadline_exceeded" {
        "deadline_exceeded"
    } else {
        "client_disconnect"
    };
    let at = Instant::now();
    finish_before_execution(job, service, error);
    Some(StoppedSubmission { at, reason })
}

async fn settle_stopped_submission(
    worker: &mut Worker,
    service: &Service,
    job: &Job,
    mut tracker: contracts::AttemptTracker,
    stopped: StoppedSubmission,
) -> Result<(), ExecutionFailure> {
    let settle = async {
        let mut control_sent = false;
        loop {
            let frame = worker
                .frames
                .recv()
                .await
                .ok_or_else(|| {
                    ApiError::worker("The worker output closed during submission settlement.")
                })?
                .map_err(ApiError::worker)?;
            if frame["worker_epoch"] != worker.epoch {
                return Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The worker epoch changed during submission settlement.",
                ));
            }
            if worker.consume_control_reply(&frame)? {
                continue;
            }
            tracker
                .observe(&frame)
                .map_err(|error| ApiError::new(500, "protocol_fault", error.to_string()))?;
            observe_worker_frame(job, service, &frame);
            match frame["kind"].as_str() {
                Some("accepted") if !control_sent => {
                    // Controls can bypass submit registration in the worker mailbox.
                    worker
                        .control(job, "cancel_request", stopped.reason, 0, 0)
                        .await?;
                    control_sent = true;
                }
                Some("resources_released") => return Ok(()),
                Some("tokens" | "terminal" | "rejected" | "prefill_progress") => {}
                _ => {
                    return Err(ApiError::new(
                        500,
                        "protocol_fault",
                        "The worker emitted an unsupported submission settlement event.",
                    ));
                }
            }
        }
    };
    tokio::time::timeout(
        SETTLEMENT_GRACE.saturating_sub(stopped.at.elapsed()),
        settle,
    )
    .await
    .map_err(|_| ExecutionFailure {
        error: ApiError::worker("Submission did not settle within 20 seconds."),
        reported: true,
    })?
    .map_err(|error| ExecutionFailure {
        error,
        reported: true,
    })
}

fn file_digest(path: &std::path::Path) -> Result<(u64, String), String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() {
        return Err("An identity artifact is not a regular file.".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("An identity artifact changed during inspection.".into());
    }
    Ok((before.len(), format!("{:x}", hash.finalize())))
}

fn verify_manifest(manifest: &Value, config: &Config) -> Result<(), String> {
    let mut names = std::fs::read_dir(&config.model)
        .map_err(|e| e.to_string())?
        .map(|item| {
            item.map_err(|e| e.to_string()).and_then(|e| {
                e.file_name()
                    .into_string()
                    .map_err(|_| "A model artifact name is not UTF-8.".into())
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    names.sort();
    let artifacts = manifest["artifacts"]
        .as_array()
        .ok_or("The model manifest has no artifacts.")?;
    if artifacts.len() != names.len() {
        return Err("The model artifact count differs from the manifest.".into());
    }
    for (name, record) in names.iter().zip(artifacts) {
        if record["path"].as_str() != Some(name) {
            return Err("The model artifact list differs from the manifest.".into());
        }
        let path = config.model.join(name);
        let resolved = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
        let (size, digest) = file_digest(&resolved)?;
        if record["size_bytes"].as_u64() != Some(size)
            || record["sha256"].as_str() != Some(&digest)
            || std::fs::canonicalize(&path).map_err(|e| e.to_string())? != resolved
        {
            return Err(format!("The model artifact identity differs: {name}"));
        }
    }
    let root = config
        .worker
        .ancestors()
        .nth(5)
        .ok_or("Cannot locate the runtime pin helper.")?;
    let files = [
        ("adapter_sha256", config.worker.clone()),
        ("python_sha256", config.python.clone()),
        (
            "contracts_sha256",
            config.worker.with_file_name("contracts.py"),
        ),
        (
            "runtime_pins_sha256",
            root.join("scripts/apxinf_mlx_generate.py"),
        ),
    ];
    for (field, path) in files {
        let (_, digest) = file_digest(&path)?;
        if manifest[field].as_str() != Some(&digest) {
            return Err(format!("The runtime identity differs: {field}"));
        }
    }
    let execution = &manifest["execution"];
    if execution["provider"] != "mlx-lm"
        || execution["precision"] != "bundle"
        || execution["prefill_step_size"].as_u64() != Some(config.prefill_step_size as u64)
        || execution["output_batch_tokens"].as_u64() != Some(config.output_batch_tokens as u64)
        || execution["memory_limit_bytes"].as_u64() != Some(config.memory_budget)
    {
        return Err("The execution profile differs from the launch configuration.".into());
    }
    Ok(())
}

async fn execute(
    worker: &mut Worker,
    service: &Service,
    mut job: Job,
    timeout: Duration,
) -> Result<(), ExecutionFailure> {
    let Some(prepared) = prepare_request(worker, service, &mut job, timeout).await? else {
        return Ok(());
    };
    let input_tokens = prepared["effective_input_tokens"]
        .as_u64()
        .ok_or_else(|| ApiError::worker("Prepared input has no token count."))?;
    if input_tokens + job.request.max_tokens as u64
        > service.ready["capabilities"]["max_context"]
            .as_u64()
            .unwrap_or(0)
    {
        finish_before_execution(
            &mut job,
            service,
            ApiError::new(
                400,
                "context_limit",
                "Input and output reservation exceed the model context limit.",
            ),
        );
        return Ok(());
    }
    let begin = Begin {
        request_id: job.id.clone(),
        input_tokens,
    };
    if job.request.count_only {
        service.metrics.completed.fetch_add(1, Ordering::Relaxed);
        job.observation.public_terminal(
            &service.metrics.observations,
            Outcome::Completed,
            Instant::now(),
        );
        let _ = job.begin.send(Ok(begin));
        return Ok(());
    }
    if let Some(error) = unstarted_error(&job, timeout) {
        finish_before_execution(&mut job, service, error);
        return Ok(());
    }
    if let Err(error) = service.check_host_pressure(true) {
        finish_before_execution(&mut job, service, error);
        return Ok(());
    }
    let mut submit = worker.command("submit");
    submit["request_id"] = json!(job.id);
    submit["attempt"] = json!(1);
    submit["model_revision"] = service.ready["model_revision"].clone();
    submit["capability_revision"] = service.ready["capability_revision"].clone();
    submit["token_ids"] = prepared["token_ids"].clone();
    submit["max_tokens"] = json!(job.request.max_tokens);
    submit["remaining_timeout_ms"] = json!(timeout
        .saturating_sub(job.ingress.elapsed())
        .as_millis()
        .max(1) as u64);
    submit["eos_token_ids"] = service.ready["eos_token_ids"].clone();
    submit["capacity_lease_ids"] = json!([uuid::Uuid::new_v4().to_string()]);
    let mut tracker = contracts::AttemptTracker::new(&submit)
        .map_err(|e| ApiError::new(500, "protocol_fault", e.to_string()))?;
    if let Some(stopped) = send_submission(worker, service, &mut job, timeout, &submit).await? {
        return settle_stopped_submission(worker, service, &job, tracker, stopped).await;
    }
    let mut public = GenerationPublic::new(&mut job, begin);
    let mut parser = OutputParser::new(job.request.tools.clone());
    let mut stops = StopMatcher::new(job.request.stops.clone());
    let mut output_tokens = 0;
    let mut control_sent = false;
    let mut cancelled_at = None;
    let mut terminal = None;
    let mut public_error: Option<ApiError> = None;
    let mut consumer_stop: Option<&'static str> = None;
    let mut total_text_bytes = 0usize;
    let result: Result<(), ApiError> = async {
    loop {
        public.expire(&job, service, timeout, public_error.as_ref(), terminal.as_ref());
        if let Some(at) = public.ended_at {
            cancelled_at.get_or_insert(at);
        }
        if job.cancel.load(Ordering::Acquire) || job.output.is_closed() {
            consumer_stop.get_or_insert("client_disconnect");
        }
        if consumer_stop.is_some() {
            cancelled_at.get_or_insert_with(Instant::now);
        }
        if !control_sent && public.accepted && terminal.is_none() {
            let reason = if job.ingress.elapsed() >= timeout {
                Some("deadline_exceeded")
            } else if consumer_stop.is_some() {
                consumer_stop
            } else {
                None
            };
            if let Some(reason) = reason {
                let at = *cancelled_at.get_or_insert_with(Instant::now);
                wait_for_generation_control(
                    worker.control(&job, "cancel_request", reason, output_tokens, 0),
                    &job, service, &mut public, timeout, at, public_error.as_ref(), terminal.as_ref(),
                ).await?;
                control_sent = true;
            }
        }
        if cancelled_at.is_some_and(|t: Instant| t.elapsed() >= SETTLEMENT_GRACE) {
            return Err(ApiError::worker(
                "The worker did not settle cancellation within 20 seconds.",
            )
            .into());
        }
        let event = tokio::select! {
            event = worker.frames.recv() => event,
            _ = tokio::time::sleep(public.next_wait(&job, timeout, cancelled_at)) => continue,
        };
        // A frame and the original public deadline can become observable together.
        public.expire(&job, service, timeout, public_error.as_ref(), terminal.as_ref());
        let frame = event.ok_or_else(|| ApiError::worker("The worker output closed during execution."))?
            .map_err(ApiError::worker)?;
        if frame["worker_epoch"] != worker.epoch {
            return Err(ApiError::new(
                500,
                "protocol_fault",
                "The worker epoch changed during execution.",
            )
            .into());
        }
        let kind = frame["kind"].as_str().unwrap_or("");
        if worker.consume_control_reply(&frame)? {
            continue;
        }
        tracker
            .observe(&frame)
            .map_err(|e| ApiError::new(500, "protocol_fault", e.to_string()))?;
        observe_worker_frame(&job, service, &frame);
        match kind {
            "tokens" | "terminal" => {
                if kind == "tokens" {
                    output_tokens += frame["token_ids"].as_array().map_or(0, |v| v.len()) as u64;
                }
                total_text_bytes += frame["text_delta"].as_str().unwrap_or("").len();
                if total_text_bytes > 4 * 1024 * 1024 {
                    public_error = Some(ApiError::new(
                        500,
                        "internal_error",
                        "The output exceeded its byte limit.",
                    ));
                }
                if public_error.is_none() && public.status.is_none() {
                    let text = stops.push(
                        frame["text_delta"].as_str().unwrap_or(""),
                        kind == "terminal",
                    );
                    match parser.push(&text, kind == "terminal" && frame["status"] == "completed") {
                        Ok(parts) => {
                            public.expire(&job, service, timeout, public_error.as_ref(),
                                if kind == "terminal" { Some(&frame) } else { terminal.as_ref() });
                            if consumer_stop.is_none() && public.status.is_none() {
                                for part in parts {
                                    public.expire(&job, service, timeout, public_error.as_ref(),
                                        if kind == "terminal" { Some(&frame) } else { terminal.as_ref() });
                                    if public.status.is_some() {
                                        break;
                                    }
                                    let visible = match &part {
                                        Part::Text(text) => !text.is_empty(),
                                        Part::Tool { .. } => true,
                                    };
                                    if let Some(reason) = deliver_part(&job.output, part) {
                                        consumer_stop = Some(reason);
                                        break;
                                    }
                                    if visible {
                                        job.observation.public_output(
                                            &service.metrics.observations,
                                            Instant::now(),
                                        );
                                    }
                                }
                            }
                        }
                        Err(message) => {
                            public_error = Some(ApiError::new(500, "internal_error", message))
                        }
                    }
                }
                if kind == "terminal" {
                    terminal = Some(frame);
                } else if !control_sent && public.status.is_none() {
                    if public_error.is_some() || consumer_stop.is_some() {
                        let at = *cancelled_at.get_or_insert_with(Instant::now);
                        wait_for_generation_control(worker.control(
                                &job,
                                "cancel_request",
                                consumer_stop.unwrap_or("slow_consumer"),
                                output_tokens,
                                0,
                            ), &job, service, &mut public, timeout, at, public_error.as_ref(), terminal.as_ref()).await?;
                        control_sent = true;
                    } else if stops.matched.is_some() {
                        let at = *cancelled_at.get_or_insert_with(Instant::now);
                        wait_for_generation_control(worker.control(
                                &job,
                                "stop_generation",
                                "stop_sequence",
                                output_tokens,
                                stops.cutoff,
                            ), &job, service, &mut public, timeout, at, public_error.as_ref(), terminal.as_ref()).await?;
                        control_sent = true;
                    }
                }
            }
            "resources_released" => break,
            "accepted" => {
                public.accept(&job);
            }
            "rejected" => {
                public.finish(&job, service, "failed", Event::Error(error_from_frame(&frame)));
            }
            "prefill_progress" => {}
            _ => {
                return Err(ApiError::new(
                    500,
                    "protocol_fault",
                    "The worker emitted an unsupported request event.",
                )
                .into())
            }
        }
    }
    let terminal = terminal.ok_or_else(|| {
        ApiError::new(
            500,
            "protocol_fault",
            "Cleanup preceded the terminal result.",
        )
    })?;
    let output_tokens = terminal["usage"]["output_tokens"].as_u64().unwrap_or(0);
    service
        .metrics
        .input_tokens
        .fetch_add(input_tokens, Ordering::Relaxed);
    service
        .metrics
        .output_tokens
        .fetch_add(output_tokens, Ordering::Relaxed);
    service
        .metrics
        .elapsed_ns
        .fetch_add(job.ingress.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if job.cancel.load(Ordering::Acquire) || job.output.is_closed() {
        consumer_stop.get_or_insert("client_disconnect");
    }
    public.expire(&job, service, timeout, public_error.as_ref(), Some(&terminal));
    let cause = if stops.matched.is_some() {
        "stop_sequence"
    } else if parser.tool_count > 0 {
        "tool_calls"
    } else {
        terminal["cause"].as_str().unwrap_or("eos")
    };
    if public.status.is_none() {
    let (status, result) = public_result(
        &terminal,
        public_error,
        consumer_stop,
        Event::Done {
            cause: cause.into(),
            input_tokens,
            output_tokens,
            matched_stop: stops.matched,
        },
    );
    public.finish(&job, service, status, result);
    }
    diagnostics::emit(format_args!(
        "{}",
        json!({"event":"request_settled","request_id":job.id,"status":public.status,"worker_status":terminal["status"],"consumer_stop":consumer_stop,"input_tokens":input_tokens,"output_tokens":output_tokens,"elapsed_ms":job.ingress.elapsed().as_millis(),"worker_metrics":terminal["metrics"]})
    ));
    Ok(())
    }.await;
    result.map_err(|error| ExecutionFailure {
        error,
        reported: public.status.is_some(),
    })
}

fn observe_worker_frame(job: &Job, service: &Service, frame: &Value) {
    let now = Instant::now();
    match frame["kind"].as_str() {
        Some("tokens") => job
            .observation
            .worker_output(&service.metrics.observations, now),
        Some("terminal") => {
            service.metrics.peak_bytes.fetch_max(
                frame["metrics"]["peak_memory_bytes"].as_u64().unwrap_or(0),
                Ordering::Relaxed,
            );
            let first_token_ns = (frame["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0)
                .then(|| frame["metrics"]["ttft_ns"].as_u64())
                .flatten();
            job.observation
                .worker_terminal(&service.metrics.observations, now, first_token_ns);
        }
        Some("resources_released") => job.observation.settled(&service.metrics.observations, now),
        _ => {}
    }
}

fn deliver_part(output: &mpsc::Sender<Event>, part: Part) -> Option<&'static str> {
    match output.try_send(Event::Part(part)) {
        Ok(()) => None,
        Err(mpsc::error::TrySendError::Closed(_)) => Some("client_disconnect"),
        Err(mpsc::error::TrySendError::Full(_)) => Some("slow_consumer"),
    }
}

fn public_result(
    terminal: &Value,
    public_error: Option<ApiError>,
    consumer_stop: Option<&str>,
    completed: Event,
) -> (&'static str, Event) {
    if let Some(error) = public_error {
        return ("failed", Event::Error(error));
    }
    match terminal["status"].as_str() {
        Some("failed") => ("failed", Event::Error(error_from_frame(terminal))),
        Some("expired") => (
            "expired",
            Event::Error(ApiError::new(
                504,
                "deadline_exceeded",
                "The request expired.",
            )),
        ),
        Some("cancelled") => (
            "cancelled",
            Event::Error(ApiError::new(
                499,
                "cancelled",
                "The request was cancelled.",
            )),
        ),
        Some("completed") if consumer_stop.is_some() => (
            "cancelled",
            Event::Error(ApiError::new(
                499,
                "cancelled",
                "The client stopped consuming output.",
            )),
        ),
        Some("completed") => ("completed", completed),
        _ => (
            "failed",
            Event::Error(ApiError::new(
                500,
                "protocol_fault",
                "The worker terminal status is invalid.",
            )),
        ),
    }
}

fn publish_result(
    metrics: &Metrics,
    terminal_output: mpsc::OwnedPermit<Event>,
    status: &str,
    result: Event,
) {
    record_public_status(metrics, status);
    // The permit guarantees final-event capacity. A closed receiver needs no reply.
    terminal_output.send(result);
}

fn record_public_status(metrics: &Metrics, status: &str) {
    match status {
        "completed" => {
            metrics.completed.fetch_add(1, Ordering::Relaxed);
        }
        "cancelled" => {
            metrics.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        "expired" => {
            metrics.expired.fetch_add(1, Ordering::Relaxed);
        }
        _ => {
            metrics.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn error_from_frame(frame: &Value) -> ApiError {
    let code = frame["error"]["code"].as_str().unwrap_or("internal_error");
    let status = match code {
        "invalid_request" | "unsupported_feature" | "context_limit" => 400,
        "model_not_found" => 404,
        "queue_full" | "quota_exceeded" => 429,
        "deadline_exceeded" => 504,
        _ => 500,
    };
    ApiError::new(
        status,
        code,
        frame["error"]["message"]
            .as_str()
            .unwrap_or("The model worker failed."),
    )
}

/// Observe gateway admission without a process, model, or device.
#[cfg(test)]
pub(crate) fn gateway_test_service() -> (Arc<Service>, mpsc::UnboundedReceiver<Request>) {
    gateway_test_service_with_pressure(Arc::new(HostPressure::new(HostPressurePolicy::Disabled)))
}

#[cfg(test)]
pub(crate) fn gateway_test_service_with_pressure(
    pressure: Arc<HostPressure>,
) -> (Arc<Service>, mpsc::UnboundedReceiver<Request>) {
    let (observed, requests) = mpsc::unbounded_channel();
    let service = Arc::new(Service {
        queue: PendingQueue::new(8),
        host_pressure: pressure,
        stopped: watch::channel(None).1,
        ready_state: RwLock::new(json!({"worker_epoch":"00000000-0000-4000-8000-000000000001"})),
        ready: json!({
            "worker_epoch": "00000000-0000-4000-8000-000000000001",
            "model_revision": "1111111111111111111111111111111111111111111111111111111111111111",
            "capabilities": {"max_context": 4096, "max_output_tokens": 1024}
        }),
        model_id: "test-model".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout: Duration::from_secs(1),
        memory_budget: 1024,
    });
    let coordinator = service.clone();
    tokio::spawn(async move {
        while let Some(job) = coordinator.queue.recv(&coordinator.metrics).await {
            let _ = observed.send(job.request);
            let _ = job.begin.send(Err(ApiError::worker(
                "Test coordinator observed admission.",
            )));
        }
    });
    (service, requests)
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    const EPOCH: &str = "00000000-0000-4000-8000-000000000001";
    const CONTROL: &str = "00000000-0000-4000-8000-000000000002";

    fn idle_service() -> Arc<Service> {
        Arc::new(Service {
            queue: PendingQueue::new(8),
            host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
            stopped: watch::channel(None).1,
            ready_state: RwLock::new(json!({"worker_epoch":EPOCH})),
            ready: json!({}),
            model_id: "test-model".into(),
            available: AtomicBool::new(true),
            rotating: AtomicBool::new(false),
            metrics: Metrics::default(),
            timeout: Duration::from_secs(1),
            memory_budget: 1024,
        })
    }

    fn idle_request() -> Request {
        Request {
            messages: vec![json!({"role": "user", "content": "hello"})],
            tools: Vec::new(),
            max_tokens: 1,
            stops: Vec::new(),
            count_only: false,
        }
    }

    fn completed_event() -> Event {
        Event::Done {
            cause: "length".into(),
            input_tokens: 1,
            output_tokens: 64,
            matched_stop: None,
        }
    }

    #[tokio::test]
    async fn final_result_survives_a_full_content_queue_before_the_client_reads() {
        let service = idle_service();
        let mut ticket = service.enqueue(idle_request()).unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        for index in 0..OUTPUT_PART_CAPACITY {
            assert_eq!(
                deliver_part(&job.output, Part::Text(index.to_string())),
                None
            );
        }
        assert_eq!(job.output.capacity(), 0);
        publish_result(
            &service.metrics,
            job.terminal_output.unwrap(),
            "completed",
            completed_event(),
        );
        // Reading begins only after content and the final result are produced.
        for index in 0..OUTPUT_PART_CAPACITY {
            match ticket.output.recv().await.unwrap() {
                Event::Part(Part::Text(text)) => assert_eq!(text, index.to_string()),
                _ => panic!("Content order changed."),
            }
        }
        assert!(matches!(
            ticket.output.recv().await,
            Some(Event::Done { .. })
        ));
        drop(job.output);
        assert!(ticket.output.recv().await.is_none());
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn disconnect_with_in_flight_output_counts_as_cancelled() {
        for worker_status in ["cancelled", "completed"] {
            let service = idle_service();
            let ticket = service.enqueue(idle_request()).unwrap();
            let job = service.queue.recv(&service.metrics).await.unwrap();
            drop(ticket.output);
            let reason = deliver_part(&job.output, Part::Text("in-flight text".into()));
            assert_eq!(reason, Some("client_disconnect"));
            let (status, result) = public_result(
                &json!({"status": worker_status}),
                None,
                reason,
                completed_event(),
            );
            assert_eq!(status, "cancelled");
            assert!(matches!(&result, Event::Error(error) if error.code == "cancelled"));
            publish_result(
                &service.metrics,
                job.terminal_output.unwrap(),
                status,
                result,
            );
            assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 1);
            assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
            assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
        }
    }

    #[tokio::test]
    async fn a_slow_consumer_receives_cancellation_after_buffered_content() {
        let service = idle_service();
        let mut ticket = service.enqueue(idle_request()).unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        for _ in 0..OUTPUT_PART_CAPACITY {
            assert_eq!(
                deliver_part(&job.output, Part::Text("buffered".into())),
                None
            );
        }
        let reason = deliver_part(&job.output, Part::Text("cannot fit".into()));
        assert_eq!(reason, Some("slow_consumer"));
        let (status, result) = public_result(
            &json!({"status": "completed"}),
            None,
            reason,
            completed_event(),
        );
        publish_result(
            &service.metrics,
            job.terminal_output.unwrap(),
            status,
            result,
        );
        for _ in 0..OUTPUT_PART_CAPACITY {
            assert!(matches!(ticket.output.recv().await, Some(Event::Part(_))));
        }
        assert!(
            matches!(ticket.output.recv().await, Some(Event::Error(error)) if error.code == "cancelled")
        );
        assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn parser_and_worker_faults_remain_failures_after_disconnect() {
        let service = idle_service();
        let ticket = service.enqueue(idle_request()).unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        drop(ticket.output);
        let reason = deliver_part(&job.output, Part::Text("in-flight text".into()));
        let parser_error = OutputParser::new(Vec::new())
            .push(
                "<tool_call>{\"name\":\"undeclared\",\"arguments\":{}}</tool_call>",
                true,
            )
            .unwrap_err();
        let (status, result) = public_result(
            &json!({"status": "cancelled"}),
            Some(ApiError::new(500, "internal_error", parser_error)),
            reason,
            completed_event(),
        );
        assert_eq!(status, "failed");
        assert!(matches!(&result, Event::Error(error) if error.message.contains("undeclared")));
        publish_result(
            &service.metrics,
            job.terminal_output.unwrap(),
            status,
            result,
        );
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 0);

        let terminal = json!({"status": "failed", "error": {"code": "protocol_fault", "message": "Protocol failed."}});
        let (status, result) = public_result(&terminal, None, reason, completed_event());
        assert_eq!(status, "failed");
        assert!(matches!(result, Event::Error(error) if error.code == "protocol_fault"));
        let (status, result) = public_result(
            &json!({"status": "expired"}),
            None,
            reason,
            completed_event(),
        );
        assert_eq!(status, "expired");
        assert!(matches!(result, Event::Error(error) if error.code == "deadline_exceeded"));
    }

    fn shell_worker(script: &str) -> Worker {
        let mut child = Command::new("/bin/sh")
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let frames = read_worker_frames(child.stdout.take().unwrap());
        Worker {
            child,
            _diagnostics: None,
            input: Some(input),
            frames,
            epoch: EPOCH.into(),
            controls: Default::default(),
            sent_commands: 0,
        }
    }

    async fn run_bounded(worker: Worker, service: Arc<Service>) {
        tokio::time::timeout(
            Duration::from_secs(2),
            coordinate(worker, service, 64, || async {
                Err("No replacement is configured.".to_owned())
            }),
        )
        .await
        .expect("An idle fault must settle without a new request.")
        .expect_err("An idle worker fault must remain a failed coordinator outcome.");
    }

    #[tokio::test]
    async fn idle_pipe_eof_and_invalid_json_withdraw_readiness_without_a_request() {
        for script in ["exit 0", "printf '{invalid}\\n'; read signal"] {
            let worker = shell_worker(script);
            let service = idle_service();
            run_bounded(worker, service.clone()).await;
            assert!(!service.available.load(Ordering::Acquire));
            assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
            assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
            assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
            let error = service
                .enqueue(idle_request())
                .err()
                .expect("A fenced worker cannot admit requests.");
            assert_eq!(error.status, 503);
            assert_eq!(error.code, "worker_lost");
        }
    }

    #[tokio::test]
    async fn pending_idle_fault_rejects_queued_work_before_admission() {
        let mut worker = shell_worker("read signal");
        let (frames, receiver) = mpsc::channel(1);
        let unexpected = json!({"protocol": PROTOCOL, "kind": "drained", "worker_epoch": EPOCH});
        contracts::validate_frame(&unexpected).unwrap();
        frames.send(Ok(unexpected)).await.unwrap();
        worker.frames = receiver;
        let service = idle_service();
        let first = service.enqueue(idle_request()).unwrap();
        let second = service.enqueue(idle_request()).unwrap();
        assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 2);
        run_bounded(worker, service.clone()).await;
        for ticket in [first, second] {
            let result = ticket.begin.await.unwrap();
            let error = result
                .err()
                .expect("Queued work must fail before preparation.");
            assert_eq!(error.code, "worker_lost");
            assert_eq!(error.message, "The worker requires a restart.");
        }
        assert!(!service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn closed_idle_frame_channel_fences_the_live_process() {
        let mut worker = shell_worker("read signal");
        let (sender, frames) = mpsc::channel(1);
        drop(sender);
        worker.frames = frames;
        let service = idle_service();
        run_bounded(worker, service.clone()).await;
        assert!(!service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn idle_control_reply_requires_a_pending_id_and_current_epoch() {
        let mut worker = shell_worker("read signal");
        worker.controls.insert(CONTROL.into());
        let reply = json!({"protocol": PROTOCOL, "kind": "command_result", "worker_epoch": EPOCH,
            "command_id": CONTROL, "status": "already_terminal"});
        contracts::validate_frame(&reply).unwrap();
        let mut stale = reply.clone();
        stale["worker_epoch"] = json!("00000000-0000-4000-8000-000000000009");
        assert_eq!(
            worker.consume_idle_frame(&stale).unwrap_err().code,
            "protocol_fault"
        );
        assert!(worker.controls.contains(CONTROL));
        worker.consume_idle_frame(&reply).unwrap();
        assert!(worker.controls.is_empty());
        assert_eq!(
            worker.consume_idle_frame(&reply).unwrap_err().code,
            "protocol_fault"
        );
        worker.controls.insert(CONTROL.into());
        let mut rejected = reply;
        rejected["status"] = json!("error");
        rejected["error"] = json!({"code": "invalid_request", "message": "Control failed.", "scope": "request", "state_validity": "none"});
        assert_eq!(
            worker.consume_idle_frame(&rejected).unwrap_err().code,
            "protocol_fault"
        );
        let service = idle_service();
        fence_worker(&mut worker, &service, ApiError::worker("Test finished.")).await;
        assert!(worker.child.try_wait().unwrap().is_some());
    }

    #[tokio::test]
    async fn fencing_clears_reservation_after_confirmed_process_exit() {
        let mut worker = shell_worker("read signal");
        let service = idle_service();
        service.metrics.active.store(1, Ordering::Relaxed);
        service.metrics.reserved_bytes.store(64, Ordering::Relaxed);
        fence_worker(
            &mut worker,
            &service,
            ApiError::worker("Injected execution fault."),
        )
        .await;
        assert!(worker.child.try_wait().unwrap().is_some());
        assert!(!service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    }
}
