use crate::{
    config::Config,
    contracts::{
        self, AttemptKey, AttemptPhase, AttemptState, AudioOperation, AudioResult, Frame, Message,
        QUEUE_LIMIT,
    },
    media::{normal_pressure, MediaLease, MediaRoot, ResponseMedia},
};
use std::{
    collections::{HashMap, VecDeque},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{mpsc, oneshot},
};

#[derive(Debug, Clone)]
pub struct AudioError {
    pub code: &'static str,
    pub message: &'static str,
}
impl AudioError {
    pub fn new(code: &'static str) -> Self {
        Self {
            code,
            message: match code {
                "queue_full" => "The audio request queue is full.",
                "deadline_exceeded" => "The audio request deadline expired.",
                "cancelled" => "The audio request was cancelled.",
                "worker_lost" => "The audio worker exited before settlement.",
                "protocol_fault" => "The audio worker violated its protocol.",
                "capacity_unavailable" => "Audio capacity is unavailable.",
                "model_unavailable" => "The configured audio model is unavailable.",
                "model_not_found" => "The request model does not match this deployment.",
                "task_mismatch" => "This deployment does not support the requested audio task.",
                "unsupported_feature" => "The audio profile does not support this parameter.",
                "invalid_audio" => "The recording is not a supported PCM16 mono WAV file.",
                "payload_too_large" => "The audio request exceeds a declared limit.",
                "output_limit" => "The generated audio exceeds a declared limit.",
                "missing_resource" => "A frozen local audio resource is missing.",
                "provider_error" => "The audio provider could not complete the request.",
                _ => "The audio request is invalid.",
            },
        }
    }
    pub fn status(&self) -> u16 {
        match self.code {
            "model_not_found" => 404,
            "payload_too_large" => 413,
            "queue_full" => 429,
            "deadline_exceeded" => 504,
            "model_unavailable" | "capacity_unavailable" | "worker_lost" => 503,
            "protocol_fault" | "provider_error" | "output_limit" | "missing_resource" => 500,
            _ => 400,
        }
    }
}

pub enum Output {
    Transcript(String),
    Speech(ResponseMedia),
}
pub struct Observations {
    pub ready: AtomicBool,
    pub failed: AtomicBool,
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
    pub completed: AtomicU64,
    pub faults: AtomicU64,
    pub reaped: AtomicBool,
    pub diagnostics: std::sync::Mutex<VecDeque<u8>>,
    pub failure: std::sync::Mutex<Option<&'static str>>,
}
impl Default for Observations {
    fn default() -> Self {
        Self {
            ready: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            faults: AtomicU64::new(0),
            reaped: AtomicBool::new(false),
            diagnostics: std::sync::Mutex::new(VecDeque::with_capacity(8192)),
            failure: std::sync::Mutex::new(None),
        }
    }
}

pub struct Job {
    pub key: AttemptKey,
    pub operation: AudioOperation,
    pub lease: Arc<MediaLease>,
    pub deadline: Instant,
    pub queued_at: Instant,
    pub cancelled: Arc<AtomicBool>,
    pub reply: Option<oneshot::Sender<Result<Output, AudioError>>>,
}
pub enum CommandMessage {
    Submit(Box<Job>),
    Shutdown,
}
#[derive(Clone)]
pub struct AudioService {
    pub config: Arc<Config>,
    pub epoch: String,
    pub media: Arc<MediaRoot>,
    pub observations: Arc<Observations>,
    commands: mpsc::Sender<CommandMessage>,
}
pub struct Cancellation {
    cancelled: Arc<AtomicBool>,
}
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

impl AudioService {
    pub async fn start(
        config: Config,
        media: Arc<MediaRoot>,
        fixture_delay: u64,
    ) -> Result<(Self, tokio::task::JoinHandle<()>), String> {
        let config = Arc::new(config);
        let epoch = uuid::Uuid::new_v4().to_string();
        let (commands, receiver) = mpsc::channel(16);
        let observations = Arc::new(Observations::default());
        let service = Self {
            config,
            epoch,
            media,
            observations,
            commands,
        };
        let mut child_command = Command::new(&service.config.python);
        child_command
            .arg(&service.config.worker)
            .arg("--manifest")
            .arg(&service.config.manifest_path)
            .arg("--manifest-revision")
            .arg(&service.config.revision)
            .arg("--epoch")
            .arg(&service.epoch)
            .arg("--media-dir")
            .arg(&service.media.path)
            .env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if service.config.manifest.provider == "cpu_fixture" {
            child_command.args([
                "--allow-cpu-fixture",
                "--disable-pressure-check",
                "--fixture-delay-ms",
                &fixture_delay.to_string(),
            ]);
        }
        let mut child = child_command.spawn().map_err(|e| e.to_string())?;
        let stdin = child.stdin.take().ok_or("worker stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("worker stdout unavailable")?;
        let mut stderr = child.stderr.take().ok_or("worker stderr unavailable")?;
        let (event_tx, event_rx) = mpsc::channel(32);
        tokio::spawn(read_worker(stdout, event_tx.clone()));
        let log_observations = service.observations.clone();
        tokio::spawn(async move {
            let mut bytes = [0u8; 4096];
            while let Ok(count) = stderr.read(&mut bytes).await {
                if count == 0 {
                    break;
                }
                let mut retained = log_observations.diagnostics.lock().unwrap();
                for &byte in &bytes[..count] {
                    if retained.len() == 8192 {
                        retained.pop_front();
                    }
                    retained.push_back(byte);
                }
            }
        });
        let (write_tx, mut write_rx) = mpsc::channel::<Frame>(4);
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(frame) = write_rx.recv().await {
                let mut bytes = serde_json::to_vec(&frame).unwrap();
                bytes.push(b'\n');
                let result = tokio::time::timeout(Duration::from_secs(2), async {
                    stdin.write_all(&bytes).await?;
                    stdin.flush().await
                })
                .await;
                if !matches!(result, Ok(Ok(()))) {
                    let _ = event_tx.send(WorkerEvent::Lost).await;
                    break;
                }
            }
        });
        let actor_service = service.clone();
        let handle = tokio::spawn(async move {
            run_actor(actor_service, child, receiver, write_tx, event_rx).await;
        });
        Ok((service, handle))
    }
    pub fn submit(
        &self,
        operation: AudioOperation,
        lease: Arc<MediaLease>,
        key: AttemptKey,
        deadline: Instant,
    ) -> Result<(oneshot::Receiver<Result<Output, AudioError>>, Cancellation), AudioError> {
        let (tx, rx) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            key,
            operation,
            lease,
            deadline,
            queued_at: Instant::now(),
            cancelled: cancelled.clone(),
            reply: Some(tx),
        };
        self.commands
            .try_send(CommandMessage::Submit(Box::new(job)))
            .map_err(|_| AudioError::new("queue_full"))?;
        Ok((rx, Cancellation { cancelled }))
    }
    pub async fn shutdown(&self) {
        let _ = self.commands.send(CommandMessage::Shutdown).await;
    }
}

enum WorkerEvent {
    Frame(Box<Frame>),
    Lost,
    ProtocolFault,
}
async fn read_worker(mut source: impl AsyncRead + Unpin, events: mpsc::Sender<WorkerEvent>) {
    let mut partial = Vec::new();
    let mut began = None;
    let mut buffer = [0u8; 4096];
    loop {
        let read = if let Some(start) = began {
            let remaining =
                Duration::from_secs(2).saturating_sub(Instant::now().duration_since(start));
            match tokio::time::timeout(remaining, source.read(&mut buffer)).await {
                Ok(result) => result,
                Err(_) => {
                    let _ = events.send(WorkerEvent::ProtocolFault).await;
                    return;
                }
            }
        } else {
            source.read(&mut buffer).await
        };
        let count = match read {
            Ok(n) if n > 0 => n,
            _ => {
                let _ = events
                    .send(if partial.is_empty() {
                        WorkerEvent::Lost
                    } else {
                        WorkerEvent::ProtocolFault
                    })
                    .await;
                return;
            }
        };
        for &byte in &buffer[..count] {
            if byte == b'\n' {
                let frame = contracts::parse_frame(&partial);
                partial.clear();
                began = None;
                match frame {
                    Ok(frame) => {
                        if events
                            .send(WorkerEvent::Frame(Box::new(frame)))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(_) => {
                        let _ = events.send(WorkerEvent::ProtocolFault).await;
                        return;
                    }
                }
            } else {
                if partial.len() == contracts::MAX_FRAME {
                    let _ = events.send(WorkerEvent::ProtocolFault).await;
                    return;
                }
                if began.is_none() {
                    began = Some(Instant::now());
                }
                partial.push(byte);
            }
        }
    }
}

struct Active {
    job: Job,
    state: AttemptState,
    result: Option<Result<Output, AudioError>>,
    cancellation_at: Option<Instant>,
    output_lease: Option<Arc<MediaLease>>,
    terminal_at: Option<Instant>,
}
fn respond(job: &mut Job, result: Result<Output, AudioError>) {
    if let Some(reply) = job.reply.take() {
        let _ = reply.send(result);
    }
}
fn send_command(
    sender: &mpsc::Sender<Frame>,
    epoch: &str,
    message: Message,
    commands: &mut HashMap<String, &'static str>,
) -> bool {
    let (id, kind) = match &message {
        Message::Submit { command_id, .. } => (command_id, "submit"),
        Message::Cancel { command_id, .. } => (command_id, "cancel"),
        Message::Shutdown { command_id } => (command_id, "shutdown"),
        _ => return false,
    };
    if commands.len() >= 1024 {
        return false;
    }
    commands.insert(id.clone(), kind);
    sender.try_send(Frame::new(epoch, message)).is_ok()
}
fn ready_matches(service: &AudioService, frame: &Frame) -> bool {
    match &frame.message {
        Message::Ready {
            task,
            model,
            model_revision,
            capability_revision,
            manifest_revision,
            provider,
            languages,
            voices,
            ..
        } => {
            *task == service.config.manifest.task
                && *model == service.config.manifest.model
                && *model_revision == service.config.revision
                && *manifest_revision == service.config.revision
                && *capability_revision == service.config.capability_revision
                && *provider == service.config.manifest.provider
                && *languages == service.config.manifest.languages
                && *voices
                    == service
                        .config
                        .manifest
                        .voices
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
        }
        _ => false,
    }
}

async fn run_actor(
    service: AudioService,
    child: Child,
    mut receiver: mpsc::Receiver<CommandMessage>,
    writer: mpsc::Sender<Frame>,
    mut events: mpsc::Receiver<WorkerEvent>,
) {
    let obs = &service.observations;
    let mut child = Some(child);
    let mut queue = VecDeque::<Job>::new();
    let mut active: Option<Active> = None;
    let mut commands = HashMap::new();
    let mut stopping = false;
    let mut fault: Option<&'static str> = None;
    let mut stop_at = None;
    let mut stopped_event = false;
    let started = Instant::now();
    let mut pressure_checked = started;
    let mut timer = tokio::time::interval(Duration::from_millis(20));
    loop {
        tokio::select! {
            command = receiver.recv() => match command {
                Some(CommandMessage::Submit(job)) => {
                    let mut job = *job;
                    let error = if stopping || fault.is_some() || !obs.ready.load(Ordering::SeqCst) { Some("model_unavailable") }
                        else if job.operation.task() != service.config.manifest.task { Some("task_mismatch") }
                        else if queue.len() >= QUEUE_LIMIT { Some("queue_full") } else { None };
                    if let Some(code) = error { respond(&mut job, Err(AudioError::new(code))); } else { queue.push_back(job); }
                }
                Some(CommandMessage::Shutdown) | None => { stopping = true; }
            },
            event = events.recv(), if child.is_some() && fault.is_none() => match event {
                Some(WorkerEvent::Frame(frame)) => {
                    if frame.worker_epoch != service.epoch { fault = Some("protocol_fault"); }
                    else { match &frame.message {
                        Message::Ready { .. } if !obs.ready.load(Ordering::SeqCst) && !stopping => {
                            if !ready_matches(&service, &frame) { fault = Some("protocol_fault"); }
                            else if service.config.pressure && !normal_pressure() { fault = Some("capacity_unavailable"); }
                            else { obs.ready.store(true, Ordering::SeqCst); }
                        }
                        Message::Accepted { .. } | Message::Terminal { .. } | Message::ResourcesReleased { .. } => {
                            if let Some(current) = active.as_mut() {
                                if current.state.event(&frame.message).is_err() { fault = Some("protocol_fault"); }
                                else if let Message::Terminal { result, .. } = &frame.message {
                                    current.terminal_at = Some(Instant::now());
                                    let checked = match (result, &current.job.operation) {
                                        (AudioResult::Transcript { text }, AudioOperation::Asr { .. }) => Ok(Output::Transcript(text.clone())),
                                        (AudioResult::Speech { output }, AudioOperation::Tts { output_handle, .. }) => {
                                            current.output_lease.as_ref().unwrap().read_output(output, output_handle).map(|bytes| Output::Speech(ResponseMedia { bytes, lease: current.output_lease.as_ref().unwrap().clone() }))
                                                .map_err(|_| AudioError::new("protocol_fault"))
                                        },
                                        (AudioResult::Cancelled {}, _) => Err(AudioError::new("cancelled")),
                                        (AudioResult::Failed { code, .. }, _) => Err(AudioError::new(match code.as_str() { "output_limit" => "output_limit", "missing_resource" => "missing_resource", _ => "provider_error" })),
                                        _ => Err(AudioError::new("protocol_fault")),
                                    };
                                    if matches!(&checked, Err(error) if error.code == "protocol_fault") { fault = Some("protocol_fault"); }
                                    current.result = Some(checked);
                                }
                            } else { fault = Some("protocol_fault"); }
                        }
                        Message::ControlAck { command_id, status } => {
                            match (commands.get(command_id), status.as_str()) {
                                (Some(&"cancel"), "cancel_requested" | "already_terminal") | (Some(&"shutdown"), "shutdown_requested") => {
                                    commands.insert(command_id.clone(), "acknowledged");
                                },
                                _ => fault = Some("protocol_fault"),
                            }
                        }
                        Message::Stopped {} if stopping && active.is_none() && !stopped_event => { stopped_event = true; }
                        _ => fault = Some("protocol_fault"),
                    } }
                }
                Some(WorkerEvent::Lost) | None => {
                    if !stopping || active.is_some() || !stopped_event { fault = Some("worker_lost"); }
                }
                Some(WorkerEvent::ProtocolFault) => fault = Some("protocol_fault"),
            },
            _ = timer.tick() => {}
        }
        let now = Instant::now();
        if service.config.pressure
            && child.is_some()
            && now.duration_since(pressure_checked) >= Duration::from_millis(200)
        {
            pressure_checked = now;
            if !normal_pressure() {
                fault = Some("capacity_unavailable");
            }
        }
        if let Some(current) = active.as_mut() {
            if current.job.cancelled.load(Ordering::SeqCst)
                || current.job.reply.as_ref().is_some_and(|r| r.is_closed())
                || now >= current.job.deadline
                || stopping
            {
                let code = if now >= current.job.deadline {
                    "deadline_exceeded"
                } else {
                    "cancelled"
                };
                respond(&mut current.job, Err(AudioError::new(code)));
                if current.cancellation_at.is_none() && current.state.phase != AttemptPhase::Settled
                {
                    current.cancellation_at = Some(now);
                    if !send_command(
                        &writer,
                        &service.epoch,
                        Message::Cancel {
                            command_id: uuid::Uuid::new_v4().to_string(),
                            key: current.job.key.clone(),
                        },
                        &mut commands,
                    ) {
                        fault = Some("worker_lost");
                    }
                }
            }
            if current.state.phase == AttemptPhase::Settled {
                let mut current = active.take().unwrap();
                let result = current
                    .result
                    .take()
                    .unwrap_or_else(|| Err(AudioError::new("protocol_fault")));
                if result.is_ok()
                    && current
                        .job
                        .reply
                        .as_ref()
                        .is_some_and(|reply| !reply.is_closed())
                {
                    obs.completed.fetch_add(1, Ordering::SeqCst);
                }
                respond(&mut current.job, result);
            } else if current
                .cancellation_at
                .is_some_and(|at| now.duration_since(at) >= Duration::from_secs(1))
            {
                fault = Some("worker_lost");
            } else if current
                .terminal_at
                .is_some_and(|at| now.duration_since(at) >= Duration::from_secs(1))
            {
                fault = Some("protocol_fault");
            }
        }
        let mut kept = VecDeque::new();
        while let Some(mut job) = queue.pop_front() {
            let code = if stopping {
                Some("cancelled")
            } else if fault.is_some() {
                fault
            } else if job.cancelled.load(Ordering::SeqCst) {
                Some("cancelled")
            } else if now >= job.deadline
                || now.duration_since(job.queued_at) >= Duration::from_secs(5)
            {
                Some("deadline_exceeded")
            } else {
                None
            };
            if let Some(code) = code {
                respond(&mut job, Err(AudioError::new(code)));
            } else {
                kept.push_back(job);
            }
        }
        queue = kept;
        if !stopping && fault.is_none() && active.is_none() && obs.ready.load(Ordering::SeqCst) {
            if commands.len() >= 1020 {
                stopping = true;
            } else if let Some(mut job) = queue.pop_front() {
                if service.config.pressure && !normal_pressure() {
                    respond(&mut job, Err(AudioError::new("capacity_unavailable")));
                } else {
                    let output_lease =
                        if let AudioOperation::Tts { output_handle, .. } = &job.operation {
                            match service.media.reserve(2 * contracts::MAX_OUTPUT) {
                                Ok(lease) => {
                                    lease.add_handle(output_handle);
                                    Some(lease)
                                }
                                Err(_) => {
                                    respond(&mut job, Err(AudioError::new("capacity_unavailable")));
                                    continue;
                                }
                            }
                        } else {
                            None
                        };
                    let lease_id = uuid::Uuid::new_v4().to_string();
                    let message = Message::Submit {
                        command_id: uuid::Uuid::new_v4().to_string(),
                        key: job.key.clone(),
                        lease_id: lease_id.clone(),
                        model_revision: service.config.revision.clone(),
                        timeout_ms: job
                            .deadline
                            .saturating_duration_since(now)
                            .as_millis()
                            .clamp(1, 120000) as u64,
                        operation: job.operation.clone(),
                    };
                    let state = AttemptState::new(job.key.clone(), lease_id);
                    if !send_command(&writer, &service.epoch, message, &mut commands) {
                        fault = Some("worker_lost");
                    }
                    active = Some(Active {
                        job,
                        state,
                        result: None,
                        cancellation_at: None,
                        output_lease,
                        terminal_at: None,
                    });
                }
            }
        }
        if !obs.ready.load(Ordering::SeqCst)
            && now.duration_since(started) > Duration::from_secs(120)
            && fault.is_none()
        {
            fault = Some("worker_lost");
        }
        if stopping && stop_at.is_none() && child.is_some() {
            obs.ready.store(false, Ordering::SeqCst);
            stop_at = Some(now);
            if !send_command(
                &writer,
                &service.epoch,
                Message::Shutdown {
                    command_id: uuid::Uuid::new_v4().to_string(),
                },
                &mut commands,
            ) {
                fault = Some("worker_lost");
            }
        }
        if let Some(code) = fault {
            *obs.failure.lock().unwrap() = Some(code);
            if !obs.failed.swap(true, Ordering::SeqCst) {
                obs.faults.fetch_add(1, Ordering::SeqCst);
            }
            obs.ready.store(false, Ordering::SeqCst);
            if let Some(current) = active.as_mut() {
                respond(&mut current.job, Err(AudioError::new(code)));
            }
            while let Some(mut job) = queue.pop_front() {
                respond(&mut job, Err(AudioError::new(code)));
            }
        }
        if fault.is_some()
            || stop_at.is_some_and(|at| now.duration_since(at) >= Duration::from_secs(1))
        {
            if let Some(process) = child.as_mut() {
                let _ = process.start_kill();
            }
        }
        if let Some(process) = child.as_mut() {
            match process.try_wait() {
                Ok(Some(_)) => {
                    child = None;
                    obs.reaped.store(true, Ordering::SeqCst);
                    obs.ready.store(false, Ordering::SeqCst);
                    if let Some(mut current) = active.take() {
                        respond(
                            &mut current.job,
                            Err(AudioError::new(fault.unwrap_or("worker_lost"))),
                        );
                    }
                    if !stopping && fault.is_none() {
                        fault = Some("worker_lost");
                    }
                }
                Err(_) => {
                    fault.get_or_insert("worker_lost");
                }
                Ok(None) => {}
            }
        }
        obs.active
            .store(usize::from(active.is_some()), Ordering::SeqCst);
        obs.queued.store(queue.len(), Ordering::SeqCst);
        if stopping && child.is_none() {
            break;
        }
    }
    drop(writer);
}
