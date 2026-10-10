//! CPU regressions for request lifetime and worker replacement.
mod pressure_regressions;
use super::*;

pub(super) struct CpuModel(PathBuf);

impl CpuModel {
    pub(super) fn new(settings: Value) -> Self {
        let directory =
            std::env::temp_dir().join(format!("apxinf-lifecycle-{}", uuid::Uuid::new_v4()));
        let model = directory.join("model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(
            model.join("lifecycle.json"),
            serde_json::to_vec(&settings).unwrap(),
        )
        .unwrap();
        Self(directory)
    }
    pub(super) fn model(&self) -> PathBuf {
        self.0.join("model")
    }
    pub(super) fn events(&self) -> Vec<Value> {
        std::fs::read_to_string(self.0.join("model-trace.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

impl Drop for CpuModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) async fn cpu_worker(model: PathBuf) -> Result<(Worker, Value), String> {
    let epoch = uuid::Uuid::new_v4().to_string();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/serving/lifecycle_worker.py");
    let mut child = Command::new("python3")
        .arg("-u")
        .arg(fixture)
        .arg("--model")
        .arg(model)
        .arg("--worker-epoch")
        .arg(&epoch)
        .args(["--max-context", "64", "--max-output-tokens", "16"])
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| e.to_string())?;
    let input = child.stdin.take();
    let frames = read_worker_frames(child.stdout.take().unwrap());
    let mut worker = Worker {
        child,
        _diagnostics: None,
        input,
        frames,
        epoch,
        controls: Default::default(),
        sent_commands: 0,
    };
    let ready = worker
        .receive(Duration::from_secs(3))
        .await
        .map_err(|e| e.message)?;
    Ok((worker, ready))
}

pub(super) fn cpu_service(ready: Value, timeout: Duration, capacity: usize) -> Arc<Service> {
    cpu_service_with_completion(ready, timeout, capacity).0
}

fn cpu_service_with_completion(
    ready: Value,
    timeout: Duration,
    capacity: usize,
) -> (Arc<Service>, watch::Sender<Option<StopResult>>) {
    let (sender, stopped) = watch::channel(None);
    let service = Arc::new(Service {
        queue: PendingQueue::new(capacity),
        host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
        ready_state: RwLock::new(ready.clone()),
        stopped,
        ready,
        model_id: "cpu-fixture".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout,
        memory_budget: 4096,
    });
    (service, sender)
}

pub(super) fn request(count_only: bool) -> Request {
    Request {
        messages: vec![json!({"role":"user","content":"original lifecycle request"})],
        tools: vec![],
        max_tokens: 0,
        stops: vec![],
        count_only,
    }
}

async fn run_empty_request(service: &Service) {
    let mut ticket = service.enqueue(request(false)).unwrap();
    assert!(ticket.begin.await.unwrap().is_ok());
    assert!(matches!(
        ticket.output.recv().await,
        Some(Event::Done {
            output_tokens: 0,
            ..
        })
    ));
}

async fn wait_for_event(model: &CpuModel, event: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if model.events().iter().any(|record| record["event"] == event) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

async fn stop_coordinator(service: &Service, task: tokio::task::JoinHandle<StopResult>) {
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

fn observed_metric(service: &Service, name: &str, labels: &str) -> f64 {
    let key = format!("{name}{{{labels}}} ");
    service
        .metrics_text()
        .lines()
        .find_map(|line| line.strip_prefix(&key))
        .unwrap_or_else(|| panic!("Missing metric: {key}"))
        .parse()
        .unwrap()
}

#[tokio::test]
async fn observations_separate_count_tokens_and_generate_outcomes() {
    let model = CpuModel::new(json!({}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(3), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let count = service.enqueue(request(true)).unwrap();
    assert!(count.begin.await.unwrap().is_ok());
    let mut generation = request(false);
    generation.max_tokens = 3;
    let mut generated = service.enqueue(generation).unwrap();
    assert!(generated.begin.await.unwrap().is_ok());
    loop {
        match generated.output.recv().await {
            Some(Event::Done { .. }) => break,
            Some(Event::Part(_)) => {}
            _ => panic!("Generation must complete."),
        }
    }
    stop_coordinator(&service, task).await;
    for operation in ["count_tokens", "generate"] {
        let labels = format!("operation=\"{operation}\"");
        let completed = format!("{labels},status=\"completed\"");
        assert_eq!(
            observed_metric(&service, "apxinf_request_outcomes_total", &completed),
            1.0
        );
        assert_eq!(
            observed_metric(&service, "apxinf_service_request_seconds_count", &completed),
            1.0
        );
        assert_eq!(
            observed_metric(&service, "apxinf_queue_wait_seconds_count", &labels),
            1.0
        );
        assert_eq!(
            observed_metric(&service, "apxinf_preparation_seconds_count", &labels),
            1.0
        );
        assert_eq!(
            observed_metric(&service, "apxinf_cleanup_pending", &labels),
            0.0
        );
        for name in [
            "apxinf_time_to_first_worker_event_seconds_count",
            "apxinf_worker_first_token_seconds_count",
            "apxinf_time_to_first_public_output_ready_seconds_count",
        ] {
            assert_eq!(
                observed_metric(&service, name, &labels),
                if operation == "generate" { 1.0 } else { 0.0 }
            );
        }
    }
    for status in ["completed", "cancelled", "failed", "expired"] {
        let count = observed_metric(
            &service,
            "apxinf_request_outcomes_total",
            &format!("operation=\"count_tokens\",status=\"{status}\""),
        );
        let generate = observed_metric(
            &service,
            "apxinf_request_outcomes_total",
            &format!("operation=\"generate\",status=\"{status}\""),
        );
        assert_eq!(
            observed_metric(
                &service,
                "apxinf_requests_total",
                &format!("status=\"{status}\"")
            ),
            count + generate
        );
    }
}

#[tokio::test]
async fn observations_queue_expiry_has_wait_samples_without_execution() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[600]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_millis(150), 2);
    start_queue_sweeper(&service);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let mut active = service.enqueue(request(true)).unwrap();
    wait_for_event(&model, "prepare_started").await;
    let queued = service.enqueue(request(false)).unwrap();
    let error = tokio::time::timeout(Duration::from_millis(350), queued.begin)
        .await
        .unwrap()
        .unwrap()
        .err()
        .unwrap();
    assert_eq!(error.code, "deadline_exceeded");
    assert_eq!(
        active.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    let labels = "operation=\"generate\"";
    let expired = "operation=\"generate\",status=\"expired\"";
    assert_eq!(
        observed_metric(&service, "apxinf_request_outcomes_total", expired),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_service_request_seconds_count", expired),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_queue_wait_seconds_count", labels),
        1.0
    );
    assert!(observed_metric(&service, "apxinf_queue_wait_seconds_sum", labels) >= 0.150);
    assert!(observed_metric(&service, "apxinf_service_request_seconds_sum", expired) >= 0.150);
    for name in [
        "apxinf_preparation_seconds_count",
        "apxinf_time_to_first_worker_event_seconds_count",
        "apxinf_worker_first_token_seconds_count",
        "apxinf_time_to_first_public_output_ready_seconds_count",
        "apxinf_cleanup_pending",
    ] {
        assert_eq!(observed_metric(&service, name, labels), 0.0);
    }
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            labels
        ),
        1.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_sum",
            labels
        ),
        0.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_cleanup_pending",
            "operation=\"count_tokens\""
        ),
        1.0
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), active.output.recv())
            .await
            .unwrap()
            .is_none()
    );
    stop_coordinator(&service, task).await;
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_cleanup_pending",
            "operation=\"count_tokens\""
        ),
        0.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", labels),
        0.0
    );
}

#[tokio::test]
async fn observations_preparation_expiry_tracks_public_result_until_settlement() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[500]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_millis(100), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let mut ticket = service.enqueue(request(false)).unwrap();
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    let labels = "operation=\"generate\"";
    let expired = "operation=\"generate\",status=\"expired\"";
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", labels),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_request_outcomes_total", expired),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_service_request_seconds_count", expired),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_preparation_seconds_count", labels),
        0.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            labels
        ),
        0.0
    );
    let public_seconds = observed_metric(&service, "apxinf_service_request_seconds_sum", expired);
    assert!((0.100..0.350).contains(&public_seconds));
    assert!(
        tokio::time::timeout(Duration::from_secs(2), ticket.output.recv())
            .await
            .unwrap()
            .is_none()
    );
    stop_coordinator(&service, task).await;
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", labels),
        0.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_preparation_seconds_count", labels),
        1.0
    );
    assert!(observed_metric(&service, "apxinf_preparation_seconds_sum", labels) >= 0.500);
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            labels
        ),
        1.0
    );
    assert!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_sum",
            labels
        ) >= 0.150
    );
    assert_eq!(
        observed_metric(&service, "apxinf_service_request_seconds_sum", expired),
        public_seconds
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_worker_terminal_to_settlement_seconds_count",
            labels
        ),
        0.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_time_to_first_public_output_ready_seconds_count",
            labels
        ),
        0.0
    );
}

#[tokio::test]
async fn late_preparation_settles_before_the_next_request_and_preserves_readiness() {
    for reject in [false, true] {
        let model = CpuModel::new(
            json!({"prepare_delays_ms":[200], "prepare_error_indices": if reject { vec![0] } else { vec![] }}),
        );
        let (worker, ready) = cpu_worker(model.model()).await.unwrap();
        let service = cpu_service(ready, Duration::from_millis(150), 2);
        start_queue_sweeper(&service);
        let path = model.model();
        let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
            cpu_worker(path.clone())
        }));
        let first = service.enqueue(request(true)).unwrap();
        assert_eq!(
            first.begin.await.unwrap().err().unwrap().code,
            "deadline_exceeded"
        );
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
        let second = service.enqueue(request(true)).unwrap();
        assert!(second.begin.await.unwrap().is_ok());
        assert!(service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
        let events = model.events();
        let starts: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event["event"] == "prepare_started")
            .map(|(i, _)| i)
            .collect();
        let settled = events
            .iter()
            .position(|event| {
                event["event"]
                    == if reject {
                        "prepare_rejected"
                    } else {
                        "prepare_finished"
                    }
            })
            .unwrap();
        assert!(settled < starts[1]);
        stop_coordinator(&service, task).await;
    }
}

#[tokio::test]
async fn cancelled_waiters_release_slots_while_generation_remains_active() {
    let model = CpuModel::new(json!({"generation_pause_ms":1000}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(3), 2);
    start_queue_sweeper(&service);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let mut active_request = request(false);
    active_request.max_tokens = 3;
    let mut active = service.enqueue(active_request).unwrap();
    active.begin.await.unwrap().unwrap();
    wait_for_event(&model, "generate_started").await;
    let first = service.enqueue(request(true)).unwrap();
    let second = service.enqueue(request(true)).unwrap();
    assert_eq!(
        service.enqueue(request(true)).err().unwrap().code,
        "queue_full"
    );
    first.cancel.store(true, Ordering::Release);
    second.cancel.store(true, Ordering::Release);
    for ticket in [first, second] {
        let error = tokio::time::timeout(Duration::from_millis(300), ticket.begin)
            .await
            .unwrap()
            .unwrap()
            .err()
            .unwrap();
        assert_eq!(error.code, "cancelled");
    }
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
    let replacement = service.enqueue(request(true)).unwrap();
    active.cancel.store(true, Ordering::Release);
    while let Some(event) = active.output.recv().await {
        if matches!(event, Event::Error(_)) {
            break;
        }
    }
    assert!(replacement.begin.await.unwrap().is_ok());
    assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 3);
    stop_coordinator(&service, task).await;
}

#[tokio::test]
async fn coordinator_rotates_before_command_exhaustion_and_serves_the_next_request() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let old_epoch = worker.epoch.clone();
    // Fill real duplicate history. The remaining records cover one request and shutdown.
    while worker.sent_commands < contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE {
        let count =
            (contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE - worker.sent_commands).min(16);
        let mut commands = Vec::new();
        for _ in 0..count {
            let command = worker.command("query_stats");
            worker.send(&command).await.unwrap();
            commands.push(command);
        }
        for command in commands {
            let reply = worker.receive(Duration::from_secs(3)).await.unwrap();
            assert_eq!(reply["command_id"], command["command_id"]);
        }
    }
    assert!(!worker.requires_rotation());
    let service = cpu_service(ready, Duration::from_secs(5), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    run_empty_request(&service).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while service.metrics.worker_rotations.load(Ordering::Relaxed) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_ne!(service.ready_snapshot()["worker_epoch"], old_epoch);
    assert!(service.available.load(Ordering::Acquire));
    run_empty_request(&service).await;
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 2);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    let events = model.events();
    let starts: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| event["event"] == "started")
        .map(|(i, _)| i)
        .collect();
    let drained = events
        .iter()
        .position(|event| event["kind"] == "drained")
        .unwrap();
    assert_eq!(starts.len(), 2);
    assert!(drained < starts[1]);
    stop_coordinator(&service, task).await;
}

#[tokio::test]
async fn replacement_identity_change_is_rejected_and_reaped() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(5), 2);
    let path = model.model();
    let result = rotate_worker(&mut worker, &service, &mut || {
        let path = path.clone();
        async move {
            let (replacement, mut ready) = cpu_worker(path).await?;
            ready["capability_revision"] = json!("f".repeat(64));
            Ok::<_, String>((replacement, ready))
        }
    })
    .await;
    assert_eq!(result.unwrap_err().code, "worker_lost");
    assert!(!service.available.load(Ordering::Acquire));
    assert!(worker.child.try_wait().unwrap().unwrap().success());
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn preparation_deadline_returns_expiry_before_late_worker_reply() {
    for blocked_write in [false, true] {
        let epoch = "00000000-0000-4000-8000-000000000001";
        let script = r#"
import hashlib, json, struct, sys, time
if sys.argv[1] == "blocked": time.sleep(0.5)
command = json.loads(sys.stdin.readline())
if sys.argv[1] == "response": time.sleep(0.5)
print(json.dumps({"protocol": command["protocol"], "kind": "prepared_input",
    "worker_epoch": command["worker_epoch"], "command_id": command["command_id"],
    "model_revision": command["model_revision"], "token_ids": [3],
    "effective_input_tokens": 1, "prompt_digest": hashlib.sha256(
        b"apxinf-token-prefix-v1\0" + struct.pack("<QI", 1, 3)).hexdigest()}), flush=True)
sys.stdin.readline()
"#;
        let mut child = Command::new("python3")
            .args(["-u", "-c", script])
            .arg(if blocked_write { "blocked" } else { "response" })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let frames = read_worker_frames(child.stdout.take().unwrap());
        let mut worker = Worker {
            child,
            _diagnostics: None,
            input: Some(input),
            frames,
            epoch: epoch.into(),
            controls: Default::default(),
            sent_commands: 0,
        };
        let service = Arc::new(Service {
            queue: PendingQueue::new(1),
            host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
            stopped: watch::channel(None).1,
            ready_state: RwLock::new(json!({"worker_epoch":epoch})),
            ready: json!({"model_revision": "1".repeat(64), "capabilities": {"max_context": 64}}),
            model_id: "test-model".into(),
            available: AtomicBool::new(true),
            rotating: AtomicBool::new(false),
            metrics: Metrics::default(),
            timeout: Duration::from_millis(150),
            memory_budget: 1024,
        });
        let content = if blocked_write {
            "x".repeat(700_000)
        } else {
            "hello".into()
        };
        let ticket = service
            .enqueue(Request {
                messages: vec![json!({"role":"user","content":content})],
                tools: vec![],
                max_tokens: 0,
                stops: vec![],
                count_only: true,
            })
            .unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        let execution_service = service.clone();
        let execution = tokio::spawn(async move {
            let result = execute(
                &mut worker,
                &execution_service,
                job,
                execution_service.timeout,
            )
            .await;
            let _ = worker.child.start_kill();
            let _ = worker.child.wait().await;
            result
        });
        let response = tokio::time::timeout(Duration::from_millis(350), ticket.begin)
            .await
            .expect("The public deadline must not wait for preparation.")
            .expect("The coordinator must return a request error.");
        assert_eq!(
            response.err().expect("Preparation must expire.").code,
            "deadline_exceeded"
        );
        let outcome = execution.await.unwrap();
        assert!(
            outcome.is_ok(),
            "Late preparation must not lose the worker: {outcome:?}"
        );
        assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn rotation_publishes_replacement_memory_and_shutdown_releases_ownership() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(3), 2);
    let path = model.model();
    rotate_worker(&mut worker, &service, &mut || {
        let path = path.clone();
        async move {
            let (replacement, mut ready) = cpu_worker(path).await?;
            ready["memory"]["active_bytes"] = json!(2048);
            ready["memory"]["peak_bytes"] = json!(3072);
            Ok::<_, String>((replacement, ready))
        }
    })
    .await
    .unwrap();
    assert_eq!(service.ready_snapshot()["memory"]["active_bytes"], 2048);
    assert_eq!(service.metrics.peak_bytes.load(Ordering::Relaxed), 3072);
    start_queue_sweeper(&service);
    let weak = Arc::downgrade(&service);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    stop_coordinator(&service, task).await;
    drop(service);
    tokio::time::timeout(Duration::from_secs(1), async {
        while weak.upgrade().is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn preparation_cancellation_returns_before_settlement_and_does_not_generate() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[250]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(3), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let first = service.enqueue(request(false)).unwrap();
    wait_for_event(&model, "prepare_started").await;
    first.cancel.store(true, Ordering::Release);
    let error = tokio::time::timeout(Duration::from_millis(150), first.begin)
        .await
        .unwrap()
        .unwrap()
        .err()
        .unwrap();
    assert_eq!(error.code, "cancelled");
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    let next = service.enqueue(request(true)).unwrap();
    assert!(next.begin.await.unwrap().is_ok());
    assert!(!model
        .events()
        .iter()
        .any(|event| event["event"] == "generate_started"));
    assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 0);
    stop_coordinator(&service, task).await;
}

#[tokio::test]
async fn preparation_grace_expiry_fences_without_a_second_public_terminal() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[60000]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_millis(60), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let ticket = service.enqueue(request(true)).unwrap();
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    assert!(service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    tokio::time::timeout(Duration::from_secs(25), task)
        .await
        .unwrap()
        .unwrap()
        .expect_err("An unresponsive worker must produce a failed coordinator outcome.");
    assert!(!service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn wait_stopped_retains_ownership_after_a_public_preparation_deadline() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[600]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let (service, completion) = cpu_service_with_completion(ready, Duration::from_millis(100), 2);
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        completion,
    ));
    let ticket = service.enqueue(request(true)).unwrap();
    wait_for_event(&model, "prepare_started").await;
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);

    service.shutdown();
    let stopped = service.wait_stopped();
    tokio::pin!(stopped);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut stopped)
            .await
            .is_err(),
        "A public terminal result must not prove worker process reaping."
    );
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    tokio::time::timeout(Duration::from_secs(3), stopped)
        .await
        .unwrap()
        .unwrap();
    task.await.unwrap();

    assert!(model
        .events()
        .iter()
        .any(|event| event["event"] == "prepare_finished"));
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    // The completed result remains available to callers that subscribe later.
    service.wait_stopped().await.unwrap();
}

#[tokio::test]
async fn wait_stopped_reports_a_disappearing_completion_sender() {
    let (service, completion) = cpu_service_with_completion(json!({}), Duration::from_secs(1), 1);
    let stopped = service.wait_stopped();
    tokio::pin!(stopped);
    assert!(futures_util::poll!(&mut stopped).is_pending());

    drop(completion);

    let error = tokio::time::timeout(Duration::from_secs(1), stopped)
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.contains("without confirming worker process reaping"));
}

#[tokio::test]
async fn wait_stopped_reports_worker_failure_after_reaping() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let (service, completion) = cpu_service_with_completion(ready, Duration::from_secs(1), 1);
    worker.child.start_kill().unwrap();
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        completion,
    ));

    tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .unwrap()
        .expect_err("Worker failure must remain a failed coordinator outcome.");
    task.await.unwrap();
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert!(!service.available.load(Ordering::Acquire));
}

#[tokio::test]
async fn wait_stopped_covers_a_replacement_that_is_still_loading() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    let (service, completion) = cpu_service_with_completion(ready, Duration::from_secs(3), 2);
    let (started, started_rx) = oneshot::channel();
    let (resume, resume_rx) = oneshot::channel();
    let mut gate = Some((started, resume_rx));
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || {
            let (started, resume_rx) = gate.take().unwrap();
            let path = path.clone();
            async move {
                started.send(()).unwrap();
                resume_rx.await.unwrap();
                cpu_worker(path).await
            }
        },
        completion,
    ));
    started_rx.await.unwrap();
    service.shutdown();
    let stopped = service.wait_stopped();
    tokio::pin!(stopped);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), &mut stopped)
            .await
            .is_err(),
        "Shutdown must still own an in-progress replacement launch."
    );

    resume.send(()).unwrap();

    tokio::time::timeout(Duration::from_secs(3), stopped)
        .await
        .unwrap()
        .unwrap();
    task.await.unwrap();
    assert!(service.queue.is_closed());
    assert!(!service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn shutdown_during_replacement_loading_cannot_restore_readiness() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = cpu_service(ready, Duration::from_secs(3), 2);
    let (started, started_rx) = oneshot::channel();
    let (resume, resume_rx) = oneshot::channel();
    let mut started = Some(started);
    let mut resume_rx = Some(resume_rx);
    let path = model.model();
    let mut launch = || {
        let started = started.take().unwrap();
        let resume_rx = resume_rx.take().unwrap();
        let path = path.clone();
        async move {
            started.send(()).unwrap();
            resume_rx.await.unwrap();
            cpu_worker(path).await
        }
    };
    let (rotation, ()) = tokio::join!(rotate_worker(&mut worker, &service, &mut launch), async {
        started_rx.await.unwrap();
        service.shutdown();
        resume.send(()).unwrap();
    });
    rotation.unwrap();
    assert!(service.queue.is_closed());
    assert!(!service.available.load(Ordering::Acquire));
    assert!(service.enqueue(request(true)).is_err());
    let _ = worker.child.start_kill();
    worker.child.wait().await.unwrap();
    assert!(worker.child.try_wait().unwrap().is_some());
}

#[test]
fn shutdown_wins_over_concurrent_availability_publication() {
    for _ in 0..100 {
        let service = cpu_service(
            json!({"worker_epoch":uuid::Uuid::new_v4().to_string()}),
            Duration::from_secs(1),
            1,
        );
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let publication = service.clone();
        let other = barrier.clone();
        let thread = std::thread::spawn(move || {
            other.wait();
            publication
                .queue
                .publish_availability(&publication.available);
        });
        barrier.wait();
        service.shutdown();
        thread.join().unwrap();
        assert!(service.queue.is_closed());
        assert!(!service.available.load(Ordering::Acquire));
    }
}
