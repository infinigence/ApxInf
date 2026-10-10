//! CPU checks for public outcomes while a submit write blocks.
use super::*;

struct SubmitProbe(PathBuf);

impl SubmitProbe {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("apxinf-submit-{}.jsonl", uuid::Uuid::new_v4())))
    }

    fn commands(&self) -> Vec<String> {
        std::fs::read_to_string(&self.0)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}

impl Drop for SubmitProbe {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn blocked_submit_service(
    probe: &SubmitProbe,
    timeout: Duration,
    pipe_mode: &str,
) -> (Arc<Service>, tokio::task::JoinHandle<()>) {
    // A large prepared prompt fills the real stdin pipe until this child resumes.
    let script = r#"
import hashlib, json, struct, sys, time
trace_path = sys.argv[1]
def read_command():
    line = sys.stdin.readline()
    if not line: return None
    command = json.loads(line)
    with open(trace_path, "a") as trace: trace.write(command["kind"] + "\n")
    return command
def emit(command, kind, **fields):
    print(json.dumps(dict(protocol=command["protocol"], worker_epoch=command["worker_epoch"],
                         kind=kind, **fields)), flush=True)
for index in range(2):
    prepare = read_command()
    tokens = [3] * (100000 if index == 0 else 1)
    digest = hashlib.sha256(b"apxinf-token-prefix-v1\0" + struct.pack("<Q", len(tokens)) +
                            b"".join(struct.pack("<I", token) for token in tokens)).hexdigest()
    emit(prepare, "prepared_input", command_id=prepare["command_id"],
         model_revision=prepare["model_revision"], token_ids=tokens,
         effective_input_tokens=len(tokens), prompt_digest=digest)
    if index == 0:
        time.sleep(0.75)
        if sys.argv[2] == "close": sys.exit(0)
    submit = read_command()
    identity = dict(request_id=submit["request_id"], attempt=submit["attempt"])
    emit(submit, "accepted", event_seq=0, **identity)
    emit(submit, "tokens", event_seq=1, output_index=0, token_ids=[10], text_delta="late", **identity)
    emit(submit, "terminal", event_seq=2, status="completed", cause="length",
         usage=dict(input_tokens=len(tokens), output_tokens=1),
         state_result=dict(validity="none", consumed_position=0, history_count=len(tokens) + 1),
         text_delta="", metrics=dict(elapsed_ns=20, ttft_ns=10, peak_memory_bytes=4096), **identity)
    if index == 0:
        cancel = read_command()
        emit(cancel, "command_result", command_id=cancel["command_id"], status="already_terminal")
        if sys.argv[2] == "after_terminal": sys.exit(0)
    emit(submit, "resources_released", event_seq=3,
         released_lease_ids=submit["capacity_lease_ids"], retained_lease_ids=[], **identity)
sys.stdin.readline()
"#;
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut child = Command::new("python3")
        .args(["-u", "-c", script])
        .arg(&probe.0)
        .arg(pipe_mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let input = child.stdin.take();
    let frames = read_worker_frames(child.stdout.take().unwrap());
    let worker = Worker {
        child,
        _diagnostics: None,
        input,
        frames,
        epoch: epoch.clone(),
        controls: Default::default(),
        sent_commands: 0,
    };
    let ready = json!({"worker_epoch":epoch, "model_revision":"1".repeat(64),
        "capability_revision":"2".repeat(64), "eos_token_ids":[2],
        "capabilities":{"max_context":131072}});
    let (sender, stopped) = watch::channel(None);
    let service = Arc::new(Service {
        queue: PendingQueue::new(2),
        host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
        ready_state: RwLock::new(ready.clone()),
        ready,
        stopped,
        model_id: "submit-probe".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout,
        memory_budget: 4096,
    });
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        || async { Err("The probe must not rotate.".to_owned()) },
        sender,
    ));
    (service, task)
}

fn request() -> Request {
    Request {
        messages: vec![json!({"role":"user","content":"submit write probe"})],
        tools: vec![],
        max_tokens: 1,
        stops: vec![],
        count_only: false,
    }
}

fn observed_metric(service: &Service, name: &str, status: Option<&str>) -> f64 {
    let labels = status.map_or_else(
        || "operation=\"generate\"".to_owned(),
        |status| format!("operation=\"generate\",status=\"{status}\""),
    );
    let key = format!("{name}{{{labels}}} ");
    service
        .metrics_text()
        .lines()
        .find_map(|line| line.strip_prefix(&key))
        .unwrap_or_else(|| panic!("Missing metric: {key}"))
        .parse()
        .unwrap()
}

fn assert_stopped_submission_observations(service: &Service, status: &str) {
    assert_eq!(service.metrics.peak_bytes.load(Ordering::Relaxed), 4096);
    assert_eq!(
        observed_metric(service, "apxinf_cleanup_pending", None),
        0.0
    );
    assert_eq!(
        observed_metric(service, "apxinf_request_outcomes_total", Some(status)),
        1.0
    );
    assert_eq!(
        observed_metric(
            service,
            "apxinf_service_request_seconds_count",
            Some(status)
        ),
        1.0
    );
    for other in [
        "completed",
        "failed",
        if status == "expired" {
            "cancelled"
        } else {
            "expired"
        },
    ] {
        assert_eq!(
            observed_metric(service, "apxinf_request_outcomes_total", Some(other)),
            0.0
        );
    }
    for name in [
        "apxinf_time_to_first_worker_event_seconds_count",
        "apxinf_worker_first_token_seconds_count",
        "apxinf_worker_terminal_to_settlement_seconds_count",
        "apxinf_public_terminal_to_settlement_seconds_count",
    ] {
        assert_eq!(observed_metric(service, name, None), 1.0);
    }
    assert_eq!(
        observed_metric(
            service,
            "apxinf_time_to_first_public_output_ready_seconds_count",
            None
        ),
        0.0
    );
    assert!(
        observed_metric(
            service,
            "apxinf_public_terminal_to_settlement_seconds_sum",
            None
        ) >= 0.250
    );
}

#[tokio::test]
async fn blocked_submit_deadline_finishes_public_request_before_pipe_recovers() {
    let probe = SubmitProbe::new();
    let (service, task) =
        blocked_submit_service(&probe, Duration::from_millis(150), "resume").await;
    let mut ticket = service.enqueue(request()).unwrap();
    let response = tokio::time::timeout(Duration::from_millis(450), ticket.begin)
        .await
        .expect("A blocked submit must not postpone the public deadline.")
        .unwrap();
    assert_eq!(response.err().unwrap().code, "deadline_exceeded");
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", None),
        1.0
    );
    assert_eq!(probe.commands(), ["prepare_input"]);
    assert!(
        tokio::time::timeout(Duration::from_millis(80), ticket.output.recv())
            .await
            .is_err()
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), ticket.output.recv())
            .await
            .unwrap()
            .is_none()
    );
    assert!(service.available.load(Ordering::Acquire));
    assert_stopped_submission_observations(&service, "expired");

    let mut next = service.enqueue(request()).unwrap();
    assert!(next.begin.await.unwrap().is_ok());
    assert!(matches!(next.output.recv().await, Some(Event::Part(_))));
    assert!(matches!(next.output.recv().await, Some(Event::Done { .. })));
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(
        probe.commands(),
        [
            "prepare_input",
            "submit",
            "cancel_request",
            "prepare_input",
            "submit"
        ]
    );
    service.shutdown();
    service.wait_stopped().await.unwrap();
    task.await.unwrap();
}

#[tokio::test]
async fn blocked_submit_cancellation_finishes_public_request_and_suppresses_late_output() {
    let probe = SubmitProbe::new();
    let (service, task) = blocked_submit_service(&probe, Duration::from_secs(3), "resume").await;
    let mut ticket = service.enqueue(request()).unwrap();
    // Preparation is small in compute cost; the large submit remains blocked for 750 ms.
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(probe.commands(), ["prepare_input"]);
    ticket.cancel.store(true, Ordering::Release);
    let response = tokio::time::timeout(Duration::from_millis(200), ticket.begin)
        .await
        .expect("A blocked submit must not postpone cancellation.")
        .unwrap();
    assert_eq!(response.err().unwrap().code, "cancelled");
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", None),
        1.0
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(80), ticket.output.recv())
            .await
            .is_err()
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), ticket.output.recv())
            .await
            .unwrap()
            .is_none()
    );
    assert_stopped_submission_observations(&service, "cancelled");
    let mut next = service.enqueue(request()).unwrap();
    assert!(next.begin.await.unwrap().is_ok());
    assert!(matches!(next.output.recv().await, Some(Event::Part(_))));
    assert!(matches!(next.output.recv().await, Some(Event::Done { .. })));
    assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(
        probe.commands(),
        [
            "prepare_input",
            "submit",
            "cancel_request",
            "prepare_input",
            "submit"
        ]
    );
    service.shutdown();
    service.wait_stopped().await.unwrap();
    task.await.unwrap();
}

#[tokio::test]
async fn blocked_submit_pipe_failure_does_not_count_an_expired_request_twice() {
    let probe = SubmitProbe::new();
    let (service, task) = blocked_submit_service(&probe, Duration::from_millis(150), "close").await;
    let ticket = service.enqueue(request()).unwrap();
    let response = tokio::time::timeout(Duration::from_millis(450), ticket.begin)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.err().unwrap().code, "deadline_exceeded");
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", None),
        1.0
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), service.wait_stopped())
            .await
            .unwrap()
            .is_err()
    );
    task.await.unwrap();
    assert!(!service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", None),
        0.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_request_outcomes_total", Some("expired")),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_request_outcomes_total", Some("failed")),
        0.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            None
        ),
        1.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_worker_terminal_to_settlement_seconds_count",
            None
        ),
        0.0
    );
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_time_to_first_public_output_ready_seconds_count",
            None
        ),
        0.0
    );
}

#[tokio::test]
async fn memory_peak_survives_fault_after_stopped_submit_terminal() {
    let probe = SubmitProbe::new();
    let (service, task) =
        blocked_submit_service(&probe, Duration::from_millis(150), "after_terminal").await;
    let ticket = service.enqueue(request()).unwrap();
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    assert!(
        tokio::time::timeout(Duration::from_secs(2), service.wait_stopped())
            .await
            .unwrap()
            .is_err()
    );
    task.await.unwrap();
    assert_eq!(service.metrics.peak_bytes.load(Ordering::Relaxed), 4096);
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_worker_terminal_to_settlement_seconds_count",
            None
        ),
        1.0
    );
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", None),
        0.0
    );
}
