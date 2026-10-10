//! CPU protocol probes for public deadlines after submission finishes.
use super::*;
use futures_util::FutureExt;

struct GenerationProbe(PathBuf);

impl GenerationProbe {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "apxinf-generation-deadline-{}.jsonl",
            uuid::Uuid::new_v4()
        )))
    }

    fn commands(&self) -> Vec<String> {
        std::fs::read_to_string(&self.0)
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }
}

impl Drop for GenerationProbe {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

async fn service_for(
    probe: &GenerationProbe,
    mode: &str,
) -> (Arc<Service>, tokio::task::JoinHandle<()>) {
    // Model execution is absent. Real pipes carry the complete checked protocol.
    let script = r#"
import hashlib, json, queue, struct, sys, threading, time
mode, trace_path = sys.argv[1:]
generation = 0
commands = queue.Queue()
def receive():
    for line in sys.stdin:
        command = json.loads(line)
        with open(trace_path, "a") as trace: trace.write(command["kind"] + "\n")
        commands.put(command)
    commands.put(None)
threading.Thread(target=receive, daemon=True).start()
def read():
    return commands.get()
def emit(command, kind, **fields):
    print(json.dumps(dict(protocol=command["protocol"], worker_epoch=command["worker_epoch"],
                         kind=kind, **fields)), flush=True)
def control(command, status="already_terminal"):
    emit(command, "command_result", command_id=command["command_id"], status=status)
while True:
    command = read()
    if command is None: break
    kind = command["kind"]
    if kind == "prepare_input":
        tokens = [3, 4]
        payload = b"apxinf-token-prefix-v1\0" + struct.pack("<QII", 2, 3, 4)
        emit(command, "prepared_input", command_id=command["command_id"],
             model_revision=command["model_revision"], token_ids=tokens,
             effective_input_tokens=2, prompt_digest=hashlib.sha256(payload).hexdigest())
    elif kind == "submit":
        selected = mode if generation == 0 else "normal"
        generation += 1
        identity = dict(request_id=command["request_id"], attempt=command["attempt"])
        if selected == "delayed_accept": time.sleep(0.8)
        if selected == "cancel_before_accept": time.sleep(22)
        emit(command, "accepted", event_seq=0, **identity)
        if selected == "fault":
            time.sleep(0.8)
            sys.exit(0)
        if selected == "late_tokens": time.sleep(0.8)
        text = "prefix STOP hidden" if selected == "stop" else "visible"
        emit(command, "tokens", event_seq=1, output_index=0, token_ids=[10], text_delta=text, **identity)
        if selected == "stop":
            stopped = read()
            control(stopped, "accepted")
            time.sleep(0.8)
        emit(command, "terminal", event_seq=2, status="completed",
             cause="stop_sequence" if selected == "stop" else "length",
             usage=dict(input_tokens=2, output_tokens=1),
             state_result=dict(validity="none", consumed_position=3, history_count=3),
             text_delta="", metrics=dict(elapsed_ns=20, ttft_ns=10, peak_memory_bytes=2048), **identity)
        if selected == "terminal_before_deadline": time.sleep(0.8)
        if selected == "cancel_after_terminal": time.sleep(22)
        emit(command, "resources_released", event_seq=3,
             released_lease_ids=command["capacity_lease_ids"], retained_lease_ids=[], **identity)
    elif kind in ("cancel_request", "stop_generation"):
        control(command)
    else:
        raise RuntimeError("The CPU probe received an unexpected command.")
"#;
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut child = Command::new("python3")
        .args(["-u", "-c", script, mode])
        .arg(&probe.0)
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
    let ready = json!({"worker_epoch":epoch,"model_revision":"1".repeat(64),
        "capability_revision":"2".repeat(64),"eos_token_ids":[2],
        "capabilities":{"max_context":64}});
    let (sender, stopped) = watch::channel(None);
    let service = Arc::new(Service {
        queue: PendingQueue::new(2),
        host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
        ready_state: RwLock::new(ready.clone()),
        ready,
        stopped,
        model_id: "generation-deadline-probe".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout: if mode.starts_with("cancel_") {
            Duration::from_secs(60)
        } else {
            Duration::from_millis(180)
        },
        memory_budget: 4096,
    });
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        || async { Err("The deadline probe must not rotate.".to_owned()) },
        sender,
    ));
    (service, task)
}

fn request() -> Request {
    Request {
        messages: vec![json!({"role":"user","content":"original generation deadline probe"})],
        tools: vec![],
        max_tokens: 1,
        stops: vec![],
        count_only: false,
    }
}

async fn run_probe<F, Fut>(mode: &str, check: F)
where
    F: FnOnce(Arc<Service>, Arc<GenerationProbe>) -> Fut,
    Fut: Future<Output = ()>,
{
    let probe = Arc::new(GenerationProbe::new());
    let (service, task) = service_for(&probe, mode).await;
    // A red assertion must still close admission and wait for the owned process.
    let result = std::panic::AssertUnwindSafe(check(service.clone(), probe))
        .catch_unwind()
        .await;
    service.shutdown();
    let stopped = tokio::time::timeout(Duration::from_secs(3), service.wait_stopped()).await;
    assert!(
        stopped.is_ok(),
        "The CPU probe must finish process reaping."
    );
    tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("The CPU coordinator must stop.")
        .unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}

fn metric(service: &Service, name: &str, status: Option<&str>) -> f64 {
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

fn assert_public_expiry_holds_resources(service: &Service) {
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(metric(service, "apxinf_cleanup_pending", None), 1.0);
    assert_eq!(
        metric(service, "apxinf_request_outcomes_total", Some("expired")),
        1.0
    );
}

async fn require_expiry(ticket: &mut Ticket) {
    let event = tokio::time::timeout(Duration::from_millis(450), ticket.output.recv())
        .await
        .expect("The public deadline must not wait for worker settlement.");
    match event {
        Some(Event::Error(error)) => assert_eq!(error.code, "deadline_exceeded"),
        _ => panic!("The expired request must return its deadline error."),
    }
}

async fn require_no_late_output(ticket: &mut Ticket, service: &Service) {
    assert!(
        tokio::time::timeout(Duration::from_secs(2), ticket.output.recv())
            .await
            .expect("The settled attempt must close its output channel.")
            .is_none()
    );
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(metric(service, "apxinf_cleanup_pending", None), 0.0);
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
}

async fn next_request(service: &Service) {
    let mut next = service.enqueue(request()).unwrap();
    assert!(next.begin.await.unwrap().is_ok());
    assert!(matches!(next.output.recv().await, Some(Event::Part(_))));
    assert!(matches!(next.output.recv().await, Some(Event::Done { .. })));
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.expired.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn generation_deadline_before_acceptance_ends_begin_without_late_success() {
    run_probe("delayed_accept", |service, _probe| async move {
        let mut ticket = service.enqueue(request()).unwrap();
        let begin = tokio::time::timeout(Duration::from_millis(450), &mut ticket.begin)
            .await
            .expect("A delayed accepted event must not extend the public deadline.")
            .unwrap();
        match begin {
            Err(error) => assert_eq!(error.code, "deadline_exceeded"),
            Ok(_) => panic!("Expired admission must not return success."),
        }
        assert_public_expiry_holds_resources(&service);
        require_no_late_output(&mut ticket, &service).await;
        assert_eq!(
            metric(
                &service,
                "apxinf_time_to_first_public_output_ready_seconds_count",
                None
            ),
            0.0
        );
        next_request(&service).await;
    })
    .await;
}

#[tokio::test]
async fn generation_deadline_after_terminal_waits_for_cleanup_without_public_success() {
    run_probe("terminal_before_deadline", |service, _probe| async move {
        let mut ticket = service.enqueue(request()).unwrap();
        assert!((&mut ticket.begin).await.unwrap().is_ok());
        assert!(matches!(ticket.output.recv().await, Some(Event::Part(_))));
        require_expiry(&mut ticket).await;
        assert_public_expiry_holds_resources(&service);
        require_no_late_output(&mut ticket, &service).await;
        assert_eq!(
            metric(
                &service,
                "apxinf_worker_terminal_to_settlement_seconds_count",
                None
            ),
            1.0
        );
        next_request(&service).await;
    })
    .await;
}

#[tokio::test]
async fn generation_deadline_after_stop_control_does_not_send_a_second_control() {
    run_probe("stop", |service, probe| async move {
    let mut stopped = request();
    stopped.stops = vec!["STOP".into()];
    let mut ticket = service.enqueue(stopped).unwrap();
    assert!((&mut ticket.begin).await.unwrap().is_ok());
    assert!(matches!(ticket.output.recv().await, Some(Event::Part(Part::Text(text))) if text == "prefix "));
    require_expiry(&mut ticket).await;
    assert_public_expiry_holds_resources(&service);
    require_no_late_output(&mut ticket, &service).await;
    assert_eq!(probe.commands(), ["prepare_input", "submit", "stop_generation"]);
    next_request(&service).await;
    }).await;
}

#[tokio::test]
async fn generation_deadline_suppresses_tokens_that_arrive_after_public_expiry() {
    run_probe("late_tokens", |service, _probe| async move {
        let mut ticket = service.enqueue(request()).unwrap();
        assert!((&mut ticket.begin).await.unwrap().is_ok());
        require_expiry(&mut ticket).await;
        assert_public_expiry_holds_resources(&service);
        require_no_late_output(&mut ticket, &service).await;
        assert_eq!(
            metric(
                &service,
                "apxinf_time_to_first_worker_event_seconds_count",
                None
            ),
            1.0
        );
        assert_eq!(
            metric(
                &service,
                "apxinf_time_to_first_public_output_ready_seconds_count",
                None
            ),
            0.0
        );
        next_request(&service).await;
    })
    .await;
}

#[tokio::test]
async fn generation_deadline_survives_later_worker_fault_without_a_second_outcome() {
    run_probe("fault", |service, _probe| async move {
        let mut ticket = service.enqueue(request()).unwrap();
        assert!((&mut ticket.begin).await.unwrap().is_ok());
        require_expiry(&mut ticket).await;
        assert_public_expiry_holds_resources(&service);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), service.wait_stopped())
                .await
                .unwrap()
                .is_err()
        );
        require_no_late_output(&mut ticket, &service).await;
        assert!(!service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
        assert_eq!(
            metric(&service, "apxinf_request_outcomes_total", Some("failed")),
            0.0
        );
        assert_eq!(
            metric(&service, "apxinf_request_outcomes_total", Some("expired")),
            1.0
        );
    })
    .await;
}

async fn cancellation_without_control(mode: &str) {
    let after_terminal = mode == "cancel_after_terminal";
    run_probe(mode, |service, probe| async move {
        let mut ticket = service.enqueue(request()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            if after_terminal {
                assert!((&mut ticket.begin).await.unwrap().is_ok());
                while service.metrics.peak_bytes.load(Ordering::Relaxed) != 2048 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            } else {
                while !probe.commands().iter().any(|command| command == "submit") {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        })
        .await
        .expect("The worker must reach the cancellation boundary.");
        if after_terminal {
            drop(ticket.output);
        } else {
            ticket.cancel.store(true, Ordering::Release);
        }
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
        let stopped = tokio::time::timeout(
            SETTLEMENT_GRACE + Duration::from_millis(700),
            service.wait_stopped(),
        )
        .await
        .expect("Cancellation must start the grace before acceptance and after terminal.");
        assert!(stopped.is_err(), "A stalled worker must require recovery.");
        assert!(!service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
        assert_eq!(probe.commands(), ["prepare_input", "submit"]);
    })
    .await;
}

#[tokio::test]
async fn cancellation_before_acceptance_starts_grace_without_control() {
    cancellation_without_control("cancel_before_accept").await;
}

#[tokio::test]
async fn cancellation_after_terminal_starts_grace_without_control() {
    cancellation_without_control("cancel_after_terminal").await;
}
