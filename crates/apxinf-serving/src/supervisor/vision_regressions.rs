use super::*;

struct Probe(PathBuf);

impl Probe {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("apxinf-vision-service-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn commands(&self) -> Vec<Value> {
        std::fs::read_to_string(self.0.join("commands.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    async fn prepared(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !self.0.join("prepared").exists() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn corpus() -> Value {
    serde_json::from_str(include_str!(
        "../../../../tests/fixtures/serving/minicpm-vision-v0.1.json"
    ))
    .unwrap()
}

async fn worker(probe: &Probe, mode: &str) -> Worker {
    let script = r#"
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
templates = json.loads(pathlib.Path(sys.argv[3]).read_text())["templates"]
for line in sys.stdin:
    command = json.loads(line)
    with (root / "commands.jsonl").open("a") as trace:
        trace.write(json.dumps(command) + "\n")
    if command["kind"] == "prepare_input":
        event = json.loads(json.dumps(templates["prepared"]))
        event["command_id"] = command["command_id"]
        event["replay"]["preparation_id"] = command["command_id"]
        if "processor" in mode:
            event["replay"]["processor_revision"] = "0" * 64
        if "capability" in mode:
            event["replay"]["capability_revision"] = "0" * 64
        if "media" in mode:
            event["replay"]["media"] = []
            event["replay"]["image_bindings"] = []
            event["token_ids"] = [1, 3]
            event["effective_input_tokens"] = 2
            import hashlib, struct
            event["prompt_digest"] = hashlib.sha256(b"apxinf-token-prefix-v1\0" + struct.pack("<QII", 2, 1, 3)).hexdigest()
        (root / "prepared").touch()
        if "late" in mode:
            while not (root / "resume").exists(): time.sleep(0.002)
        (root / "response.json").write_text(json.dumps(event))
        print(json.dumps(event), flush=True)
    elif command["kind"] == "submit":
        for name in ("accepted", "tokens", "terminal", "released"):
            event = json.loads(json.dumps(templates[name]))
            event["request_id"] = command["request_id"]
            if name == "released": event["released_lease_ids"] = command["capacity_lease_ids"]
            print(json.dumps(event), flush=True)
    else:
        raise RuntimeError("unexpected command")
"#;
    let mut child = Command::new("python3")
        .args(["-u", "-c", script])
        .arg(&probe.0)
        .arg(mode)
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/serving/minicpm-vision-v0.1.json"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    Worker {
        protocol: contracts::VISION_PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: corpus()["templates"]["ready"]["worker_epoch"]
            .as_str()
            .unwrap()
            .into(),
        controls: Default::default(),
        sent_commands: 0,
        retired: false,
    }
}

fn service() -> Service {
    let ready = corpus()["templates"]["ready"].clone();
    Service {
        queue: PendingQueue::new(4),
        host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
        ready_state: RwLock::new(ready.clone()),
        ready,
        stopped: watch::channel(None).1,
        model_id: "vision-probe".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout: Duration::from_secs(5),
        memory_budget: 4194304,
    }
}

fn request() -> Request {
    Request {
        selection: crate::sampling::Selection::Random {
            temperature: 0.01,
            top_p: 0.95,
        },
        messages: corpus()["templates"]["prepare"]["messages"]
            .as_array()
            .unwrap()
            .clone(),
        tools: vec![],
        max_tokens: 2,
        stops: vec![],
        count_only: false,
    }
}

#[tokio::test]
async fn submit_retains_checked_replay_ids_and_valid_temperature() {
    let probe = Probe::new();
    let mut worker = worker(&probe, "normal").await;
    let service = service();
    let mut ticket = service.enqueue(request()).unwrap();
    let job = service.queue.recv(&service.metrics).await.unwrap();
    let result = execute(&mut worker, &service, job, service.timeout).await;
    assert!(
        result.is_ok(),
        "{}",
        result
            .err()
            .map(|failure| failure.error.message)
            .unwrap_or_default()
    );
    assert_eq!(ticket.begin.await.unwrap().unwrap().input_tokens, 3);
    assert!(
        matches!(ticket.output.recv().await, Some(Event::Part(Part::Text(text))) if text == "red")
    );
    assert!(matches!(
        ticket.output.recv().await,
        Some(Event::Done {
            output_tokens: 2,
            ..
        })
    ));
    let commands = probe.commands();
    assert_eq!(commands.len(), 2);
    let response: Value =
        serde_json::from_slice(&std::fs::read(probe.0.join("response.json")).unwrap()).unwrap();
    assert_eq!(commands[1]["replay"], response["replay"]);
    assert_eq!(commands[1]["token_ids"], response["token_ids"]);
    assert_eq!(commands[1]["sampling"]["temperature"].as_f64(), Some(0.01));
    assert_eq!(
        commands[1]["replay"]["preparation_id"],
        commands[0]["command_id"]
    );
    assert_ne!(commands[1]["command_id"], commands[0]["command_id"]);
    stop_child(&mut worker.child).await.unwrap();
}

#[tokio::test]
async fn count_only_and_context_rejection_never_submit() {
    for count_only in [true, false] {
        let probe = Probe::new();
        let mut worker = worker(&probe, "normal").await;
        let mut service = service();
        let mut request = request();
        request.count_only = count_only;
        if !count_only {
            service.ready["capabilities"]["max_context"] = json!(4);
        }
        let ticket = service.enqueue(request).unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        assert!(execute(&mut worker, &service, job, service.timeout)
            .await
            .is_ok());
        let begin = ticket.begin.await.unwrap();
        if count_only {
            assert_eq!(begin.unwrap().input_tokens, 3);
        } else {
            assert_eq!(begin.err().unwrap().code, "context_limit");
        }
        assert_eq!(probe.commands().len(), 1);
        stop_child(&mut worker.child).await.unwrap();
    }
}

#[tokio::test]
async fn replay_tampering_fails_even_after_public_cancellation() {
    for mode in [
        "processor",
        "capability",
        "media",
        "late_processor",
        "late_capability",
        "late_media",
    ] {
        let probe = Probe::new();
        let mut worker = worker(&probe, mode).await;
        let service = service();
        let ticket = service.enqueue(request()).unwrap();
        let job = service.queue.recv(&service.metrics).await.unwrap();
        let cancellation = async {
            probe.prepared().await;
            if mode.starts_with("late") {
                ticket.cancel.store(true, Ordering::Release);
                tokio::time::sleep(Duration::from_millis(50)).await;
                std::fs::write(probe.0.join("resume"), []).unwrap();
            }
        };
        let (result, _) = tokio::join!(
            execute(&mut worker, &service, job, service.timeout),
            cancellation
        );
        let failure = result.err().unwrap();
        assert_eq!(
            failure.error.code, "protocol_fault",
            "{mode}: {}",
            failure.error.message
        );
        assert!(failure.error.message.contains("prepared vision replay"));
        assert!(failure.reported);
        let begin = ticket.begin.await.unwrap().err().unwrap();
        assert_eq!(
            begin.code,
            if mode.starts_with("late") {
                "cancelled"
            } else {
                "protocol_fault"
            }
        );
        assert_eq!(probe.commands().len(), 1);
        stop_child(&mut worker.child).await.unwrap();
    }
}

#[tokio::test]
async fn matching_late_replay_is_discarded_without_submit() {
    let probe = Probe::new();
    let mut worker = worker(&probe, "late").await;
    let service = service();
    let ticket = service.enqueue(request()).unwrap();
    let job = service.queue.recv(&service.metrics).await.unwrap();
    let cancellation = async {
        probe.prepared().await;
        ticket.cancel.store(true, Ordering::Release);
        tokio::time::sleep(Duration::from_millis(50)).await;
        std::fs::write(probe.0.join("resume"), []).unwrap();
    };
    let (result, _) = tokio::join!(
        execute(&mut worker, &service, job, service.timeout),
        cancellation
    );
    assert!(result.is_ok());
    assert_eq!(ticket.begin.await.unwrap().err().unwrap().code, "cancelled");
    assert_eq!(probe.commands().len(), 1);
    stop_child(&mut worker.child).await.unwrap();
}
