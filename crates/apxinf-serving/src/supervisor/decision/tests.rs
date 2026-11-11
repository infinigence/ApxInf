use super::*;
use crate::decision::parse_decision_request;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Probe(PathBuf);
impl Probe {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("apxinf-clef-cpu-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn events(&self) -> Vec<Value> {
        std::fs::read_to_string(self.0.join("trace.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    async fn event(&self, name: &str) {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if self.events().iter().any(|event| event["event"] == name) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
    fn release(&self) {
        std::fs::write(self.0.join("release"), b"release").unwrap();
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn worker(root: PathBuf, mode: &str) -> Result<(Worker, Value), String> {
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut child = Command::new("python3")
        .arg("-u")
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/serving/clef_cpu_worker.py"),
        )
        .arg("--root")
        .arg(root)
        .arg("--epoch")
        .arg(&epoch)
        .arg("--mode")
        .arg(mode)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| error.to_string())?;
    let input = child.stdin.take();
    let frames =
        read_worker_frames_for_protocol(child.stdout.take().unwrap(), contracts::CLEF_PROTOCOL);
    let worker = Worker {
        protocol: contracts::CLEF_PROTOCOL,
        child,
        _diagnostics: None,
        input,
        frames,
        epoch: epoch.clone(),
        controls: Default::default(),
        sent_commands: 0,
        retired: false,
    };
    let ready = json!({"protocol":contracts::CLEF_PROTOCOL,"worker_epoch":epoch,"model_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","capability_revision":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","capabilities":{"max_context":4096,"max_output_tokens":0,"vocab_size":248320},"memory":{"active_bytes":128,"peak_bytes":128,"cache_bytes":0}});
    Ok((worker, ready))
}

async fn start(
    probe: &Probe,
    mode: &str,
    timeout: Duration,
    replacement_delay: Duration,
) -> (Arc<Service>, tokio::task::JoinHandle<StopResult>) {
    let (initial_worker, ready) = worker(probe.0.clone(), mode).await.unwrap();
    let service = super::super::lifecycle_regressions::cpu_service(ready, timeout, 4);
    start_queue_sweeper(&service);
    let root = probe.0.clone();
    let task = tokio::spawn(coordinate(initial_worker, service.clone(), 64, move || {
        let root = root.clone();
        async move {
            tokio::time::sleep(replacement_delay).await;
            worker(root, "success").await
        }
    }));
    (service, task)
}
fn request() -> crate::decision::DecisionRequest {
    parse_decision_request(br#"{"model":"cpu-fixture","state":"warm","questions":{"grade":{"type":"score","instructions":"Rate","criteria":["low","high"]},"choice":{"type":"choice","instructions":"Choose","criteria":{"a":"A","b":"B"}},"warm":{"type":"noul","instructions":"Warm?"}}}"#).unwrap()
}
async fn stop(service: &Service, task: tokio::task::JoinHandle<StopResult>) {
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
async fn available(service: &Service) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !service.available.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

fn decision_metric(service: &Service, name: &str) -> f64 {
    let prefix = format!("{name}{{operation=\"decision\"}} ");
    service
        .metrics_text()
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap()
        .parse()
        .unwrap()
}

async fn stopped_attempt(
    probe: &Probe,
    mode: &str,
) -> (Worker, Arc<Service>, DecisionTracker, Value, Vec<Value>) {
    let (mut worker, ready) = worker(probe.0.clone(), mode).await.unwrap();
    let service =
        super::super::lifecycle_regressions::cpu_service(ready, Duration::from_secs(10), 4);
    let mut prepare = worker.command("prepare_decision");
    prepare["request_id"] = json!(uuid::Uuid::new_v4().to_string());
    prepare["attempt"] = json!(1);
    prepare["model_revision"] = service.ready["model_revision"].clone();
    prepare["capability_revision"] = service.ready["capability_revision"].clone();
    prepare["record"] = json!(request().record);
    let tracker = DecisionTracker::new(&prepare).unwrap();
    worker.send(&prepare).await.unwrap();
    let prepared = worker.receive(Duration::from_secs(5)).await.unwrap();
    let mut submit = worker.command("submit_decision");
    submit["request_id"] = prepare["request_id"].clone();
    submit["attempt"] = json!(1);
    submit["model_revision"] = service.ready["model_revision"].clone();
    submit["capability_revision"] = service.ready["capability_revision"].clone();
    submit["preparation"] = prepared["preparation"].clone();
    submit["capacity_lease_ids"] = json!([uuid::Uuid::new_v4().to_string()]);
    worker.send(&submit).await.unwrap();
    let mut frames = vec![prepared];
    for _ in 0..3 {
        frames.push(worker.receive(Duration::from_secs(5)).await.unwrap());
    }
    assert!(worker.child.wait().await.unwrap().success());
    assert!(worker.frames.recv().await.is_none());
    (worker, service, tracker, submit, frames)
}

fn pending_frames(worker: &mut Worker, frames: impl IntoIterator<Item = Value>) {
    let (sender, receiver) = mpsc::channel(4);
    for frame in frames {
        sender.try_send(Ok(frame)).unwrap();
    }
    worker.frames = receiver;
}

#[tokio::test]
async fn checked_decision_frames_preserve_preparation_and_cleanup_intervals() {
    let probe = Probe::new();
    let (_, service, mut tracker, submit, frames) =
        stopped_attempt(&probe, "inference_failure").await;
    let start = Instant::now();
    let observation = RequestObservation::decision(start);
    observation.preparation_start(start);
    check_and_observe_frame(
        &mut tracker,
        &service,
        &observation,
        &frames[0],
        start + Duration::from_millis(20),
    )
    .unwrap();
    tracker.observe(&submit).unwrap();
    for frame in &frames[1..] {
        check_and_observe_frame(
            &mut tracker,
            &service,
            &observation,
            frame,
            start + Duration::from_millis(60),
        )
        .unwrap();
    }
    observation.public_terminal(
        &service.metrics.observations,
        Outcome::Failed,
        start + Duration::from_millis(70),
    );
    observation.settled(
        &service.metrics.observations,
        start + Duration::from_millis(90),
    );
    assert_eq!(
        decision_metric(&service, "apxinf_preparation_seconds_count"),
        1.0
    );
    assert_eq!(
        decision_metric(&service, "apxinf_preparation_seconds_sum"),
        0.02
    );
    assert_eq!(
        decision_metric(&service, "apxinf_worker_terminal_to_settlement_seconds_sum"),
        0.03
    );
    assert_eq!(
        decision_metric(&service, "apxinf_public_terminal_to_settlement_seconds_sum"),
        0.02
    );
    assert_eq!(
        decision_metric(&service, "apxinf_worker_first_token_seconds_count"),
        0.0
    );
}

#[tokio::test]
async fn stopped_decision_preparation_ends_before_resource_settlement() {
    let probe = Probe::new();
    let (mut worker, service, mut tracker, _, frames) = stopped_attempt(&probe, "success").await;
    pending_frames(&mut worker, frames.into_iter().take(1));
    let observation = RequestObservation::decision(Instant::now());
    observation.preparation_start(Instant::now());
    observation.public_terminal(
        &service.metrics.observations,
        Outcome::Cancelled,
        Instant::now(),
    );
    check_stopped_frames(&mut worker, &service, &mut tracker, &observation)
        .await
        .unwrap();
    let preparation_count = decision_metric(&service, "apxinf_preparation_seconds_count");
    observation.settled(&service.metrics.observations, Instant::now());
    assert_eq!(preparation_count, 1.0);
    assert_eq!(
        decision_metric(&service, "apxinf_preparation_seconds_count"),
        1.0
    );
    assert_eq!(
        decision_metric(
            &service,
            "apxinf_worker_terminal_to_settlement_seconds_count"
        ),
        0.0
    );
}

#[tokio::test]
async fn stopped_decision_terminals_record_cleanup_without_replacing_cancellation() {
    for mode in ["success", "inference_failure"] {
        let probe = Probe::new();
        let (mut worker, service, mut tracker, submit, mut frames) =
            stopped_attempt(&probe, mode).await;
        tracker.observe(&frames.remove(0)).unwrap();
        tracker.observe(&submit).unwrap();
        if mode == "success" {
            frames[1]["metrics"]["peak_device_bytes"] = json!(1024);
        }
        pending_frames(&mut worker, frames);
        let observation = RequestObservation::decision(Instant::now());
        observation.preparation_start(Instant::now());
        observation.preparation_end(&service.metrics.observations, Instant::now());
        observation.public_terminal(
            &service.metrics.observations,
            Outcome::Cancelled,
            Instant::now(),
        );
        check_stopped_frames(&mut worker, &service, &mut tracker, &observation)
            .await
            .unwrap();
        observation.settled(&service.metrics.observations, Instant::now());
        assert_eq!(
            decision_metric(&service, "apxinf_preparation_seconds_count"),
            1.0,
            "{mode}"
        );
        assert_eq!(
            decision_metric(
                &service,
                "apxinf_worker_terminal_to_settlement_seconds_count"
            ),
            1.0,
            "{mode}"
        );
        assert_eq!(
            service.metrics.peak_bytes.load(Ordering::Relaxed),
            if mode == "success" { 1024 } else { 0 }
        );
        let metrics = service.metrics_text();
        assert!(metrics.contains(
            "apxinf_request_outcomes_total{operation=\"decision\",status=\"cancelled\"} 1"
        ));
        assert!(metrics.contains(
            "apxinf_request_outcomes_total{operation=\"decision\",status=\"completed\"} 0"
        ));
    }
}

#[tokio::test]
async fn stopped_decision_invalid_frames_add_no_observations() {
    for invalid in [
        "epoch",
        "command",
        "probabilities",
        "prepared_digest",
        "prepared_context",
    ] {
        let probe = Probe::new();
        let (mut worker, service, mut tracker, submit, mut frames) =
            stopped_attempt(&probe, "success").await;
        if invalid.starts_with("prepared") {
            frames.truncate(1);
            if invalid == "prepared_digest" {
                frames[0]["preparation"]["preparation_digest"] = json!("0".repeat(64));
            }
        } else {
            tracker.observe(&frames.remove(0)).unwrap();
            tracker.observe(&submit).unwrap();
            tracker.observe(&frames.remove(0)).unwrap();
            frames.truncate(1);
            frames[0]["metrics"]["peak_device_bytes"] = json!(2048);
            match invalid {
                "epoch" => frames[0]["worker_epoch"] = json!(uuid::Uuid::new_v4().to_string()),
                "command" => frames[0]["command_id"] = json!(uuid::Uuid::new_v4().to_string()),
                "probabilities" => frames[0]["probabilities"][0]["values"][0] = json!(2.0),
                _ => unreachable!(),
            }
        }
        let service = if invalid == "prepared_context" {
            let mut ready = service.ready.clone();
            ready["capabilities"]["max_context"] = json!(1);
            super::super::lifecycle_regressions::cpu_service(ready, Duration::from_secs(10), 4)
        } else {
            service
        };
        pending_frames(&mut worker, frames);
        let observation = RequestObservation::decision(Instant::now());
        observation.preparation_start(Instant::now());
        let error = check_stopped_frames(&mut worker, &service, &mut tracker, &observation)
            .await
            .unwrap_err();
        assert_eq!(error.code, "protocol_fault", "{invalid}");
        assert_eq!(
            decision_metric(&service, "apxinf_preparation_seconds_count"),
            0.0,
            "{invalid}"
        );
        observation.settled(&service.metrics.observations, Instant::now());
        assert_eq!(
            decision_metric(
                &service,
                "apxinf_worker_terminal_to_settlement_seconds_count"
            ),
            0.0,
            "{invalid}"
        );
        assert_eq!(
            service.metrics.peak_bytes.load(Ordering::Relaxed),
            0,
            "{invalid}"
        );
    }
}

#[tokio::test]
async fn decision_result_stays_private_until_clean_process_reap() {
    let probe = Probe::new();
    let (service, task) = start(
        &probe,
        "result_hold",
        Duration::from_secs(10),
        Duration::ZERO,
    )
    .await;
    let mut ticket = service.enqueue_decision(request()).unwrap();
    assert_eq!(ticket.begin.await.unwrap().unwrap().input_tokens, 9);
    probe.event("result_pending").await;
    assert!(ticket.output.try_recv().is_err());
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 64);
    probe.release();
    assert!(
        matches!(ticket.output.recv().await,Some(Event::Decision(result)) if result.input_tokens==9)
    );
    available(&service).await;
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
    assert!(service
        .metrics_text()
        .contains("operation=\"decision\",status=\"completed\"} 1"));
    stop(&service, task).await;
}

#[tokio::test]
async fn decision_nonzero_exit_and_extra_output_discard_pending_success() {
    for mode in ["nonzero_exit", "extra_output"] {
        let probe = Probe::new();
        let (service, task) = start(&probe, mode, Duration::from_secs(10), Duration::ZERO).await;
        let mut ticket = service.enqueue_decision(request()).unwrap();
        ticket.begin.await.unwrap().unwrap();
        assert!(
            matches!(ticket.output.recv().await, Some(Event::Error(_))),
            "{mode}"
        );
        assert!(task.await.unwrap().is_err());
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn decision_context_rejection_reaps_and_loads_a_replacement() {
    let probe = Probe::new();
    let (service, task) = start(
        &probe,
        "context_limit",
        Duration::from_secs(10),
        Duration::ZERO,
    )
    .await;
    let ticket = service.enqueue_decision(request()).unwrap();
    let error = ticket.begin.await.unwrap().err().unwrap();
    assert_eq!((error.status, error.code.as_str()), (400, "context_limit"));
    available(&service).await;
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
    assert!(!probe
        .events()
        .iter()
        .any(|event| event["event"] == "submitted"));
    let cleanup_count = decision_metric(
        &service,
        "apxinf_worker_terminal_to_settlement_seconds_count",
    );
    let mut next = service.enqueue_decision(request()).unwrap();
    next.begin.await.unwrap().unwrap();
    assert!(matches!(next.output.recv().await, Some(Event::Decision(_))));
    stop(&service, task).await;
    assert_eq!(
        cleanup_count, 1.0,
        "A checked Clef failure must contribute one worker cleanup sample."
    );
}

#[tokio::test]
async fn decision_inference_failure_reaps_and_records_worker_cleanup() {
    let probe = Probe::new();
    let (service, task) = start(
        &probe,
        "inference_failure",
        Duration::from_secs(10),
        Duration::ZERO,
    )
    .await;
    let mut ticket = service.enqueue_decision(request()).unwrap();
    assert_eq!(ticket.begin.await.unwrap().unwrap().input_tokens, 9);
    assert!(
        matches!(ticket.output.recv().await, Some(Event::Error(error)) if error.status == 500 && error.code == "internal_error")
    );
    assert!(ticket.output.recv().await.is_none());
    available(&service).await;
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    let metrics = service.metrics_text();
    assert!(metrics
        .contains("apxinf_request_outcomes_total{operation=\"decision\",status=\"failed\"} 1"));
    let cleanup_count = decision_metric(
        &service,
        "apxinf_worker_terminal_to_settlement_seconds_count",
    );
    let pid = probe
        .events()
        .iter()
        .find(|event| event["event"] == "submitted")
        .unwrap()["pid"]
        .as_u64()
        .unwrap();
    let alive = Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .unwrap()
        .success();
    assert!(
        !alive,
        "The failed decision child still exists after settlement."
    );
    let mut next = service.enqueue_decision(request()).unwrap();
    next.begin.await.unwrap().unwrap();
    assert!(matches!(next.output.recv().await, Some(Event::Decision(_))));
    stop(&service, task).await;
    assert_eq!(
        cleanup_count, 1.0,
        "A checked inference failure must contribute one worker cleanup sample."
    );
}

#[tokio::test]
async fn decision_deadline_and_cancel_kill_before_capacity_release() {
    for (mode, cancel) in [
        ("prepare_hold", false),
        ("accepted_hold", true),
        ("result_hold", true),
    ] {
        let probe = Probe::new();
        let (service, task) = start(&probe, mode, Duration::from_millis(300), Duration::ZERO).await;
        let mut ticket = service.enqueue_decision(request()).unwrap();
        if mode == "prepare_hold" {
            assert_eq!(
                ticket.begin.await.unwrap().err().unwrap().code,
                "deadline_exceeded"
            );
        } else {
            ticket.begin.await.unwrap().unwrap();
            probe
                .event(if mode == "accepted_hold" {
                    "accepted"
                } else {
                    "result_pending"
                })
                .await;
            if cancel {
                ticket.cancel.store(true, Ordering::Release);
            }
            assert!(
                matches!(ticket.output.recv().await,Some(Event::Error(error)) if error.code=="cancelled")
            );
        }
        available(&service).await;
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
        stop(&service, task).await;
    }
}

#[tokio::test]
async fn decision_queue_expires_during_replacement_loading() {
    let probe = Probe::new();
    let (service, task) = start(
        &probe,
        "result_hold",
        Duration::from_millis(300),
        Duration::from_millis(600),
    )
    .await;
    let mut first = service.enqueue_decision(request()).unwrap();
    first.begin.await.unwrap().unwrap();
    probe.event("result_pending").await;
    let queued = service.enqueue_decision(request()).unwrap();
    probe.release();
    assert!(matches!(
        first.output.recv().await,
        Some(Event::Decision(_))
    ));
    let error = queued.begin.await.unwrap().err().unwrap();
    assert_eq!(error.code, "deadline_exceeded");
    assert!(!service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
    stop(&service, task).await;
}

#[tokio::test]
async fn decision_http_projection_and_schema_rejection() {
    let probe = Probe::new();
    let (service, task) = start(&probe, "success", Duration::from_secs(10), Duration::ZERO).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = crate::gateway::router(service.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    for (path,body,status) in [("/v1/systemone",serde_json::to_string(&json!({"model":"cpu-fixture","state":request().record.state,"questions":request().record.questions})).unwrap(),"200"),("/v1/systemone","{\"model\":\"a\",\"model\":\"b\"}".into(),"400"),("/v1/messages","{}".into(),"400")]{
        if path=="/v1/systemone" && status=="200"{available(&service).await;}
        let mut stream=tokio::net::TcpStream::connect(address).await.unwrap();let bytes=format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());stream.write_all(bytes.as_bytes()).await.unwrap();let mut response=String::new();stream.read_to_string(&mut response).await.unwrap();assert!(response.starts_with(&format!("HTTP/1.1 {status}")),"{response}");
        if status=="200"{let (_,body)=response.split_once("\r\n\r\n").unwrap();let result:Value=serde_json::from_str(body).unwrap();assert_eq!(result["usage"],json!({"input_tokens":9,"output_tokens":0}));assert_eq!(result["answers"]["choice"]["choice"],"a");assert_eq!(result["answers"]["grade"]["score"],0.5);assert_eq!(result["answers"]["warm"]["noul"],0.5);}
    }
    server.abort();
    stop(&service, task).await;
}

#[tokio::test]
async fn decision_disconnect_and_service_shutdown_reap_owned_processes() {
    for shutdown in [false, true] {
        let probe = Probe::new();
        let (service, task) = start(
            &probe,
            "accepted_hold",
            Duration::from_secs(10),
            Duration::ZERO,
        )
        .await;
        let mut ticket = service.enqueue_decision(request()).unwrap();
        ticket.begin.await.unwrap().unwrap();
        probe.event("accepted").await;
        let pid = probe
            .events()
            .iter()
            .find(|event| event["event"] == "accepted")
            .unwrap()["pid"]
            .as_u64()
            .unwrap();
        if shutdown {
            service.shutdown();
            assert!(
                matches!(ticket.output.recv().await,Some(Event::Error(error)) if error.code=="worker_lost")
            );
            task.await.unwrap().unwrap();
        } else {
            ticket.output.close();
            tokio::time::timeout(Duration::from_secs(3), async {
                while service.metrics.cancelled.load(Ordering::Relaxed) == 0 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap();
            available(&service).await;
            stop(&service, task).await;
        }
        let alive = Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .unwrap()
            .success();
        assert!(
            !alive,
            "The decision child still exists after confirmed settlement."
        );
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 0);
    }
}
