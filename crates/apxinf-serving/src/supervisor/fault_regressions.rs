//! CPU worker failures must produce explicit public results before channels close.
use super::lifecycle_regressions::cpu_service;
use super::*;

struct FaultResult {
    service: Arc<Service>,
    begin: Result<Result<Begin, ApiError>, oneshot::error::RecvError>,
    output: Vec<Event>,
    waiting: Result<Result<Begin, ApiError>, oneshot::error::RecvError>,
    available_at_begin: bool,
    available_at_waiting_result: bool,
    stopped: StopResult,
}

async fn fault_probe(mode: &str) -> FaultResult {
    let script = r#"
import hashlib, json, struct, sys, time
mode = sys.argv[1]
for line in sys.stdin:
    command = json.loads(line)
    def emit(kind, **fields):
        print(json.dumps(dict(protocol=command['protocol'], worker_epoch=command['worker_epoch'],
                              kind=kind, **fields)), flush=True)
    if command['kind'] == 'prepare_input':
        identity = command['command_id']
        if mode == 'preparation': identity = '00000000-0000-4000-8000-000000000099'
        digest = hashlib.sha256(b'apxinf-token-prefix-v1\0' + struct.pack('<QII', 2, 3, 4)).hexdigest()
        emit('prepared_input', command_id=identity, model_revision=command['model_revision'],
             token_ids=[3, 4], effective_input_tokens=2, prompt_digest=digest)
        if mode == 'preparation': time.sleep(60)
    elif command['kind'] == 'submit':
        request_id = command['request_id']
        if mode == 'admission': request_id = '00000000-0000-4000-8000-000000000099'
        key = dict(request_id=request_id, attempt=command['attempt'])
        emit('accepted', event_seq=0, **key)
        if mode == 'admission': time.sleep(60)
        count = 64 if mode == 'full_output' else 3
        for index in range(count):
            emit('tokens', event_seq=index+1, output_index=index, token_ids=[10],
                 text_delta=str(index), **key)
        if mode == 'eof': sys.exit(0)
        emit('tokens', event_seq=count, output_index=count, token_ids=[10],
             text_delta='invalid sequence', **key)
        time.sleep(60)
"#;
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut child = Command::new("python3")
        .args(["-u", "-c", script, mode])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let worker = Worker {
        protocol: PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: epoch.clone(),
        controls: Default::default(),
        sent_commands: 0,
    };
    let ready = json!({"worker_epoch":epoch,"model_revision":"1".repeat(64),
        "capability_revision":"2".repeat(64),"eos_token_ids":[2],
        "capabilities":{"max_context":128}});
    let mut service = cpu_service(ready, Duration::from_secs(10), 2);
    let (completion, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    let request = Request {
        selection: crate::sampling::Selection::Greedy,
        messages: vec![json!({"role":"user", "content":"original fault delivery probe"})],
        tools: vec![],
        max_tokens: 65,
        stops: vec![],
        count_only: false,
    };
    let mut ticket = service.enqueue(request.clone()).unwrap();
    let waiting = service.enqueue(request).unwrap();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        || async { Err::<(Worker, Value), _>("The fault probe must not rotate.".to_owned()) },
        completion,
    ));
    let begin = tokio::time::timeout(Duration::from_secs(3), ticket.begin)
        .await
        .expect("A fault must end admission without waiting for the request deadline.");
    let available_at_begin = service.available.load(Ordering::Acquire);
    let waiting = tokio::time::timeout(Duration::from_secs(3), waiting.begin)
        .await
        .expect("A failed epoch must reject its waiting requests.");
    let available_at_waiting_result = service.available.load(Ordering::Acquire);
    // Do not read content until recovery completes. All 64 content slots can fill.
    let stopped = tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .expect("Recovery must reap the failed CPU worker.");
    task.await.unwrap();
    let mut output = Vec::new();
    while let Some(event) = ticket.output.recv().await {
        output.push(event);
    }
    FaultResult {
        service,
        begin,
        output,
        waiting,
        available_at_begin,
        available_at_waiting_result,
        stopped,
    }
}

fn check_recovery(result: &FaultResult) {
    assert!(result.stopped.is_err());
    assert!(!result.service.available.load(Ordering::Acquire));
    assert!(!result.available_at_waiting_result);
    assert!(result.service.queue.is_closed());
    assert!(matches!(&result.waiting, Ok(Err(error)) if error.code == "worker_lost"));
    assert_eq!(
        result.service.metrics.worker_faults.load(Ordering::Relaxed),
        1
    );
    assert_eq!(result.service.metrics.failed.load(Ordering::Relaxed), 2);
    assert_eq!(result.service.metrics.completed.load(Ordering::Relaxed), 0);
    assert_eq!(result.service.metrics.queued.load(Ordering::Relaxed), 0);
    assert_eq!(result.service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(
        result
            .service
            .metrics
            .reserved_bytes
            .load(Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn protocol_fault_before_admission_returns_the_checked_error() {
    for mode in ["preparation", "admission"] {
        let result = fault_probe(mode).await;
        check_recovery(&result);
        assert!(!result.available_at_begin);
        let error = match result.begin {
            Ok(Err(error)) => error,
            Err(error) => panic!("{mode}: admission lost its explicit protocol error: {error}"),
            Ok(Ok(_)) => panic!("{mode}: invalid identity must not admit the request"),
        };
        assert_eq!(error.code, "protocol_fault");
        assert_eq!(error.status, 500);
        assert!(result.output.is_empty());
    }
}

#[tokio::test]
async fn protocol_fault_uses_reserved_terminal_capacity_after_full_content() {
    let result = fault_probe("full_output").await;
    check_recovery(&result);
    assert!(matches!(result.begin, Ok(Ok(_))));
    assert_eq!(
        result.output.len(),
        OUTPUT_PART_CAPACITY + 1,
        "A full content queue must retain the explicit protocol failure."
    );
    for (index, event) in result.output[..OUTPUT_PART_CAPACITY].iter().enumerate() {
        assert!(matches!(event, Event::Part(Part::Text(text)) if text == &index.to_string()));
    }
    assert!(
        matches!(&result.output[OUTPUT_PART_CAPACITY], Event::Error(error)
        if error.code == "protocol_fault" && error.status == 500)
    );
}

#[tokio::test]
async fn worker_eof_after_output_has_one_explicit_error_and_no_success() {
    let result = fault_probe("eof").await;
    check_recovery(&result);
    assert!(matches!(result.begin, Ok(Ok(_))));
    assert_eq!(
        result.output.len(),
        4,
        "Worker EOF must produce an explicit final error."
    );
    assert_eq!(
        result
            .output
            .iter()
            .filter(|event| matches!(event, Event::Error(_)))
            .count(),
        1
    );
    assert!(matches!(result.output.last(), Some(Event::Error(error))
        if error.code == "worker_lost" && error.status == 503));
    assert!(!result
        .output
        .iter()
        .any(|event| matches!(event, Event::Done { .. })));
}
