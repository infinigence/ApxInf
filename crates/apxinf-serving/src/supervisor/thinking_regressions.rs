use super::lifecycle_regressions::cpu_service;
use super::*;
use futures_util::FutureExt;

async fn thinking_output(mode: &str) -> Vec<Event> {
    let script = r#"
import hashlib, json, struct, sys
mode = sys.argv[1]
def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None
def emit(command, kind, **fields):
    print(json.dumps(dict(protocol=command['protocol'], worker_epoch=command['worker_epoch'],
                         kind=kind, **fields)), flush=True)
while True:
    command = read()
    if command is None: break
    if command['kind'] == 'prepare_input':
        assert command['template_options'] == {'enable_thinking': True}
        digest = hashlib.sha256(b'apxinf-token-prefix-v1\0' + struct.pack('<QII', 2, 3, 4)).hexdigest()
        emit(command, 'prepared_input', command_id=command['command_id'],
             model_revision=command['model_revision'], token_ids=[3,4],
             effective_input_tokens=2, prompt_digest=digest)
    elif command['kind'] == 'submit':
        assert command['sampling']['mode'] == 'random'
        assert command['sampling']['temperature'] == 1.0
        key = dict(request_id=command['request_id'], attempt=command['attempt'])
        emit(command, 'accepted', event_seq=0, **key)
        hidden = '隐藏 STOP <function name="draft">'
        chunks = [hidden, '</thi', 'nk>visible ST', 'OP discarded'] if mode == 'stop' else [hidden]
        for index, chunk in enumerate(chunks):
            emit(command, 'tokens', event_seq=index+1, output_index=index,
                 token_ids=[10], text_delta=chunk, **key)
        status, cause, tail = 'completed', mode, ''
        if mode == 'terminal_close': cause, tail = 'eos', '</think>final answer'
        if mode in ('stop', 'cancel', 'expired'):
            control = read()
            if mode == 'stop':
                assert control['kind'] == 'stop_generation'
                assert control['text_byte_cutoff'] == len((hidden + '</think>visible ').encode())
                assert control['output_token_count'] == len(chunks)
                cause = 'stop_sequence'
            else:
                assert control['kind'] == 'cancel_request'
                status, cause = ('expired','deadline_exceeded') if mode == 'expired' else ('cancelled','user_cancel')
            emit(control, 'command_result', command_id=control['command_id'], status='accepted')
        emit(command, 'terminal', event_seq=len(chunks)+1, status=status, cause=cause,
             usage=dict(input_tokens=2, output_tokens=len(chunks)),
             state_result=dict(validity='none', consumed_position=2, history_count=2+len(chunks)),
             text_delta=tail, metrics=dict(elapsed_ns=20, ttft_ns=10, peak_memory_bytes=2048), **key)
        emit(command, 'resources_released', event_seq=len(chunks)+2,
             released_lease_ids=command['capacity_lease_ids'], retained_lease_ids=[], **key)
    else:
        raise RuntimeError(command['kind'])
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
        protocol: contracts::NATIVE_THINKING_PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: epoch.clone(),
        controls: Default::default(),
        sent_commands: 0,
    };
    let ready = json!({"protocol":contracts::NATIVE_THINKING_PROTOCOL,"worker_epoch":epoch,
        "model_revision":"1".repeat(64),"capability_revision":"2".repeat(64),
        "eos_token_ids":[2],"capabilities":{"max_context":64}});
    let mut service = cpu_service(
        ready,
        Duration::from_millis(if mode == "expired" { 300 } else { 2000 }),
        2,
    );
    let (completion, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    let mut ticket = service
        .enqueue(Request {
            selection: crate::sampling::Selection::Random {
                temperature: 1.0,
                top_p: 0.95,
            },
            messages: vec![json!({"role":"user","content":"Check the initial thinking boundary."})],
            tools: vec![],
            max_tokens: if mode == "length" { 1 } else { 8 },
            stops: vec!["STOP".into()],
            count_only: false,
        })
        .unwrap();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        || async { Err::<(Worker, Value), _>("The thinking probe must not rotate.".to_owned()) },
        completion,
    ));
    let result =
        std::panic::AssertUnwindSafe(tokio::time::timeout(Duration::from_secs(3), async {
            let begin = ticket.begin.await.unwrap().unwrap();
            assert_eq!(begin.input_tokens, 2);
            if mode == "cancel" {
                ticket.cancel.store(true, Ordering::Release);
            }
            let mut events = Vec::new();
            while let Some(event) = ticket.output.recv().await {
                events.push(event);
            }
            events
        }))
        .catch_unwind()
        .await;
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .unwrap()
        .unwrap();
    task.await.unwrap();
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    result.unwrap().unwrap()
}

#[tokio::test]
async fn initial_thinking_does_not_match_stops_and_preserves_raw_cutoff() {
    let events = thinking_output("stop").await;
    let text = events
        .iter()
        .filter_map(|event| match event {
            Event::Part(Part::Text(value)) => Some(value.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "visible ");
    assert!(
        matches!(events.last(), Some(Event::Done { cause, output_tokens: 4, matched_stop: Some(stop), .. })
        if cause == "stop_sequence" && stop == "STOP")
    );
}

#[tokio::test]
async fn hidden_output_budget_completion_remains_length_with_full_usage() {
    let events = thinking_output("length").await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(&events[0], Event::Done { cause, input_tokens: 2, output_tokens: 1, .. } if cause == "length")
    );
}

#[tokio::test]
async fn hidden_eos_is_invalid_but_terminal_decoder_close_is_accepted() {
    let events = thinking_output("eos").await;
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], Event::Error(error) if error.code == "internal_error"));
    let events = thinking_output("terminal_close").await;
    assert!(matches!(&events[0], Event::Part(Part::Text(text)) if text == "final answer"));
    assert!(
        matches!(events.last(), Some(Event::Done { cause, output_tokens: 1, .. }) if cause == "eos")
    );
}

#[tokio::test]
async fn hidden_reasoning_preserves_cancel_and_deadline_outcomes() {
    for (mode, code) in [("cancel", "cancelled"), ("expired", "deadline_exceeded")] {
        let events = thinking_output(mode).await;
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], Event::Error(error) if error.code == code),
            "unexpected outcome for {mode}"
        );
    }
}
