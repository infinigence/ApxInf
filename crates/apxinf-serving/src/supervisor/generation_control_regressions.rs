//! CPU checks for public expiry while generation control is pending.
use super::generation::{wait_for_generation_control, GenerationPublic};
use super::*;
use std::pin::Pin;
use std::task::{Context, Poll};

struct ControlledWrite {
    release: oneshot::Receiver<()>,
    polls: Arc<AtomicU64>,
    dropped: Arc<AtomicBool>,
}

impl Future for ControlledWrite {
    type Output = Result<(), ApiError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        match Pin::new(&mut self.release).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(_)) => Poll::Ready(Err(ApiError::worker("The test write was closed."))),
        }
    }
}

impl Drop for ControlledWrite {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

fn controlled_write() -> (
    ControlledWrite,
    oneshot::Sender<()>,
    Arc<AtomicU64>,
    Arc<AtomicBool>,
) {
    let (sender, release) = oneshot::channel();
    let polls = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    (
        ControlledWrite {
            release,
            polls: polls.clone(),
            dropped: dropped.clone(),
        },
        sender,
        polls,
        dropped,
    )
}

fn request_fixture(timeout: Duration) -> (Service, Job, Ticket) {
    let (_, stopped) = watch::channel(None);
    let ready = json!({});
    let service = Service {
        queue: PendingQueue::new(1),
        host_pressure: Arc::new(HostPressure::new(HostPressurePolicy::Disabled)),
        ready_state: RwLock::new(ready.clone()),
        ready,
        stopped,
        model_id: "generation-control-test".into(),
        available: AtomicBool::new(true),
        rotating: AtomicBool::new(false),
        metrics: Metrics::default(),
        timeout,
        memory_budget: 4096,
    };
    service.metrics.active.store(1, Ordering::Relaxed);
    service
        .metrics
        .reserved_bytes
        .store(1024, Ordering::Relaxed);
    let ingress = Instant::now();
    let (begin, begin_rx) = oneshot::channel();
    let (output, output_rx) = mpsc::channel(OUTPUT_PART_CAPACITY + 1);
    let terminal_output = Some(output.clone().try_reserve_owned().unwrap());
    let cancel = Arc::new(AtomicBool::new(false));
    let job = Job {
        observation: Arc::new(RequestObservation::new(ingress, false)),
        request: Request {
            messages: vec![json!({"role":"user", "content":"control lifetime probe"})],
            tools: vec![],
            max_tokens: 4,
            stops: vec![],
            count_only: false,
        },
        id: "control-probe".into(),
        ingress,
        begin,
        output,
        terminal_output,
        cancel: cancel.clone(),
    };
    (
        service,
        job,
        Ticket {
            begin: begin_rx,
            output: output_rx,
            cancel,
        },
    )
}

fn public_for(job: &mut Job) -> GenerationPublic {
    let begin = Begin {
        request_id: job.id.clone(),
        input_tokens: 3,
    };
    GenerationPublic::new(job, begin)
}

fn metric(service: &Service, name: &str, status: Option<&str>) -> f64 {
    let suffix = status.map_or(String::new(), |value| format!(",status=\"{value}\""));
    let prefix = format!("{name}{{operation=\"generate\"{suffix}}} ");
    service
        .metrics_text()
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("Missing metric {prefix}"))
        .parse()
        .unwrap()
}

fn assert_one_outcome(service: &Service, expected: &str) {
    for status in ["completed", "cancelled", "failed", "expired"] {
        let count = if status == expected { 1.0 } else { 0.0 };
        assert_eq!(
            metric(service, "apxinf_request_outcomes_total", Some(status)),
            count
        );
        assert_eq!(
            metric(
                service,
                "apxinf_service_request_seconds_count",
                Some(status)
            ),
            count
        );
    }
    assert_eq!(
        service.metrics.completed.load(Ordering::Relaxed),
        u64::from(expected == "completed")
    );
    assert_eq!(
        service.metrics.cancelled.load(Ordering::Relaxed),
        u64::from(expected == "cancelled")
    );
    assert_eq!(
        service.metrics.failed.load(Ordering::Relaxed),
        u64::from(expected == "failed")
    );
    assert_eq!(
        service.metrics.expired.load(Ordering::Relaxed),
        u64::from(expected == "expired")
    );
}

fn take_error(receiver: &mut mpsc::Receiver<Event>) -> ApiError {
    match receiver.try_recv() {
        Ok(Event::Error(error)) => error,
        _ => panic!("The reserved result must contain one public error."),
    }
}

fn done() -> Event {
    Event::Done {
        cause: "length".into(),
        input_tokens: 3,
        output_tokens: 4,
        matched_stop: None,
    }
}

#[tokio::test]
async fn pending_control_preserves_reserved_expiry_and_continues_until_released() {
    let timeout = Duration::from_millis(30);
    let (service, mut job, mut ticket) = request_fixture(timeout);
    let mut public = public_for(&mut job);
    public.accept(&job);
    assert!(ticket.begin.await.unwrap().is_ok());
    for _ in 0..OUTPUT_PART_CAPACITY {
        assert!(job
            .output
            .try_send(Event::Part(Part::Text("buffered".into())))
            .is_ok());
    }
    assert_eq!(job.output.capacity(), 0);
    let (operation, release, polls, dropped) = controlled_write();
    {
        let wait = wait_for_generation_control(
            operation,
            &job,
            &service,
            &mut public,
            timeout,
            Instant::now(),
            None,
            None,
        );
        tokio::pin!(wait);
        assert!(tokio::time::timeout(Duration::from_millis(90), &mut wait)
            .await
            .is_err());
        assert!(!dropped.load(Ordering::Acquire));
        assert!(!release.is_closed());
        assert!(polls.load(Ordering::Relaxed) > 0);
        assert_one_outcome(&service, "expired");
        assert_eq!(metric(&service, "apxinf_cleanup_pending", None), 1.0);
        assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
        for _ in 0..OUTPUT_PART_CAPACITY {
            assert!(matches!(ticket.output.try_recv(), Ok(Event::Part(_))));
        }
        assert_eq!(take_error(&mut ticket.output).code, "deadline_exceeded");
        assert!(matches!(
            ticket.output.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(tokio::time::timeout(Duration::from_millis(30), &mut wait)
            .await
            .is_err());
        assert!(!dropped.load(Ordering::Acquire));
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), &mut wait)
            .await
            .unwrap()
            .unwrap();
    }
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(public.status, Some("expired"));
    assert_one_outcome(&service, "expired");
    assert_eq!(
        metric(&service, "apxinf_cleanup_pending", None),
        1.0,
        "A finished control write does not establish resource settlement."
    );
    job.observation
        .settled(&service.metrics.observations, Instant::now());
    assert_eq!(metric(&service, "apxinf_cleanup_pending", None), 0.0);
    assert_eq!(
        metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            None
        ),
        1.0
    );
}

#[tokio::test]
async fn control_write_uses_existing_grace_and_expiry_does_not_restart_it() {
    let timeout = Duration::from_millis(25);
    let (service, mut job, mut ticket) = request_fixture(timeout);
    let mut public = public_for(&mut job);
    public.accept(&job);
    assert!(ticket.begin.await.unwrap().is_ok());
    let (operation, release, _, dropped) = controlled_write();
    let stopped_at = Instant::now() - (SETTLEMENT_GRACE - Duration::from_millis(160));
    let failure = tokio::time::timeout(
        Duration::from_secs(1),
        wait_for_generation_control(
            operation,
            &job,
            &service,
            &mut public,
            timeout,
            stopped_at,
            None,
            None,
        ),
    )
    .await
    .expect("The existing grace must include time spent writing control.")
    .unwrap_err();
    assert_eq!(failure.code, "worker_lost");
    assert!(stopped_at.elapsed() >= SETTLEMENT_GRACE);
    assert!(release.is_closed());
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(take_error(&mut ticket.output).code, "deadline_exceeded");
    let ended_at = public.ended_at;
    public.finish(&job, &service, "failed", Event::Error(failure));
    assert_eq!(public.ended_at, ended_at);
    assert_eq!(public.status, Some("expired"));
    assert_one_outcome(&service, "expired");
    assert_eq!(metric(&service, "apxinf_cleanup_pending", None), 1.0);
    assert_eq!(
        metric(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            None
        ),
        0.0
    );
}

#[tokio::test]
async fn already_exhausted_grace_does_not_start_another_control_poll() {
    let timeout = Duration::from_secs(60);
    let (service, mut job, mut ticket) = request_fixture(timeout);
    let mut public = public_for(&mut job);
    let (operation, release, polls, dropped) = controlled_write();
    let stopped_at = Instant::now() - SETTLEMENT_GRACE;
    let failure = wait_for_generation_control(
        operation,
        &job,
        &service,
        &mut public,
        timeout,
        stopped_at,
        None,
        None,
    )
    .await
    .unwrap_err();
    assert_eq!(failure.code, "worker_lost");
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert!(release.is_closed());
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(public.status, None);
    assert!(matches!(
        ticket.begin.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn public_expiry_cannot_be_replaced_by_late_failure_or_success() {
    let (service, mut job, mut ticket) = request_fixture(Duration::ZERO);
    let mut public = public_for(&mut job);
    public.accept(&job);
    assert!(ticket.begin.await.unwrap().is_ok());
    public.expire(&job, &service, Duration::ZERO, None, None);
    let ended_at = public.ended_at;
    public.finish(
        &job,
        &service,
        "failed",
        Event::Error(ApiError::worker("Late cleanup failure.")),
    );
    public.finish(&job, &service, "completed", done());
    public.expire(
        &job,
        &service,
        Duration::ZERO,
        Some(&ApiError::invalid("Late parser error.")),
        None,
    );
    assert_eq!(public.status, Some("expired"));
    assert_eq!(public.ended_at, ended_at);
    assert_eq!(take_error(&mut ticket.output).code, "deadline_exceeded");
    assert!(matches!(
        ticket.output.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_one_outcome(&service, "expired");
}

#[tokio::test]
async fn expiry_before_acceptance_replies_once_through_begin() {
    let (service, mut job, mut ticket) = request_fixture(Duration::ZERO);
    let mut public = public_for(&mut job);
    public.expire(&job, &service, Duration::ZERO, None, None);
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "deadline_exceeded"
    );
    let ended_at = public.ended_at;
    public.accept(&job);
    public.finish(&job, &service, "completed", done());
    public.finish(
        &job,
        &service,
        "failed",
        Event::Error(ApiError::worker("Late worker failure.")),
    );
    assert_eq!(public.ended_at, ended_at);
    assert_eq!(public.status, Some("expired"));
    assert!(matches!(
        ticket.output.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(job.output.capacity(), OUTPUT_PART_CAPACITY + 1);
    assert_one_outcome(&service, "expired");
}

#[tokio::test]
async fn known_parser_or_worker_failure_remains_failed_at_public_deadline() {
    let worker_failure = json!({"status":"failed", "error":{
        "code":"internal_error", "message":"The worker reported a model error."}});
    for (parser_error, terminal, expected_message) in [
        (
            Some(ApiError::new(
                500,
                "internal_error",
                "Parser rejected output.",
            )),
            None,
            "Parser rejected output.",
        ),
        (
            None,
            Some(&worker_failure),
            "The worker reported a model error.",
        ),
        (
            Some(ApiError::new(500, "internal_error", "Parser failed first.")),
            Some(&worker_failure),
            "Parser failed first.",
        ),
    ] {
        let (service, mut job, mut ticket) = request_fixture(Duration::ZERO);
        let mut public = public_for(&mut job);
        public.accept(&job);
        assert!(ticket.begin.await.unwrap().is_ok());
        public.expire(
            &job,
            &service,
            Duration::ZERO,
            parser_error.as_ref(),
            terminal,
        );
        let error = take_error(&mut ticket.output);
        assert_eq!(error.code, "internal_error");
        assert_eq!(error.message, expected_message);
        assert_eq!(public.status, Some("failed"));
        public.finish(&job, &service, "completed", done());
        assert_one_outcome(&service, "failed");
        assert_eq!(metric(&service, "apxinf_cleanup_pending", None), 1.0);
    }
}
