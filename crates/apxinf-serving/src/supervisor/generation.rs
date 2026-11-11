//! Public generation deadlines remain independent of worker settlement.
use super::*;

pub(super) struct GenerationPublic {
    begin: Option<oneshot::Sender<Result<Begin, ApiError>>>,
    begin_value: Option<Begin>,
    terminal_output: Option<mpsc::OwnedPermit<Event>>,
    pub(super) accepted: bool,
    pub(super) status: Option<&'static str>,
    pub(super) ended_at: Option<Instant>,
}

impl GenerationPublic {
    pub(super) fn new(job: &mut Job, begin: Begin) -> Self {
        let (unused, _) = oneshot::channel();
        Self {
            begin: Some(std::mem::replace(&mut job.begin, unused)),
            begin_value: Some(begin),
            terminal_output: job.terminal_output.take(),
            accepted: false,
            status: None,
            ended_at: None,
        }
    }

    pub(super) fn accept(&mut self, job: &Job) {
        self.accepted = true;
        if let Some(sender) = self.begin.take() {
            if sender.send(Ok(self.begin_value.take().unwrap())).is_err() {
                job.cancel.store(true, Ordering::Release);
            }
        }
    }

    pub(super) fn finish(
        &mut self,
        job: &Job,
        service: &Service,
        status: &'static str,
        result: Event,
    ) {
        if self.status.is_some() {
            return;
        }
        let now = Instant::now();
        self.status = Some(status);
        self.ended_at = Some(now);
        job.observation.public_terminal(
            &service.metrics.observations,
            Outcome::from_status(status),
            now,
        );
        if let Some(sender) = self.begin.take() {
            let error = match result {
                Event::Error(error) => error,
                _ => unreachable!("Public success requires worker acceptance."),
            };
            self.begin_value = None;
            self.terminal_output.take();
            record_public_status(&service.metrics, status);
            let _ = sender.send(Err(error));
        } else {
            publish_result(
                &service.metrics,
                self.terminal_output.take().unwrap(),
                status,
                result,
            );
        }
    }

    pub(super) fn expire(
        &mut self,
        job: &Job,
        service: &Service,
        timeout: Duration,
        known_error: Option<&ApiError>,
        terminal: Option<&Value>,
    ) {
        if self.status.is_some() || job.ingress.elapsed() < timeout {
            return;
        }
        let failure = known_error.cloned().or_else(|| {
            terminal
                .filter(|frame| frame["status"] == "failed")
                .map(error_from_frame)
        });
        let (status, error) = match failure {
            Some(error) => ("failed", error),
            None => (
                "expired",
                ApiError::new(504, "deadline_exceeded", "The request expired."),
            ),
        };
        self.finish(job, service, status, Event::Error(error));
    }

    pub(super) fn next_wait(
        &self,
        job: &Job,
        timeout: Duration,
        stopped_at: Option<Instant>,
    ) -> Duration {
        let public_wait = if self.status.is_none() {
            timeout.saturating_sub(job.ingress.elapsed())
        } else {
            QUEUE_SWEEP_INTERVAL
        };
        let settlement_wait = stopped_at.map_or(QUEUE_SWEEP_INTERVAL, |at| {
            SETTLEMENT_GRACE.saturating_sub(at.elapsed())
        });
        public_wait.min(settlement_wait).min(QUEUE_SWEEP_INTERVAL)
    }
}

pub(super) async fn wait_for_generation_control<F>(
    operation: F,
    job: &Job,
    service: &Service,
    public: &mut GenerationPublic,
    timeout: Duration,
    stopped_at: Instant,
    known_error: Option<&ApiError>,
    terminal: Option<&Value>,
) -> Result<(), ApiError>
where
    F: Future<Output = Result<(), ApiError>>,
{
    tokio::pin!(operation);
    loop {
        public.expire(job, service, timeout, known_error, terminal);
        if stopped_at.elapsed() >= SETTLEMENT_GRACE {
            return Err(ApiError::worker(
                "Generation control did not settle within 20 seconds.",
            ));
        }
        tokio::select! {
            result = &mut operation => {
                public.expire(job, service, timeout, known_error, terminal);
                return result;
            },
            _ = tokio::time::sleep(public.next_wait(job, timeout, Some(stopped_at))) => {},
        }
    }
}
