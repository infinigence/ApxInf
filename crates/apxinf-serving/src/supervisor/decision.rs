use super::*;
use crate::{contracts::clef::DecisionTracker, decision::DecisionResult};
#[cfg(test)]
mod tests;

struct Public {
    begin: Option<oneshot::Sender<Result<Begin, ApiError>>>,
    terminal: Option<mpsc::OwnedPermit<Event>>,
    finished: bool,
}

impl Public {
    fn new(job: &mut Job) -> Self {
        let (unused, _) = oneshot::channel();
        Self {
            begin: Some(std::mem::replace(&mut job.begin, unused)),
            terminal: job.terminal_output.take(),
            finished: false,
        }
    }
    fn accept(&mut self, job: &Job, input_tokens: u64) {
        if let Some(begin) = self.begin.take() {
            if begin
                .send(Ok(Begin {
                    request_id: job.id.clone(),
                    input_tokens,
                }))
                .is_err()
            {
                job.cancel.store(true, Ordering::Release);
            }
        }
    }
    fn finish(&mut self, job: &Job, service: &Service, result: Result<DecisionResult, ApiError>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let status = match &result {
            Ok(_) => "completed",
            Err(error) if error.code == "deadline_exceeded" => "expired",
            Err(error) if error.code == "cancelled" => "cancelled",
            Err(_) => "failed",
        };
        job.observation.public_terminal(
            &service.metrics.observations,
            Outcome::from_status(status),
            Instant::now(),
        );
        record_public_status(&service.metrics, status);
        if let Some(begin) = self.begin.take() {
            let _ = begin.send(Err(result
                .err()
                .expect("Decision success requires acceptance.")));
            self.terminal.take();
        } else if let Some(terminal) = self.terminal.take() {
            terminal.send(match result {
                Ok(result) => Event::Decision(result),
                Err(error) => Event::Error(error),
            });
        }
    }
}

fn stopped(job: &Job, service: &Service, public: &Public, timeout: Duration) -> Option<ApiError> {
    if job.ingress.elapsed() >= timeout {
        Some(ApiError::new(
            504,
            "deadline_exceeded",
            "The decision request expired.",
        ))
    } else if job.cancel.load(Ordering::Acquire)
        || job.output.is_closed()
        || public
            .begin
            .as_ref()
            .is_some_and(oneshot::Sender::is_closed)
    {
        Some(ApiError::new(
            499,
            "cancelled",
            "The decision request was cancelled.",
        ))
    } else if service.queue.is_closed() {
        Some(ApiError::worker("The service is shutting down."))
    } else {
        None
    }
}

async fn wait<T>(
    operation: impl Future<Output = Result<T, ApiError>>,
    job: &Job,
    service: &Service,
    public: &Public,
    timeout: Duration,
) -> Result<T, ApiError> {
    tokio::pin!(operation);
    loop {
        if let Some(error) = stopped(job, service, public, timeout) {
            return Err(error);
        }
        tokio::select! {
            biased;
            result=&mut operation=>return result,
            _=tokio::time::sleep(QUEUE_SWEEP_INTERVAL)=>{},
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> ApiError {
    ApiError::new(500, "protocol_fault", error.to_string())
}

pub(super) async fn execute_decision(
    worker: &mut Worker,
    service: &Service,
    mut job: Job,
    timeout: Duration,
) -> Result<(), ExecutionFailure> {
    let Workload::Decision(request) = &job.workload else {
        unreachable!("Decision execution requires a decision workload.")
    };
    let mut prepare = worker.command("prepare_decision");
    prepare["request_id"] = json!(job.id);
    prepare["attempt"] = json!(1);
    prepare["model_revision"] = service.ready["model_revision"].clone();
    prepare["capability_revision"] = service.ready["capability_revision"].clone();
    prepare["record"] = json!(request.record);
    let mut tracker = DecisionTracker::new(&prepare).map_err(protocol)?;
    let mut public = Public::new(&mut job);
    let result = drive(
        worker,
        service,
        &job,
        &mut public,
        timeout,
        &prepare,
        &mut tracker,
    )
    .await;
    match result {
        Ok(result) => {
            public.finish(&job, service, Ok(result));
            Ok(())
        }
        Err(error) => {
            public.finish(&job, service, Err(error.clone()));
            if worker.retired {
                return Ok(());
            }
            if error.code == "cancelled"
                || error.code == "deadline_exceeded"
                || error.code == "capacity_unavailable"
                || service.queue.is_closed()
            {
                service.available.store(false, Ordering::Release);
                service.rotating.store(true, Ordering::Release);
                stop_child(&mut worker.child)
                    .await
                    .map_err(|message| ExecutionFailure {
                        error: ApiError::worker(message),
                        reported: true,
                    })?;
                check_stopped_frames(worker, service, &mut tracker)
                    .await
                    .map_err(|error| ExecutionFailure {
                        error,
                        reported: true,
                    })?;
                worker.retired = true;
                Ok(())
            } else {
                Err(ExecutionFailure {
                    error,
                    reported: true,
                })
            }
        }
    }
}

async fn check_stopped_frames(
    worker: &mut Worker,
    service: &Service,
    tracker: &mut DecisionTracker,
) -> Result<(), ApiError> {
    tokio::time::timeout(SETTLEMENT_GRACE, async {
        while let Some(frame) = worker.frames.recv().await {
            let frame = frame.map_err(protocol)?;
            tracker.observe(&frame).map_err(protocol)?;
            if frame["kind"] == "prepared_decision" {
                crate::clef_identity::verify_prepared(&frame["preparation"], &service.ready)
                    .map_err(protocol)?;
            }
        }
        Ok(())
    })
    .await
    .map_err(|_| ApiError::worker("The reaped decision output did not settle."))?
}

async fn drive(
    worker: &mut Worker,
    service: &Service,
    job: &Job,
    public: &mut Public,
    timeout: Duration,
    prepare: &Value,
    tracker: &mut DecisionTracker,
) -> Result<DecisionResult, ApiError> {
    job.observation.preparation_start(Instant::now());
    wait(worker.send(prepare), job, service, public, timeout).await?;
    let prepared = wait(
        worker.receive(timeout + SETTLEMENT_GRACE),
        job,
        service,
        public,
        timeout,
    )
    .await?;
    tracker.observe(&prepared).map_err(protocol)?;
    job.observation
        .preparation_end(&service.metrics.observations, Instant::now());
    let mut failure = None;
    if prepared["kind"] == "failure" {
        failure = Some(failure_error(&prepared));
    } else {
        crate::clef_identity::verify_prepared(&prepared["preparation"], &service.ready)
            .map_err(protocol)?;
        if let Some(error) = stopped(job, service, public, timeout) {
            return Err(error);
        }
        service.check_host_pressure(true)?;
        let mut submit = worker.command("submit_decision");
        submit["request_id"] = json!(job.id);
        submit["attempt"] = json!(1);
        submit["model_revision"] = service.ready["model_revision"].clone();
        submit["capability_revision"] = service.ready["capability_revision"].clone();
        submit["preparation"] = prepared["preparation"].clone();
        submit["capacity_lease_ids"] = json!([uuid::Uuid::new_v4().to_string()]);
        tracker.observe(&submit).map_err(protocol)?;
        wait(worker.send(&submit), job, service, public, timeout).await?;
        loop {
            let frame = wait(
                worker.receive(timeout + SETTLEMENT_GRACE),
                job,
                service,
                public,
                timeout,
            )
            .await?;
            tracker.observe(&frame).map_err(protocol)?;
            match frame["kind"].as_str() {
                Some("accepted") => public.accept(
                    job,
                    prepared["preparation"]["effective_input_tokens"]
                        .as_u64()
                        .unwrap(),
                ),
                Some("decision_result") => {
                    job.observation.worker_terminal(
                        &service.metrics.observations,
                        Instant::now(),
                        None,
                    );
                    service.metrics.peak_bytes.fetch_max(
                        frame["metrics"]["peak_device_bytes"].as_u64().unwrap(),
                        Ordering::Relaxed,
                    );
                    break;
                }
                Some("failure") => {
                    failure = Some(failure_error(&frame));
                    break;
                }
                _ => return Err(protocol("Unexpected decision attempt event.")),
            }
        }
    }
    let drained = wait(
        worker.receive(SETTLEMENT_GRACE),
        job,
        service,
        public,
        timeout,
    )
    .await?;
    tracker.observe(&drained).map_err(protocol)?;
    let eof = wait(
        async {
            match tokio::time::timeout(SETTLEMENT_GRACE, worker.frames.recv()).await {
                Ok(None) => Ok(()),
                Ok(Some(_)) => Err(protocol(
                    "The decision worker emitted output after draining.",
                )),
                Err(_) => Err(ApiError::worker(
                    "The decision output did not close after draining.",
                )),
            }
        },
        job,
        service,
        public,
        timeout,
    )
    .await;
    eof?;
    let status = wait(
        async {
            tokio::time::timeout(SETTLEMENT_GRACE, worker.child.wait())
                .await
                .map_err(|_| ApiError::worker("The decision worker did not exit after draining."))?
                .map_err(|_| ApiError::worker("The decision worker could not be reaped."))
        },
        job,
        service,
        public,
        timeout,
    )
    .await?;
    tracker
        .reaped(status.code().unwrap_or(-1), true)
        .map_err(protocol)?;
    worker.retired = true;
    service.available.store(false, Ordering::Release);
    service.rotating.store(true, Ordering::Release);
    drop(worker.input.take());
    if let Some(error) = stopped(job, service, public, timeout) {
        return Err(error);
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let result = tracker
        .result()
        .cloned()
        .ok_or_else(|| protocol("The decision attempt has no settled result."))?;
    service
        .metrics
        .input_tokens
        .fetch_add(result.input_tokens, Ordering::Relaxed);
    Ok(result)
}

fn failure_error(frame: &Value) -> ApiError {
    let code = frame["code"].as_str().unwrap();
    let status = match code {
        "invalid_request" | "context_limit" => 400,
        "model_unavailable" | "worker_lost" => 503,
        _ => 500,
    };
    ApiError::new(status, code, frame["message"].as_str().unwrap())
}
