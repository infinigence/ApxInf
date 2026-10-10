//! Original CPU scenarios for pressure decisions at request boundaries.
use super::*;
use crate::host_pressure::Pressure;

fn service_with_pressure(ready: Value, timeout: Duration) -> Arc<Service> {
    let mut service = cpu_service(ready, timeout, 4);
    Arc::get_mut(&mut service).unwrap().host_pressure =
        Arc::new(HostPressure::new(HostPressurePolicy::Macos));
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    service
}

fn recover(service: &Service) {
    let now = Instant::now();
    service
        .host_pressure
        .record(Ok(Pressure::Normal), now - Duration::from_secs(1));
    service.host_pressure.record(Ok(Pressure::Normal), now);
    assert!(service.admission_snapshot().0);
}

#[tokio::test]
async fn pressure_rejection_before_enqueue_has_no_request_outcome() {
    let service = service_with_pressure(json!({}), Duration::from_secs(3));
    service
        .host_pressure
        .record(Ok(Pressure::Critical), Instant::now());
    for count_only in [false, true] {
        let error = service.enqueue(request(count_only)).err().unwrap();
        assert_eq!(
            (error.status, error.code.as_str()),
            (503, "capacity_unavailable")
        );
    }
    assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(
        observed_metric(
            &service,
            "apxinf_host_pressure_rejections_total",
            "stage=\"enqueue\",reason=\"critical\""
        ),
        2.0
    );
    assert!(service.available.load(Ordering::Acquire));
    service.shutdown();
    recover(&service);
    assert!(!service.available.load(Ordering::Acquire));
    assert_eq!(
        service.enqueue(request(false)).err().unwrap().code,
        "worker_lost"
    );
}

#[tokio::test]
async fn pressure_rechecks_queued_work_before_preparation() {
    let model = CpuModel::new(json!({}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = service_with_pressure(ready, Duration::from_secs(3));
    let generation = service.enqueue(request(false)).unwrap();
    let count = service.enqueue(request(true)).unwrap();
    service
        .host_pressure
        .record(Ok(Pressure::Warning), Instant::now());
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    for ticket in [generation, count] {
        assert_eq!(
            ticket.begin.await.unwrap().err().unwrap().code,
            "capacity_unavailable"
        );
    }
    assert!(!model
        .events()
        .iter()
        .any(|event| event["event"] == "prepare_started"));
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 2);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    for operation in ["generate", "count_tokens"] {
        assert_eq!(
            observed_metric(
                &service,
                "apxinf_request_outcomes_total",
                &format!("operation=\"{operation}\",status=\"failed\"")
            ),
            1.0
        );
        assert_eq!(
            observed_metric(
                &service,
                "apxinf_cleanup_pending",
                &format!("operation=\"{operation}\"")
            ),
            0.0
        );
    }
    recover(&service);
    run_empty_request(&service).await;
    stop_coordinator(&service, task).await;
}

#[tokio::test]
async fn pressure_during_preparation_prevents_submit_after_settlement() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[250]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = service_with_pressure(ready, Duration::from_secs(3));
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let ticket = service.enqueue(request(false)).unwrap();
    wait_for_event(&model, "prepare_started").await;
    service
        .host_pressure
        .record(Ok(Pressure::Critical), Instant::now());
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(
        ticket.begin.await.unwrap().err().unwrap().code,
        "capacity_unavailable"
    );
    stop_coordinator(&service, task).await;
    let events = model.events();
    assert!(events
        .iter()
        .any(|event| event["event"] == "prepare_finished"));
    assert!(!events
        .iter()
        .any(|event| event["event"] == "generate_started"));
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(
        observed_metric(&service, "apxinf_cleanup_pending", "operation=\"generate\""),
        0.0
    );
}

#[tokio::test]
async fn pressure_keeps_active_generation_owned_until_normal_settlement() {
    let model = CpuModel::new(json!({"generation_pause_ms":250}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let service = service_with_pressure(ready, Duration::from_secs(3));
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    let mut generation = request(false);
    generation.max_tokens = 1;
    let mut active = service.enqueue(generation).unwrap();
    assert!(active.begin.await.unwrap().is_ok());
    wait_for_event(&model, "generate_started").await;
    let waiting = service.enqueue(request(false)).unwrap();
    service
        .host_pressure
        .record(Ok(Pressure::Warning), Instant::now());
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    loop {
        match active.output.recv().await {
            Some(Event::Part(_)) => {}
            Some(Event::Done {
                output_tokens: 1, ..
            }) => break,
            _ => panic!("Pressure must not cancel active generation."),
        }
    }
    assert_eq!(
        waiting.begin.await.unwrap().err().unwrap().code,
        "capacity_unavailable"
    );
    stop_coordinator(&service, task).await;
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.cancelled.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn pressure_does_not_replace_existing_queue_cancellation_or_expiry() {
    for expired in [false, true] {
        let model = CpuModel::new(json!({}));
        let (worker, ready) = cpu_worker(model.model()).await.unwrap();
        let service = service_with_pressure(ready, Duration::from_millis(50));
        let ticket = service.enqueue(request(false)).unwrap();
        if expired {
            tokio::time::sleep(Duration::from_millis(60)).await;
        }
        ticket.cancel.store(true, Ordering::Release);
        service
            .host_pressure
            .record(Ok(Pressure::Warning), Instant::now());
        let path = model.model();
        let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
            cpu_worker(path.clone())
        }));
        assert_eq!(
            ticket.begin.await.unwrap().err().unwrap().code,
            if expired {
                "deadline_exceeded"
            } else {
                "cancelled"
            }
        );
        stop_coordinator(&service, task).await;
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
        assert_eq!(
            observed_metric(
                &service,
                "apxinf_host_pressure_rejections_total",
                "stage=\"dispatch\",reason=\"warning\""
            ),
            0.0
        );
        assert!(!model
            .events()
            .iter()
            .any(|event| event["event"] == "prepare_started"));
    }
}

#[tokio::test]
async fn pressure_monitor_survives_shutdown_until_preparation_settles() {
    let model = CpuModel::new(json!({"prepare_delays_ms":[3300]}));
    let (worker, ready) = cpu_worker(model.model()).await.unwrap();
    let (mut service, sender) = cpu_service_with_completion(ready, Duration::from_secs(8), 2);
    let pressure = Arc::new(HostPressure::with_sensor(|| Ok(Pressure::Normal)));
    pressure.refresh();
    Arc::get_mut(&mut service).unwrap().host_pressure = pressure.clone();
    HostPressure::monitor(&pressure);
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        sender,
    ));
    let mut ticket = service.enqueue(request(false)).unwrap();
    wait_for_event(&model, "prepare_started").await;
    service.shutdown();
    assert!(ticket.begin.await.unwrap().is_ok());
    assert!(matches!(
        ticket.output.recv().await,
        Some(Event::Done { .. })
    ));
    service.wait_stopped().await.unwrap();
    task.await.unwrap();
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
}
