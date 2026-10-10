//! Pressure pauses planned replacement without losing queue deadlines or child ownership.
use super::lifecycle_regressions::{cpu_service, cpu_worker, request, CpuModel};
use super::*;
use crate::host_pressure::Pressure;

async fn wait_until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

fn pressure_service(ready: Value, timeout: Duration) -> Arc<Service> {
    let mut service = cpu_service(ready, timeout, 2);
    Arc::get_mut(&mut service).unwrap().host_pressure =
        Arc::new(HostPressure::with_sensor(|| Ok(Pressure::Normal)));
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    service
}

#[tokio::test]
async fn pressure_preload_refusal_precedes_spawn() {
    let config = super::memory_regressions::configuration(4096, 1024);
    let guard = HostPressure::with_sensor(|| Ok(Pressure::Warning));
    assert!(matches!(
        launch_worker(&config, &guard).await,
        Err(LaunchError::Pressure(_))
    ));
}

#[tokio::test]
async fn pressure_postload_refusal_reaps_the_created_worker() {
    let model = CpuModel::new(json!({}));
    let (mut worker, _) = cpu_worker(model.model()).await.unwrap();
    assert!(worker.child.id().is_some());
    let guard = HostPressure::with_sensor(|| Ok(Pressure::Warning));
    assert!(matches!(
        check_loaded_pressure(&mut worker, &guard).await,
        Err(LaunchError::Pressure(_))
    ));
    assert!(
        worker.child.id().is_none(),
        "The child must already be reaped."
    );
    assert!(worker.child.try_wait().unwrap().is_some());
}

#[tokio::test]
async fn pressure_rotation_wait_preserves_queue_expiry_then_serves_a_new_request() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    let old_epoch = worker.epoch.clone();
    let service = pressure_service(ready, Duration::from_millis(150));
    let queued = service.enqueue(request(true)).unwrap();
    service
        .host_pressure
        .record(Ok(Pressure::Warning), Instant::now());
    start_queue_sweeper(&service);
    let launches = Arc::new(AtomicU64::new(0));
    let count = launches.clone();
    let weak = Arc::downgrade(&service);
    let path = model.model();
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        let weak = weak.clone();
        let count = count.clone();
        let path = path.clone();
        async move {
            launch_replacement(weak, || {
                let path = path.clone();
                count.fetch_add(1, Ordering::Relaxed);
                async move { cpu_worker(path).await.map_err(LaunchError::Worker) }
            })
            .await
        }
    }));
    wait_until(|| {
        model
            .events()
            .iter()
            .any(|event| event["kind"] == "drained")
    })
    .await;
    assert!(service.rotating.load(Ordering::Acquire));
    assert_eq!(launches.load(Ordering::Relaxed), 0);
    assert_eq!(
        service.enqueue(request(true)).err().unwrap().code,
        "model_unavailable"
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_millis(350), queued.begin)
            .await
            .unwrap()
            .unwrap()
            .err()
            .unwrap()
            .code,
        "deadline_exceeded"
    );
    assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    tokio::time::sleep(crate::host_pressure::SAMPLE_PERIOD).await;
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    wait_until(|| service.metrics.worker_rotations.load(Ordering::Relaxed) == 1).await;
    assert_ne!(service.ready_snapshot()["worker_epoch"], old_epoch);
    assert!(service
        .enqueue(request(true))
        .unwrap()
        .begin
        .await
        .unwrap()
        .is_ok());
    assert_eq!(launches.load(Ordering::Relaxed), 1);
    assert_eq!(
        model
            .events()
            .iter()
            .filter(|event| event["event"] == "prepare_started")
            .count(),
        1
    );
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn pressure_rotation_wait_stops_without_loading_or_worker_fault() {
    let model = CpuModel::new(json!({}));
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    let service = pressure_service(ready, Duration::from_secs(3));
    service
        .host_pressure
        .record(Ok(Pressure::Warning), Instant::now());
    let weak = Arc::downgrade(&service);
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        let weak = weak.clone();
        async move {
            launch_replacement(weak, || async {
                panic!("Pressure must prevent replacement loading.");
            })
            .await
        }
    }));
    wait_until(|| {
        model
            .events()
            .iter()
            .any(|event| event["kind"] == "drained")
    })
    .await;
    service.shutdown();
    tokio::time::timeout(Duration::from_millis(350), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 0);
    assert!(!service.available.load(Ordering::Acquire));
}

#[tokio::test]
async fn pressure_retry_does_not_hide_worker_failure_when_shutdown_races() {
    let service = pressure_service(json!({}), Duration::from_secs(3));
    let closing = service.clone();
    let result = launch_replacement(Arc::downgrade(&service), || {
        let closing = closing.clone();
        async move {
            closing.shutdown();
            Err(LaunchError::Worker(
                "Injected process reaping failure.".into(),
            ))
        }
    })
    .await;
    assert!(matches!(result, Err(LaunchError::Worker(message)) if message.contains("reaping")));
}

#[tokio::test]
async fn pressure_retry_reaps_a_refused_child_before_the_next_load() {
    let model = CpuModel::new(json!({}));
    let service = pressure_service(json!({}), Duration::from_secs(3));
    let guard = service.host_pressure.clone();
    let launches = Arc::new(AtomicU64::new(0));
    let count = launches.clone();
    let path = model.model();
    let weak = Arc::downgrade(&service);
    let reaped = Arc::new(AtomicBool::new(false));
    let observed_reap = reaped.clone();
    let task = tokio::spawn(launch_replacement(weak, move || {
        let path = path.clone();
        let guard = guard.clone();
        let observed_reap = observed_reap.clone();
        let first = count.fetch_add(1, Ordering::Relaxed) == 0;
        async move {
            if !first {
                assert!(observed_reap.load(Ordering::Acquire));
            }
            let (mut worker, ready) = cpu_worker(path).await.map_err(LaunchError::Worker)?;
            if first {
                guard.record(Ok(Pressure::Warning), Instant::now());
                let warning = HostPressure::with_sensor(|| Ok(Pressure::Warning));
                let result = check_loaded_pressure(&mut worker, &warning).await;
                assert!(worker.child.id().is_none());
                observed_reap.store(true, Ordering::Release);
                result?;
            }
            Ok((worker, ready))
        }
    }));
    wait_until(|| reaped.load(Ordering::Acquire)).await;
    assert_eq!(launches.load(Ordering::Relaxed), 1);
    assert!(!task.is_finished());
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    tokio::time::sleep(crate::host_pressure::SAMPLE_PERIOD).await;
    service
        .host_pressure
        .record(Ok(Pressure::Normal), Instant::now());
    let (mut worker, _) = tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(launches.load(Ordering::Relaxed), 2);
    stop_child(&mut worker.child).await.unwrap();
    service.shutdown();
}
