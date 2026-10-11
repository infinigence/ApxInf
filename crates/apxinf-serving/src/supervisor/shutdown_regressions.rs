//! CPU checks for loading cancellation and explicit process settlement.
use super::*;
#[path = "../../tests/support/signal_fixture.rs"]
mod signal_fixture;
use signal_fixture::Fixture;

async fn wait_for_worker(fixture: &Fixture) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while fixture.worker_pid().is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("The loading CPU worker must start.");
}

#[tokio::test]
async fn startup_cancellation_returns_only_after_child_reaping() {
    let fixture = Fixture::new("loading");
    let (stop, stopped) = oneshot::channel();
    let config = fixture.config.clone();
    let task = tokio::spawn(start_with_shutdown(config, async {
        let _ = stopped.await;
    }));
    wait_for_worker(&fixture).await;
    stop.send(()).unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .is_none());
    fixture.assert_reaped();
}

#[tokio::test]
async fn startup_cancelled_before_spawn_creates_no_child() {
    let fixture = Fixture::new("loading");
    assert!(
        start_with_shutdown(fixture.config.clone(), std::future::ready(()))
            .await
            .unwrap()
            .is_none()
    );
    assert!(fixture.worker_pid().is_none());
}

#[tokio::test]
async fn startup_protocol_error_still_reports_failure_and_reaps() {
    let fixture = Fixture::new("invalid_ready");
    let result = start_with_shutdown(fixture.config.clone(), std::future::pending()).await;
    assert!(result.is_err());
    fixture.assert_reaped();
}

#[tokio::test]
async fn shutdown_cancels_replacement_loading_and_reaps_before_completion() {
    let fixture = Fixture::new("loading");
    let old_model = lifecycle_regressions::CpuModel::new(json!({}));
    let (mut worker, ready) = lifecycle_regressions::cpu_worker(old_model.model())
        .await
        .unwrap();
    worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    let mut service = lifecycle_regressions::cpu_service(ready, Duration::from_secs(5), 2);
    let (stopped_sender, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    let weak = Arc::downgrade(&service);
    let config = fixture.config.clone();
    let guard = service.host_pressure.clone();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || {
            let weak = weak.clone();
            let config = config.clone();
            let guard = guard.clone();
            async move {
                launch_replacement(weak.clone(), || {
                    let weak = weak.clone();
                    launch_worker_until(&config, &guard, async move {
                        if let Some(service) = weak.upgrade() {
                            service.queue.wait_closed().await;
                        }
                    })
                })
                .await
            }
        },
        stopped_sender,
    ));
    wait_for_worker(&fixture).await;
    assert!(service.rotating.load(Ordering::Acquire));
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .unwrap()
        .unwrap();
    task.await.unwrap();
    fixture.assert_reaped();
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 0);
    assert!(!service.available.load(Ordering::Acquire));
}
