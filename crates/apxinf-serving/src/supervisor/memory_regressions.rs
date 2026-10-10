//! Memory policy checks that do not load a model or access a device.
use super::*;

pub(super) fn configuration(memory_budget: u64, sequence_reservation: u64) -> Config {
    Config {
        python: PathBuf::from("/missing/apxinf-memory-test-python"),
        worker: PathBuf::from("/missing/apxinf-memory-test-worker"),
        model: PathBuf::from("/missing/apxinf-memory-test-model"),
        model_id: "memory-test".into(),
        max_context: 64,
        max_output: 16,
        prefill_step_size: 8,
        output_batch_tokens: 1,
        queue_capacity: 1,
        timeout: Duration::from_secs(1),
        memory_budget,
        sequence_reservation,
        host_pressure_policy: crate::host_pressure::HostPressurePolicy::Disabled,
    }
}

#[tokio::test]
async fn memory_limits_fail_before_the_worker_can_start() {
    for (budget, reservation) in [
        (0, 1),
        (1024, 0),
        (contracts::MAX_SAFE_INTEGER + 1, 1),
        (1024, u64::MAX),
        (1024, 1025),
    ] {
        let result = start(configuration(budget, reservation)).await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("Invalid memory limits must fail before startup."),
        };
        assert!(
            !error.contains("Cannot start worker"),
            "Memory validation ran too late: {error}"
        );
    }
}

#[test]
fn memory_limits_accept_only_the_documented_integer_boundaries() {
    for (budget, reservation) in [
        (1, 1),
        (contracts::MAX_SAFE_INTEGER, 1),
        (contracts::MAX_SAFE_INTEGER, contracts::MAX_SAFE_INTEGER),
    ] {
        assert!(configuration(budget, reservation).validate_memory().is_ok());
    }
    for (budget, reservation) in [(0, 0), (1, 0), (1, 2), (u64::MAX, 1), (u64::MAX, u64::MAX)] {
        assert!(configuration(budget, reservation)
            .validate_memory()
            .is_err());
    }
}

#[test]
fn memory_capacity_rejects_overflow_and_accepts_an_exact_fit() {
    let config = configuration(1024, 256);
    assert!(config.check_resident_capacity(768).is_ok());
    assert!(config.check_resident_capacity(769).is_err());
    assert!(configuration(u64::MAX, 1)
        .check_resident_capacity(u64::MAX)
        .unwrap_err()
        .contains("overflow"));
}

#[test]
fn memory_peak_uses_the_reported_peak_and_the_allocator_total() {
    assert_eq!(
        ready_memory_totals(&json!({"memory": {
            "active_bytes":100, "cache_bytes":20, "peak_bytes":500
        }}))
        .unwrap(),
        (120, 500)
    );
    assert_eq!(
        ready_memory_totals(&json!({"memory": {
            "active_bytes":100, "cache_bytes":400, "peak_bytes":200
        }}))
        .unwrap(),
        (500, 500)
    );
    assert!(ready_memory_totals(&json!({"memory": {
        "active_bytes":u64::MAX, "cache_bytes":1, "peak_bytes":0
    }}))
    .is_err());
    assert!(ready_memory_totals(&json!({"memory": {
        "active_bytes":100, "peak_bytes":500
    }}))
    .is_err());
}
