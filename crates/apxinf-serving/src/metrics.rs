//! Monotonic request observations with bounded Prometheus label sets.
use std::{
    fmt::Write,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

const BOUNDS_MS: [u64; 19] = [
    1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000, 60000, 120000, 300000,
    600000, 3600000,
];
const OPERATIONS: [&str; 2] = ["generate", "count_tokens"];
const STATUSES: [&str; 4] = ["completed", "cancelled", "failed", "expired"];

#[derive(Clone, Copy)]
pub(crate) enum Outcome {
    Completed,
    Cancelled,
    Failed,
    Expired,
}

impl Outcome {
    pub(crate) fn from_status(status: &str) -> Self {
        match status {
            "completed" => Self::Completed,
            "cancelled" => Self::Cancelled,
            "expired" => Self::Expired,
            _ => Self::Failed,
        }
    }
}

#[derive(Clone, Copy)]
enum Interval {
    Queue,
    Preparation,
    FirstWorker,
    WorkerFirstToken,
    FirstPublic,
    WorkerOutputGap,
    WorkerCleanup,
    PublicCleanup,
}

const HISTOGRAMS: [(&str, &str); 8] = [
    (
        "apxinf_queue_wait_seconds",
        "Time from enqueue to removal from the waiting queue.",
    ),
    (
        "apxinf_preparation_seconds",
        "Time from preparation transmission to its response or process reaping.",
    ),
    (
        "apxinf_time_to_first_worker_event_seconds",
        "Time from enqueue to receipt of the first valid tokens event.",
    ),
    (
        "apxinf_worker_first_token_seconds",
        "Worker reported attempt time to first token, on the worker clock.",
    ),
    (
        "apxinf_time_to_first_public_output_ready_seconds",
        "Time from enqueue to the first content part in the public result channel.",
    ),
    (
        "apxinf_worker_output_event_interval_seconds",
        "Receive interval between adjacent worker tokens events, not individual tokens.",
    ),
    (
        "apxinf_worker_terminal_to_settlement_seconds",
        "Time from worker terminal receipt to confirmed resource settlement.",
    ),
    (
        "apxinf_public_terminal_to_settlement_seconds",
        "Resource settlement delay after the public terminal result.",
    ),
];

#[derive(Default)]
struct HistogramState {
    bins: [u64; BOUNDS_MS.len()],
    count: u64,
    sum_ns: u128,
}

#[derive(Default)]
struct Histogram(Mutex<HistogramState>);

impl Histogram {
    fn observe(&self, value: Duration) {
        let mut state = self.0.lock().expect("The histogram lock is poisoned.");
        state.count = state.count.saturating_add(1);
        state.sum_ns = state.sum_ns.saturating_add(value.as_nanos());
        if let Some(index) = BOUNDS_MS
            .iter()
            .position(|ms| value <= Duration::from_millis(*ms))
        {
            state.bins[index] = state.bins[index].saturating_add(1);
        }
    }

    fn append(&self, output: &mut String, name: &str, labels: &str) {
        let state = self.0.lock().expect("The histogram lock is poisoned.");
        let mut cumulative = 0u64;
        for (limit, count) in BOUNDS_MS.iter().zip(state.bins) {
            cumulative = cumulative.saturating_add(count);
            writeln!(
                output,
                "{name}_bucket{{{labels},le=\"{}\"}} {cumulative}",
                *limit as f64 / 1000.0
            )
            .unwrap();
        }
        writeln!(
            output,
            "{name}_bucket{{{labels},le=\"+Inf\"}} {}",
            state.count
        )
        .unwrap();
        writeln!(
            output,
            "{name}_sum{{{labels}}} {}",
            state.sum_ns as f64 / 1e9
        )
        .unwrap();
        writeln!(output, "{name}_count{{{labels}}} {}", state.count).unwrap();
    }
}

#[derive(Default)]
struct OperationMetrics {
    intervals: [Histogram; HISTOGRAMS.len()],
    requests: [Histogram; STATUSES.len()],
    outcomes: [AtomicU64; STATUSES.len()],
    cleanup_pending: AtomicU64,
}

#[derive(Default)]
pub(crate) struct ObservationMetrics([OperationMetrics; OPERATIONS.len()]);

impl ObservationMetrics {
    pub(crate) fn append(&self, output: &mut String) -> [u64; STATUSES.len()] {
        let outcomes: [[u64; STATUSES.len()]; OPERATIONS.len()] =
            std::array::from_fn(|operation| {
                std::array::from_fn(|status| {
                    self.0[operation].outcomes[status].load(Ordering::Relaxed)
                })
            });
        for (index, (name, help)) in HISTOGRAMS.iter().enumerate() {
            writeln!(output, "# HELP {name} {help}\n# TYPE {name} histogram").unwrap();
            for (operation, metrics) in OPERATIONS.iter().zip(&self.0) {
                metrics.intervals[index].append(
                    output,
                    name,
                    &format!("operation=\"{operation}\""),
                );
            }
        }
        output.push_str("# HELP apxinf_service_request_seconds Time from enqueue to the public terminal result.\n# TYPE apxinf_service_request_seconds histogram\n");
        for (operation, metrics) in OPERATIONS.iter().zip(&self.0) {
            for (index, status) in STATUSES.iter().enumerate() {
                metrics.requests[index].append(
                    output,
                    "apxinf_service_request_seconds",
                    &format!("operation=\"{operation}\",status=\"{status}\""),
                );
            }
        }
        output.push_str("# HELP apxinf_request_outcomes_total Accepted request outcomes by operation.\n# TYPE apxinf_request_outcomes_total counter\n");
        for (operation, counts) in OPERATIONS.iter().zip(&outcomes) {
            for (status, count) in STATUSES.iter().zip(counts) {
                writeln!(output, "apxinf_request_outcomes_total{{operation=\"{operation}\",status=\"{status}\"}} {count}").unwrap();
            }
        }
        output.push_str("# HELP apxinf_cleanup_pending Public terminal requests that still own an execution position or resources.\n# TYPE apxinf_cleanup_pending gauge\n");
        for (operation, metrics) in OPERATIONS.iter().zip(&self.0) {
            writeln!(
                output,
                "apxinf_cleanup_pending{{operation=\"{operation}\"}} {}",
                metrics.cleanup_pending.load(Ordering::Relaxed)
            )
            .unwrap();
        }
        std::array::from_fn(|status| outcomes.iter().map(|operation| operation[status]).sum())
    }
}

#[derive(Default)]
struct RequestState {
    queue_ended: bool,
    preparation_start: Option<Instant>,
    preparation_ended: bool,
    last_worker_output: Option<Instant>,
    first_public_output: bool,
    worker_terminal: Option<Instant>,
    public_terminal: Option<Instant>,
    settled: Option<Instant>,
}

pub(crate) struct RequestObservation {
    ingress: Instant,
    operation: usize,
    state: Mutex<RequestState>,
}

impl RequestObservation {
    pub(crate) fn new(ingress: Instant, count_only: bool) -> Self {
        Self {
            ingress,
            operation: usize::from(count_only),
            state: Mutex::new(RequestState::default()),
        }
    }

    pub(crate) fn queue_end(&self, metrics: &ObservationMetrics, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        if !state.queue_ended {
            state.queue_ended = true;
            self.observe(
                metrics,
                Interval::Queue,
                now.saturating_duration_since(self.ingress),
            );
        }
    }

    pub(crate) fn preparation_start(&self, now: Instant) {
        self.state
            .lock()
            .expect("The request observation lock is poisoned.")
            .preparation_start
            .get_or_insert(now);
    }

    pub(crate) fn preparation_end(&self, metrics: &ObservationMetrics, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        self.end_preparation(&mut state, metrics, now);
    }

    fn end_preparation(
        &self,
        state: &mut RequestState,
        metrics: &ObservationMetrics,
        now: Instant,
    ) {
        if !state.preparation_ended {
            if let Some(start) = state.preparation_start {
                self.observe(
                    metrics,
                    Interval::Preparation,
                    now.saturating_duration_since(start),
                );
                state.preparation_ended = true;
            }
        }
    }

    pub(crate) fn worker_output(&self, metrics: &ObservationMetrics, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        match state.last_worker_output.replace(now) {
            Some(previous) => self.observe(
                metrics,
                Interval::WorkerOutputGap,
                now.saturating_duration_since(previous),
            ),
            None => self.observe(
                metrics,
                Interval::FirstWorker,
                now.saturating_duration_since(self.ingress),
            ),
        }
    }

    pub(crate) fn public_output(&self, metrics: &ObservationMetrics, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        if !state.first_public_output {
            state.first_public_output = true;
            self.observe(
                metrics,
                Interval::FirstPublic,
                now.saturating_duration_since(self.ingress),
            );
        }
    }

    pub(crate) fn worker_terminal(
        &self,
        metrics: &ObservationMetrics,
        now: Instant,
        first_token_ns: Option<u64>,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        if state.worker_terminal.is_none() {
            state.worker_terminal = Some(now);
            if state.last_worker_output.is_some() {
                if let Some(value) = first_token_ns {
                    self.observe(
                        metrics,
                        Interval::WorkerFirstToken,
                        Duration::from_nanos(value),
                    );
                }
            }
        }
    }

    pub(crate) fn public_terminal(
        &self,
        metrics: &ObservationMetrics,
        outcome: Outcome,
        now: Instant,
    ) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        if state.public_terminal.is_some() {
            return;
        }
        state.public_terminal = Some(now);
        let operation = &metrics.0[self.operation];
        operation.outcomes[outcome as usize].fetch_add(1, Ordering::Relaxed);
        operation.requests[outcome as usize].observe(now.saturating_duration_since(self.ingress));
        if let Some(settled) = state.settled {
            self.observe(
                metrics,
                Interval::PublicCleanup,
                settled.saturating_duration_since(now),
            );
        } else {
            operation.cleanup_pending.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn settled(&self, metrics: &ObservationMetrics, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The request observation lock is poisoned.");
        if state.settled.is_some() {
            return;
        }
        state.settled = Some(now);
        self.end_preparation(&mut state, metrics, now);
        if let Some(terminal) = state.worker_terminal {
            self.observe(
                metrics,
                Interval::WorkerCleanup,
                now.saturating_duration_since(terminal),
            );
        }
        if let Some(terminal) = state.public_terminal {
            metrics.0[self.operation]
                .cleanup_pending
                .fetch_sub(1, Ordering::Relaxed);
            self.observe(
                metrics,
                Interval::PublicCleanup,
                now.saturating_duration_since(terminal),
            );
        }
    }

    fn observe(&self, metrics: &ObservationMetrics, interval: Interval, value: Duration) {
        metrics.0[self.operation].intervals[interval as usize].observe(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_histogram(histogram: &Histogram, count: u64, sum_ms: u128) {
        let state = histogram.0.lock().unwrap();
        assert_eq!(state.count, count);
        assert_eq!(state.sum_ns, sum_ms * 1_000_000);
    }

    #[test]
    fn histogram_exports_inclusive_cumulative_buckets_and_infinity() {
        let histogram = Histogram::default();
        for ms in [0, 1, 2, 5, 3_600_001] {
            histogram.observe(Duration::from_millis(ms));
        }
        let mut text = String::new();
        histogram.append(&mut text, "test_seconds", "operation=\"generate\"");
        for expected in [
            "le=\"0.001\"} 2",
            "le=\"0.005\"} 4",
            "le=\"3600\"} 4",
            "le=\"+Inf\"} 5",
        ] {
            assert!(text.contains(expected), "{text}");
        }
        assert!(text.contains("test_seconds_count{operation=\"generate\"} 5"));
        assert_histogram(&histogram, 5, 3_600_009);
    }

    #[test]
    fn successful_request_distinguishes_worker_public_and_cleanup_clocks() {
        let metrics = ObservationMetrics::default();
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let request = RequestObservation::new(start, false);
        request.queue_end(&metrics, at(20));
        request.preparation_start(at(20));
        request.preparation_end(&metrics, at(50));
        request.worker_output(&metrics, at(80));
        request.public_output(&metrics, at(90));
        request.worker_output(&metrics, at(95));
        request.public_output(&metrics, at(96));
        request.worker_terminal(&metrics, at(100), Some(30_000_000));
        request.settled(&metrics, at(110));
        request.public_terminal(&metrics, Outcome::Completed, at(120));
        for (index, value) in [20, 30, 80, 30, 90, 15, 10, 0].into_iter().enumerate() {
            assert_histogram(&metrics.0[0].intervals[index], 1, value);
        }
        assert_histogram(&metrics.0[0].requests[0], 1, 120);
        assert_eq!(metrics.0[0].cleanup_pending.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn public_expiry_retains_pending_cleanup_until_real_settlement() {
        let metrics = ObservationMetrics::default();
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let request = RequestObservation::new(start, true);
        request.queue_end(&metrics, at(2));
        request.preparation_start(at(5));
        request.public_terminal(&metrics, Outcome::Expired, at(10));
        request.public_terminal(&metrics, Outcome::Failed, at(20));
        assert_eq!(metrics.0[1].cleanup_pending.load(Ordering::Relaxed), 1);
        assert_histogram(
            &metrics.0[1].intervals[Interval::Preparation as usize],
            0,
            0,
        );
        assert_histogram(
            &metrics.0[1].intervals[Interval::PublicCleanup as usize],
            0,
            0,
        );
        request.settled(&metrics, at(50));
        request.settled(&metrics, at(60));
        assert_eq!(metrics.0[1].cleanup_pending.load(Ordering::Relaxed), 0);
        assert_eq!(
            metrics.0[1].outcomes[Outcome::Expired as usize].load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            metrics.0[1].outcomes[Outcome::Failed as usize].load(Ordering::Relaxed),
            0
        );
        assert_histogram(
            &metrics.0[1].intervals[Interval::Preparation as usize],
            1,
            45,
        );
        assert_histogram(
            &metrics.0[1].intervals[Interval::PublicCleanup as usize],
            1,
            40,
        );
        for interval in [
            Interval::FirstWorker,
            Interval::WorkerFirstToken,
            Interval::FirstPublic,
            Interval::WorkerOutputGap,
            Interval::WorkerCleanup,
        ] {
            assert_histogram(&metrics.0[1].intervals[interval as usize], 0, 0);
        }
    }

    #[test]
    fn queue_expiry_and_zero_output_do_not_invent_first_token_samples() {
        let metrics = ObservationMetrics::default();
        let start = Instant::now();
        let at = start + Duration::from_millis(20);
        let request = RequestObservation::new(start, false);
        request.queue_end(&metrics, at);
        request.settled(&metrics, at);
        request.public_terminal(&metrics, Outcome::Expired, at);
        let empty = RequestObservation::new(at, false);
        empty.worker_terminal(&metrics, at, Some(0));
        empty.settled(&metrics, at);
        empty.public_terminal(&metrics, Outcome::Completed, at);
        assert_histogram(
            &metrics.0[0].intervals[Interval::WorkerFirstToken as usize],
            0,
            0,
        );
        assert_histogram(
            &metrics.0[0].intervals[Interval::PublicCleanup as usize],
            2,
            0,
        );
        let mut text = String::new();
        metrics.append(&mut text);
        assert_eq!(
            text.matches("# TYPE apxinf_queue_wait_seconds histogram")
                .count(),
            1
        );
        assert!(text.contains(
            "apxinf_request_outcomes_total{operation=\"generate\",status=\"expired\"} 1"
        ));
        assert!(!text.contains("request_id="));
    }
}
