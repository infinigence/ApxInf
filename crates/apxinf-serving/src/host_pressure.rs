//! Host memory pressure gates new work without owning worker resources.
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

pub(crate) const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
const STALE_AFTER: Duration = Duration::from_secs(3);
const STATES: [&str; 6] = [
    "disabled", "normal", "warning", "critical", "unknown", "stale",
];
const REASONS: [&str; 5] = ["warning", "critical", "unknown", "stale", "recovering"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HostPressurePolicy {
    Macos,
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pressure {
    Normal,
    Warning,
    Critical,
}

impl Pressure {
    fn from_dispatch(value: u32) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Warning),
            4 => Ok(Self::Critical),
            _ => Err(format!(
                "The host pressure sensor returned an unknown value: {value}."
            )),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

struct State {
    level: Option<Pressure>,
    sampled_at: Option<Instant>,
    allowed: bool,
    normal_since: Option<Instant>,
    read_errors: u64,
    last_error: Option<String>,
    rejections: [[u64; 5]; 2],
}

pub(crate) struct HostPressure {
    policy: HostPressurePolicy,
    state: Mutex<State>,
    stopped: AtomicBool,
    sensor: fn() -> Result<Pressure, String>,
}

pub(crate) struct Snapshot {
    policy: HostPressurePolicy,
    pub(crate) state: &'static str,
    pub(crate) allowed: bool,
    age: Option<Duration>,
    recovering: bool,
}

impl Snapshot {
    fn reason(&self) -> Option<&'static str> {
        if self.allowed {
            None
        } else if self.recovering {
            Some("recovering")
        } else {
            Some(self.state)
        }
    }
    pub(crate) fn json(&self) -> Value {
        json!({
            "policy": match self.policy { HostPressurePolicy::Macos => "macos", HostPressurePolicy::Disabled => "disabled" },
            "state": self.state,
            "admission_allowed": self.allowed,
            "recovering": self.recovering,
            "sample_age_seconds": self.age.map(|age| age.as_secs_f64()),
        })
    }
}

impl HostPressure {
    pub(crate) fn new(policy: HostPressurePolicy) -> Self {
        Self {
            policy,
            state: Mutex::new(State {
                level: None,
                sampled_at: None,
                allowed: false,
                normal_since: None,
                read_errors: 0,
                last_error: None,
                rejections: [[0; 5]; 2],
            }),
            stopped: AtomicBool::new(false),
            sensor: read_pressure,
        }
    }

    pub(crate) fn refresh(&self) {
        if self.policy == HostPressurePolicy::Macos {
            let result = (self.sensor)();
            self.record(result, Instant::now());
        }
    }

    pub(crate) fn require_load(&self) -> Result<(), String> {
        self.refresh();
        let snapshot = self.snapshot(Instant::now());
        match snapshot.reason() {
            None => Ok(()),
            Some(reason) => {
                let state = self
                    .state
                    .lock()
                    .expect("The host pressure lock is poisoned.");
                let detail = state.last_error.as_deref().unwrap_or("");
                Err(format!(
                    "Host memory pressure prevents model loading: {reason}. {detail}"
                ))
            }
        }
    }

    pub(crate) fn record(&self, result: Result<Pressure, String>, now: Instant) {
        let mut state = self
            .state
            .lock()
            .expect("The host pressure lock is poisoned.");
        let before = self.snapshot_locked(&state, now);
        let first = state.sampled_at.is_none();
        if before.state == "stale" {
            state.allowed = false;
            state.normal_since = None;
        }
        state.sampled_at = Some(now);
        state.last_error = result.as_ref().err().cloned();
        match result {
            Ok(Pressure::Normal) => {
                state.level = Some(Pressure::Normal);
                if first {
                    state.allowed = true;
                } else if !state.allowed {
                    match state.normal_since {
                        Some(start) if now.saturating_duration_since(start) >= SAMPLE_PERIOD => {
                            state.allowed = true;
                        }
                        None => state.normal_since = Some(now),
                        _ => {}
                    }
                }
            }
            level => {
                state.level = level.ok();
                state.allowed = false;
                state.normal_since = None;
                if state.level.is_none() {
                    state.read_errors += 1;
                }
            }
        }
    }

    fn snapshot_locked(&self, state: &State, now: Instant) -> Snapshot {
        if self.policy == HostPressurePolicy::Disabled {
            return Snapshot {
                policy: self.policy,
                state: "disabled",
                allowed: true,
                age: None,
                recovering: false,
            };
        }
        let age = state
            .sampled_at
            .map(|sampled| now.saturating_duration_since(sampled));
        let name = if age.is_some_and(|age| age >= STALE_AFTER) {
            "stale"
        } else {
            state.level.map(Pressure::name).unwrap_or("unknown")
        };
        Snapshot {
            policy: self.policy,
            state: name,
            allowed: name == "normal" && state.allowed,
            age,
            recovering: name == "normal" && !state.allowed,
        }
    }

    pub(crate) fn snapshot(&self, now: Instant) -> Snapshot {
        let state = self
            .state
            .lock()
            .expect("The host pressure lock is poisoned.");
        self.snapshot_locked(&state, now)
    }

    pub(crate) fn check(&self, dispatch: bool) -> Result<(), &'static str> {
        let mut state = self
            .state
            .lock()
            .expect("The host pressure lock is poisoned.");
        match self.snapshot_locked(&state, Instant::now()).reason() {
            None => Ok(()),
            Some(reason) => {
                let index = REASONS
                    .iter()
                    .position(|candidate| *candidate == reason)
                    .unwrap();
                state.rejections[usize::from(dispatch)][index] += 1;
                Err(reason)
            }
        }
    }

    pub(crate) fn append_metrics(&self, text: &mut String) {
        use std::fmt::Write;
        let state = self
            .state
            .lock()
            .expect("The host pressure lock is poisoned.");
        let snapshot = self.snapshot_locked(&state, Instant::now());
        writeln!(
            text,
            "apxinf_host_pressure_enabled {}",
            u8::from(self.policy == HostPressurePolicy::Macos)
        )
        .unwrap();
        for name in STATES {
            writeln!(
                text,
                "apxinf_host_pressure_state{{state=\"{name}\"}} {}",
                u8::from(name == snapshot.state)
            )
            .unwrap();
        }
        writeln!(
            text,
            "apxinf_host_admission_allowed {}",
            u8::from(snapshot.allowed)
        )
        .unwrap();
        writeln!(
            text,
            "apxinf_host_pressure_sample_age_seconds {}",
            snapshot
                .age
                .map(|age| age.as_secs_f64())
                .unwrap_or(f64::NAN)
        )
        .unwrap();
        writeln!(
            text,
            "apxinf_host_pressure_read_errors_total {}",
            state.read_errors
        )
        .unwrap();
        for (stage, counts) in ["enqueue", "dispatch"].into_iter().zip(state.rejections) {
            for (reason, count) in REASONS.into_iter().zip(counts) {
                writeln!(text, "apxinf_host_pressure_rejections_total{{stage=\"{stage}\",reason=\"{reason}\"}} {count}").unwrap();
            }
        }
    }

    pub(crate) fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn with_sensor(sensor: fn() -> Result<Pressure, String>) -> Self {
        let mut guard = Self::new(HostPressurePolicy::Macos);
        guard.sensor = sensor;
        guard
    }

    pub(crate) fn monitor(guard: &Arc<Self>) {
        if guard.policy == HostPressurePolicy::Disabled {
            return;
        }
        let guard = Arc::downgrade(guard);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(SAMPLE_PERIOD);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let Some(guard) = guard.upgrade() else {
                    break;
                };
                if guard.stopped.load(Ordering::Acquire) {
                    break;
                }
                guard.refresh();
            }
        });
    }
}

#[cfg(target_os = "macos")]
fn read_pressure() -> Result<Pressure, String> {
    use std::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut value = 0u32;
    let mut size = std::mem::size_of_val(&value);
    // The fixed NUL-terminated name and output buffer remain valid during this read-only call.
    let status = unsafe {
        sysctlbyname(
            b"kern.memorystatus_vm_pressure_level\0".as_ptr().cast(),
            (&mut value as *mut u32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return Err(format!(
            "The host pressure read failed: {}.",
            std::io::Error::last_os_error()
        ));
    }
    if size != std::mem::size_of_val(&value) {
        return Err("The host pressure sensor returned an invalid size.".into());
    }
    Pressure::from_dispatch(value)
}

#[cfg(not(target_os = "macos"))]
fn read_pressure() -> Result<Pressure, String> {
    Err("The macos host pressure policy requires macOS.".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_values_are_not_internal_kernel_levels() {
        assert_eq!(Pressure::from_dispatch(1).unwrap(), Pressure::Normal);
        assert_eq!(Pressure::from_dispatch(2).unwrap(), Pressure::Warning);
        assert_eq!(Pressure::from_dispatch(4).unwrap(), Pressure::Critical);
        for value in [0, 3, 5, u32::MAX] {
            assert!(Pressure::from_dispatch(value).is_err());
        }
    }

    #[test]
    fn pressure_and_failed_reads_need_two_spaced_normal_samples() {
        for level in [
            Ok(Pressure::Warning),
            Ok(Pressure::Critical),
            Err("denied".into()),
        ] {
            let guard = HostPressure::new(HostPressurePolicy::Macos);
            let now = Instant::now();
            assert!(!guard.snapshot(now).allowed);
            guard.record(Ok(Pressure::Normal), now);
            assert!(guard.snapshot(now).allowed);
            guard.record(level, now);
            assert!(!guard.snapshot(now).allowed);
            guard.record(Ok(Pressure::Normal), now);
            assert_eq!(guard.snapshot(now).reason(), Some("recovering"));
            guard.record(Ok(Pressure::Normal), now + Duration::from_millis(999));
            assert!(!guard.snapshot(now + Duration::from_millis(999)).allowed);
            guard.record(Ok(Pressure::Normal), now + SAMPLE_PERIOD);
            assert!(guard.snapshot(now + SAMPLE_PERIOD).allowed);
        }
    }

    #[test]
    fn stale_normal_samples_do_not_reopen_admission_on_one_read() {
        let guard = HostPressure::new(HostPressurePolicy::Macos);
        let now = Instant::now();
        guard.record(Ok(Pressure::Normal), now);
        assert!(
            guard
                .snapshot(now + STALE_AFTER - Duration::from_nanos(1))
                .allowed
        );
        assert_eq!(guard.snapshot(now + STALE_AFTER).state, "stale");
        guard.record(Ok(Pressure::Normal), now + STALE_AFTER);
        assert!(!guard.snapshot(now + STALE_AFTER).allowed);
        guard.record(Ok(Pressure::Normal), now + STALE_AFTER * 2);
        assert!(!guard.snapshot(now + STALE_AFTER * 2).allowed);
        guard.record(Ok(Pressure::Normal), now + STALE_AFTER * 2 + SAMPLE_PERIOD);
        assert!(
            guard
                .snapshot(now + STALE_AFTER * 2 + SAMPLE_PERIOD)
                .allowed
        );
    }

    #[test]
    fn unavailable_sensor_does_not_default_to_normal() {
        let guard = HostPressure::new(HostPressurePolicy::Macos);
        let now = Instant::now();
        guard.record(Err("not supported".into()), now);
        assert_eq!(guard.check(false), Err("unknown"));
        guard.record(Ok(Pressure::Normal), now);
        assert!(!guard.snapshot(now).allowed);
        let mut metrics = String::new();
        guard.append_metrics(&mut metrics);
        assert!(metrics.contains("apxinf_host_pressure_read_errors_total 1\n"));
        assert!(metrics.contains("{stage=\"enqueue\",reason=\"unknown\"} 1\n"));
    }

    #[test]
    fn disabled_policy_is_visible_without_invented_measurements() {
        let guard = HostPressure::new(HostPressurePolicy::Disabled);
        guard.require_load().unwrap();
        let snapshot = guard.snapshot(Instant::now());
        assert!(snapshot.allowed);
        assert_eq!(snapshot.json()["sample_age_seconds"], Value::Null);
        assert_eq!(snapshot.json()["state"], "disabled");
        let mut metrics = String::new();
        guard.append_metrics(&mut metrics);
        assert!(metrics.contains("apxinf_host_pressure_enabled 0\n"));
        assert!(metrics.contains("apxinf_host_pressure_sample_age_seconds NaN\n"));
    }

    #[test]
    fn closed_stderr_does_not_break_pressure_decisions() {
        use std::process::{Command, Stdio};
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "host_pressure::tests::unavailable_sensor_does_not_default_to_normal",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        drop(child.stderr.take());
        assert!(child.wait().unwrap().success());
    }

    #[test]
    #[ignore = "Reads the actual host sensor. Run explicitly outside the filesystem sandbox."]
    fn actual_macos_sensor() {
        let pressure = read_pressure().unwrap();
        eprintln!("Actual host pressure: {}", pressure.name());
    }
}
