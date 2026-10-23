//! Isolated CPU subprocess checks for unavailable diagnostic output.
use super::lifecycle_regressions::{cpu_service, cpu_worker, request, CpuModel};
use super::*;
use std::os::unix::process::CommandExt;
use std::process::{Child as ProbeChild, Command as ProbeCommand};
use tokio::io::AsyncReadExt;

const PROBE_MODE: &str = "APXINF_SERVING_DIAGNOSTIC_PROBE";
const PROBE_MODEL: &str = "APXINF_SERVING_DIAGNOSTIC_MODEL";
const COMPLETED_MARKER: &str = "APXINF_DIAGNOSTIC_PROBE_COMPLETED";

extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

struct OwnedProbe {
    child: ProbeChild,
    group: i32,
}

impl OwnedProbe {
    fn reap_group(&mut self) {
        // process_group(0) gives this child an exclusive, owned process group.
        unsafe { kill(-self.group, 9) };
        let _ = self.child.wait();
    }
}

impl Drop for OwnedProbe {
    fn drop(&mut self) {
        self.reap_group();
    }
}

fn group_has_live_members(group: i32) -> bool {
    let output = ProbeCommand::new("/bin/ps")
        .args(["-axo", "pgid=,stat="])
        .output()
        .expect("The owned process group must be inspectable.");
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .any(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            fields.len() == 2
                && fields[0].parse::<i32>() == Ok(group)
                && !fields[1].starts_with('Z')
        })
}

fn run_isolated(output_mode: &str, scenario: &str) {
    let model = CpuModel::new(json!({}));
    let mut command = ProbeCommand::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "supervisor::diagnostic_regressions::diagnostic_probe_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE_MODE, format!("{output_mode}:{scenario}"))
        .env(PROBE_MODEL, model.model())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let child = command.spawn().unwrap();
    let group = child.id() as i32;
    let mut probe = OwnedProbe { child, group };
    let mut output = probe.child.stdout.take().unwrap();
    let error_pipe = probe.child.stderr.take().unwrap();
    // Keep the full-pipe read endpoint open without reading any bytes.
    let _undrained = if output_mode == "closed" {
        drop(error_pipe);
        None
    } else {
        Some(error_pipe)
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = probe.child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // A failing or blocked probe cannot leave its worker alive.
    probe.reap_group();
    let cleanup_deadline = Instant::now() + Duration::from_secs(2);
    while group_has_live_members(group) && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !group_has_live_members(group),
        "An owned probe process remains alive."
    );
    let mut transcript = String::new();
    output
        .by_ref()
        .take(64 * 1024)
        .read_to_string(&mut transcript)
        .unwrap();
    let status = status.unwrap_or_else(|| {
        panic!(
            "The {output_mode}/{scenario} probe blocked past its hard parent timeout. {transcript}"
        )
    });
    assert!(
        status.success(),
        "The {output_mode}/{scenario} probe failed: {status}. {transcript}"
    );
    assert!(
        transcript.contains(COMPLETED_MARKER),
        "The probe did not confirm lifecycle completion. {transcript}"
    );
}

fn fill_own_stderr_pipe() {
    // Only this probe and its children share fd 2. Fill it without blocking here.
    let status = ProbeCommand::new("python3")
        .args([
            "-c",
            "import os\nos.set_blocking(2, False)\ntry:\n for size in (4096, 1):\n  try:\n   while True: os.write(2, b'x' * size)\n  except BlockingIOError: pass\nfinally:\n os.set_blocking(2, True)\n",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .unwrap();
    assert!(
        status.success(),
        "The isolated stderr pipe must become full."
    );
}

async fn require_completed_request(service: &Service, ticket: &mut Ticket) {
    assert!((&mut ticket.begin).await.unwrap().is_ok());
    loop {
        match ticket.output.recv().await {
            Some(Event::Part(_)) => {}
            Some(Event::Done { .. }) => break,
            _ => panic!("A CPU request must complete while diagnostic output is unavailable."),
        }
    }
    assert!(service.available.load(Ordering::Acquire));
}

async fn probe_lifecycle(scenario: &str, model: PathBuf) {
    if scenario == "flood" {
        probe_worker_stderr_flood().await;
        return;
    }
    let (mut worker, ready) = cpu_worker(model.clone()).await.unwrap();
    let worker_pid = worker.child.id().unwrap() as i32;
    let mut service = cpu_service(ready, Duration::from_secs(2), 2);
    let (sender, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    if scenario == "rotation" {
        worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    }
    let mut first = (scenario != "fault").then(|| service.enqueue(request(false)).unwrap());
    if scenario == "fault" {
        worker.child.start_kill().unwrap();
    }
    let replacement_model = model;
    let coordinator = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || cpu_worker(replacement_model.clone()),
        sender,
    ));
    if scenario == "fault" {
        let error = service.wait_stopped().await.unwrap_err();
        assert!(!service.available.load(Ordering::Acquire));
        assert!(service.queue.is_closed());
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
        assert!(
            matches!(
                error.as_str(),
                "Worker stdout ended." | "The idle worker output pipe closed."
            ),
            "A checked worker fault must reach the stop report: {error}"
        );
    } else {
        require_completed_request(&service, first.as_mut().unwrap()).await;
        let mut second = service.enqueue(request(false)).unwrap();
        require_completed_request(&service, &mut second).await;
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 2);
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
        if scenario == "rotation" {
            assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
        }
        service.shutdown();
        service.wait_stopped().await.unwrap();
    }
    coordinator.await.unwrap();
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    // wait_stopped must confirm process exit, not just abandon a worker handle.
    assert_eq!(unsafe { kill(worker_pid, 0) }, -1);
    println!("{COMPLETED_MARKER}");
}

async fn probe_worker_stderr_flood() {
    let script = "import os\nremaining = memoryview(b'\\xff\\x00A' * 350000)\nwhile remaining:\n written = os.write(2, remaining)\n remaining = remaining[written:]\nos.write(1, b'worker stdout completed\\n')\n";
    let mut child = Command::new("python3")
        .args(["-c", script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let pid = child.id().unwrap() as i32;
    let diagnostics = crate::diagnostics::WorkerDiagnostics::start(
        child.stderr.take().unwrap(),
        uuid::Uuid::new_v4().to_string(),
    );
    let mut stdout = child.stdout.take().unwrap();
    let mut output = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), stdout.read_to_end(&mut output))
        .await
        .expect("Worker stderr must not block stdout completion.")
        .unwrap();
    assert_eq!(output, b"worker stdout completed\n");
    assert!(tokio::time::timeout(Duration::from_secs(1), child.wait())
        .await
        .expect("The flood worker must exit.")
        .unwrap()
        .success());
    assert_eq!(unsafe { kill(pid, 0) }, -1);
    drop(diagnostics);
    tokio::time::timeout(Duration::from_millis(500), crate::diagnostics::finish())
        .await
        .expect("Diagnostic delivery must have a bounded completion interval.");
    let mut metrics = String::new();
    crate::diagnostics::append_metrics(&mut metrics);
    let dropped: u64 = metrics
        .lines()
        .filter(|line| line.starts_with("apxinf_diagnostic_dropped_total{"))
        .map(|line| line.rsplit_once(' ').unwrap().1.parse::<u64>().unwrap())
        .sum();
    assert!(
        dropped > 0,
        "Unavailable output must discard records instead of blocking."
    );
    println!("{COMPLETED_MARKER}");
}

#[test]
fn diagnostic_probe_entry() {
    let Ok(mode) = std::env::var(PROBE_MODE) else {
        return;
    };
    std::panic::set_hook(Box::new(|information| {
        use std::io::Write;
        let mut output = std::io::stdout().lock();
        let _ = writeln!(output, "APXINF_DIAGNOSTIC_PROBE_PANIC: {information}");
        let _ = output.flush();
    }));
    let (output_mode, scenario) = mode.split_once(':').unwrap();
    if output_mode == "full" {
        fill_own_stderr_pipe();
    }
    let model = PathBuf::from(std::env::var_os(PROBE_MODEL).unwrap());
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(probe_lifecycle(scenario, model));
}

#[test]
fn diagnostic_closed_stderr_does_not_prevent_worker_fencing() {
    run_isolated("closed", "fault");
}

#[test]
fn diagnostic_full_stderr_does_not_prevent_worker_fencing() {
    run_isolated("full", "fault");
}

#[test]
fn diagnostic_closed_stderr_preserves_settlement_and_following_requests() {
    run_isolated("closed", "settlement");
}

#[test]
fn diagnostic_full_stderr_preserves_settlement_and_following_requests() {
    run_isolated("full", "settlement");
}

#[test]
fn diagnostic_closed_stderr_preserves_rotation_and_replacement_requests() {
    run_isolated("closed", "rotation");
}

#[test]
fn diagnostic_full_stderr_preserves_rotation_and_replacement_requests() {
    run_isolated("full", "rotation");
}

#[test]
fn diagnostic_closed_stderr_drains_unterminated_invalid_utf8_worker_output() {
    run_isolated("closed", "flood");
}

#[test]
fn diagnostic_full_stderr_drains_unterminated_invalid_utf8_worker_output() {
    run_isolated("full", "flood");
}
