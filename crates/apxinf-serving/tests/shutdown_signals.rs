#![cfg(unix)]
use apxinf_serving::{host_pressure::HostPressurePolicy, supervisor::Config};
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};
#[path = "support/signal_fixture.rs"]
mod signal_fixture;
use signal_fixture::Fixture;

struct Cli {
    child: Child,
    fixture: Fixture,
}

impl Cli {
    fn start(mode: &str) -> Self {
        let fixture = Fixture::new(mode);
        let log = fs::File::create(fixture.root.join("cli.log")).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_apxinf-serve"))
            .arg("--python")
            .arg(&fixture.config.python)
            .arg("--worker")
            .arg(&fixture.config.worker)
            .arg("--model")
            .arg(&fixture.config.model)
            .args([
                "--host-pressure-policy",
                "disabled",
                "--port",
                "0",
                "--max-context",
                "64",
                "--max-output-tokens",
                "16",
                "--memory-budget-bytes",
                "4096",
                "--sequence-reservation-bytes",
                "1024",
                "--timeout-seconds",
                "5",
            ])
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        Self { child, fixture }
    }

    fn until(&mut self, mut ready: impl FnMut(&Fixture) -> bool) {
        let until = Instant::now() + Duration::from_secs(5);
        while !ready(&self.fixture) {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "CLI exited early: {}",
                self.log()
            );
            assert!(
                Instant::now() < until,
                "CLI did not reach the test boundary: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn log(&self) -> String {
        fs::read_to_string(self.fixture.root.join("cli.log")).unwrap()
    }

    fn signal(&self, signal: &str) {
        assert!(Command::new("/bin/kill")
            .args([signal, &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
    }

    fn wait(&mut self) -> ExitStatus {
        self.wait_for(Duration::from_secs(5))
    }

    fn wait_for(&mut self, timeout: Duration) -> ExitStatus {
        let until = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(
                Instant::now() < until,
                "CLI did not reap and exit: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn address(&mut self) -> String {
        self.until(|fixture| {
            fs::read_to_string(fixture.root.join("cli.log"))
                .unwrap()
                .contains("ApxInf listener bound at http://")
        });
        self.log()
            .lines()
            .find_map(|line| line.strip_prefix("ApxInf listener bound at http://"))
            .unwrap()
            .to_owned()
    }
}

impl Drop for Cli {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[test]
fn sigint_during_loading_reaps_before_cli_exit() {
    loading_signal("-INT");
}

#[test]
fn sigterm_during_loading_reaps_before_cli_exit() {
    loading_signal("-TERM");
}

fn loading_signal(signal: &str) {
    let mut cli = Cli::start("loading");
    cli.until(|fixture| fixture.worker_pid().is_some());
    cli.signal(signal);
    assert!(
        cli.wait().success(),
        "Startup cancellation failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
    assert!(!cli.log().contains("listener bound"));
}

#[test]
fn sigterm_after_ready_stops_and_reaps_the_worker() {
    let mut cli = Cli::start("ready");
    let _ = cli.address();
    cli.signal("-TERM");
    assert!(
        cli.wait().success(),
        "Runtime shutdown failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

#[test]
fn sigint_preserves_the_active_operation_until_settlement() {
    let mut cli = Cli::start("ready");
    let mut connection = count_request(&mut cli);
    cli.until(|fixture| fixture.root.join("preparing").exists());
    cli.signal("-INT");
    thread::sleep(Duration::from_millis(150));
    assert!(
        cli.child.try_wait().unwrap().is_none(),
        "Active work was abandoned."
    );
    fs::write(cli.fixture.root.join("release"), "finish").unwrap();
    let mut response = String::new();
    connection.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"input_tokens\":2"), "{response}");
    assert!(
        cli.wait().success(),
        "Active shutdown failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

fn count_request(cli: &mut Cli) -> TcpStream {
    let mut connection = TcpStream::connect(cli.address()).unwrap();
    connection
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let body = r#"{"model":"apxinf-local","messages":[{"role":"user","content":"count this"}]}"#;
    write!(connection,
        "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()).unwrap();
    connection
}

#[test]
fn unfinished_http_body_has_a_drain_limit_after_worker_reaping() {
    let mut cli = Cli::start("ready");
    let mut active = count_request(&mut cli);
    cli.until(|fixture| fixture.root.join("preparing").exists());
    let mut incomplete = TcpStream::connect(cli.address()).unwrap();
    incomplete
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    incomplete.write_all(b"POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nExpect: 100-continue\r\nContent-Length: 100000\r\n\r\n").unwrap();
    let mut interim = [0; 25];
    incomplete.read_exact(&mut interim).unwrap();
    assert_eq!(&interim, b"HTTP/1.1 100 Continue\r\n\r\n");
    incomplete.write_all(b"{").unwrap();
    cli.signal("-TERM");

    // An overall 20-second shutdown timer would incorrectly include this active operation.
    thread::sleep(Duration::from_secs(2));
    assert!(cli.child.try_wait().unwrap().is_none());
    fs::write(cli.fixture.root.join("release"), "finish").unwrap();
    let mut response = String::new();
    active.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    cli.until(|fixture| {
        !Command::new("/bin/kill")
            .args(["-0", &fixture.worker_pid().unwrap().to_string()])
            .output()
            .unwrap()
            .status
            .success()
    });
    let reaped = Instant::now();
    assert!(
        cli.wait_for(Duration::from_secs(25)).success(),
        "{}",
        cli.log()
    );
    assert!(
        reaped.elapsed() >= Duration::from_secs(19),
        "HTTP grace included worker settlement."
    );
    assert!(cli
        .log()
        .contains("HTTP connections did not drain within 20 seconds after worker reaping."));
    cli.fixture.assert_reaped();
    drop(incomplete);
}

#[test]
fn startup_error_returns_failure_after_reaping() {
    let mut cli = Cli::start("invalid_ready");
    assert!(!cli.wait().success());
    assert!(cli.log().contains("ApxInf failed:"));
    cli.fixture.assert_reaped();
}
