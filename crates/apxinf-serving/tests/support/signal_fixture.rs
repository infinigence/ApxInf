use super::{Config, HostPressurePolicy};
use std::{fs, path::PathBuf, process::Command, time::Duration};

pub struct Fixture {
    pub root: PathBuf,
    pub config: Config,
}

impl Fixture {
    pub fn new(mode: &str) -> Self {
        let root = std::env::temp_dir().join(format!("apxinf-signal-{}", uuid::Uuid::new_v4()));
        let worker = root.join("python/apxinf/apxinf/serving/worker.py");
        fs::create_dir_all(worker.parent().unwrap()).unwrap();
        fs::create_dir_all(root.join("model")).unwrap();
        fs::create_dir_all(root.join("scripts")).unwrap();
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/serving/signal_worker.py"),
            &worker,
        )
        .unwrap();
        fs::write(
            worker.with_file_name("contracts.py"),
            "# CPU identity marker\n",
        )
        .unwrap();
        fs::write(
            root.join("scripts/apxinf_mlx_generate.py"),
            "# CPU identity marker\n",
        )
        .unwrap();
        fs::write(root.join("mode"), mode).unwrap();
        let python = Command::new("python3")
            .args(["-c", "import sys; print(sys.executable)"])
            .output()
            .unwrap();
        assert!(python.status.success());
        let python = fs::canonicalize(String::from_utf8(python.stdout).unwrap().trim()).unwrap();
        let config = Config {
            python,
            worker,
            model: root.join("model"),
            model_id: "cpu-signals".into(),
            max_context: 64,
            max_output: 16,
            prefill_step_size: 4,
            output_batch_tokens: 1,
            queue_capacity: 2,
            timeout: Duration::from_secs(5),
            memory_budget: 4096,
            sequence_reservation: 1024,
            host_pressure_policy: HostPressurePolicy::Disabled,
        };
        Self { root, config }
    }

    pub fn worker_pid(&self) -> Option<u32> {
        fs::read_to_string(self.root.join("worker.pid"))
            .ok()?
            .parse()
            .ok()
    }

    pub fn assert_reaped(&self) {
        let pid = self.worker_pid().expect("The CPU worker must start.");
        let status = Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .output()
            .unwrap();
        assert!(
            !status.status.success(),
            "Worker {pid} still exists after completion."
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(pid) = self.worker_pid() {
            let observed = Command::new("/bin/ps")
                .args(["-o", "command=", "-p", &pid.to_string()])
                .output();
            if observed.is_ok_and(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .contains(self.config.worker.to_str().unwrap())
            }) {
                let _ = Command::new("/bin/kill")
                    .args(["-KILL", &pid.to_string()])
                    .output();
            }
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}
