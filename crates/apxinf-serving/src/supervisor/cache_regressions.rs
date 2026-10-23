//! CPU imports through the production Clef process launch boundary.
use super::*;
use std::path::Path;

struct Probe(PathBuf);

impl Probe {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!("apxinf-clef-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("probe_package")).unwrap();
        std::fs::create_dir_all(root.join("model")).unwrap();
        Self(root)
    }

    fn observation(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.0.join("model/observation.json")).unwrap())
            .unwrap()
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn source_digest(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

#[tokio::test]
async fn clef_launch_executes_current_source_despite_valid_stale_bytecode() {
    let probe = Probe::new();
    let discovery = tokio::time::timeout(
        Duration::from_secs(5),
        Command::new("python3")
            .args(["-I", "-c", "import sys; print(sys.executable)"])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("Interpreter discovery must finish.")
    .unwrap();
    assert!(discovery.status.success());
    let python = PathBuf::from(String::from_utf8(discovery.stdout).unwrap().trim());
    let worker = probe.0.join("worker_probe.py");
    std::fs::write(
        &worker,
        include_bytes!("../../../../tests/fixtures/serving/clef_cache_probe.py"),
    )
    .unwrap();
    let schema = probe.0.join("clef_contracts.py");
    std::fs::write(
        &schema,
        include_bytes!("../../../../python/apxinf/apxinf/serving/clef_contracts.py"),
    )
    .unwrap();
    std::fs::write(
        probe.0.join("contracts.py"),
        include_bytes!("../../../../python/apxinf/apxinf/serving/contracts.py"),
    )
    .unwrap();
    for name in ["__init__.py", "member.py"] {
        std::fs::write(
            probe.0.join("probe_package").join(name),
            b"MARKER = 'current'\n",
        )
        .unwrap();
    }
    let digest = source_digest(&schema);
    let seeded = tokio::time::timeout(
        Duration::from_secs(5),
        Command::new(&python)
            .arg("-I")
            .arg(&worker)
            .arg("--seed-bytecode")
            .arg(&probe.0)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("Bytecode seeding must finish.")
    .unwrap();
    assert!(
        seeded.status.success(),
        "{}",
        String::from_utf8_lossy(&seeded.stderr)
    );
    let cache_paths: Vec<PathBuf> = serde_json::from_slice(&seeded.stdout).unwrap();
    assert_eq!(cache_paths.len(), 3);
    let cache_bytes: Vec<_> = cache_paths
        .iter()
        .map(|path| std::fs::read(path).unwrap())
        .collect();
    assert_eq!(source_digest(&schema), digest);

    let control = tokio::time::timeout(
        Duration::from_secs(5),
        Command::new(&python)
            .args(["-u", "-B"])
            .arg(&worker)
            .arg("--model")
            .arg(probe.0.join("model"))
            .env_clear()
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("The stale-bytecode control must finish.")
    .unwrap();
    assert!(
        control.status.success(),
        "{}",
        String::from_utf8_lossy(&control.stderr)
    );
    let stale = probe.observation();
    assert_eq!(stale["revision"], "apxinf-python-clef-text-v0");
    assert_eq!(stale["initializer"], "stale__");
    assert_eq!(stale["member"], "stale__");
    assert_eq!(stale["source_sha256"], digest);

    let config = Config {
        runtime: WorkerRuntime::PythonClef {
            loader: probe.0.join("external_loader.py"),
        },
        python: python.clone(),
        worker: worker.clone(),
        model: probe.0.join("model"),
        model_id: "cache-probe".into(),
        max_context: 4096,
        max_output: 0,
        prefill_step_size: 256,
        output_batch_tokens: 1,
        queue_capacity: 1,
        timeout: Duration::from_secs(5),
        memory_budget: 4096,
        sequence_reservation: 1024,
        host_pressure_policy: HostPressurePolicy::Disabled,
    };
    config.validate_memory().unwrap();
    let pressure = HostPressure::new(HostPressurePolicy::Disabled);
    for _ in 0..2 {
        std::fs::remove_file(config.model.join("observation.json")).unwrap();
        let result =
            tokio::time::timeout(Duration::from_secs(5), launch_worker(&config, &pressure))
                .await
                .expect("The CPU probe must exit without a ready frame.");
        let failure = match result {
            Err(error) => error.message(),
            Ok((mut child, _)) => {
                stop_child(&mut child.child).await.unwrap();
                panic!("The import probe must not publish readiness.");
            }
        };
        assert!(
            matches!(
                failure.as_str(),
                "Worker stdout ended." | "The worker output pipe closed."
            ),
            "The probe must fail startup because it emits no ready frame. {failure}"
        );
        let current = probe.observation();
        assert_eq!(current["source_sha256"], digest);
        assert_eq!(
            current["revision"], "apxinf-python-clef-text-v1",
            "Clef must execute current schema source."
        );
        assert_eq!(current["initializer"], "current");
        assert_eq!(current["member"], "current");
        assert_eq!(current["python"], python.to_str().unwrap());
        let argv = current["argv"].as_array().unwrap();
        assert_eq!(argv[0], worker.to_str().unwrap());
        assert_eq!(argv[1], "--clef-loader");
        assert_eq!(
            argv[2],
            probe.0.join("external_loader.py").to_str().unwrap()
        );
        assert_eq!(argv[3], "--protocol");
        assert_eq!(argv[4], contracts::CLEF_PROTOCOL);
        assert_eq!(argv[5], "--worker-epoch");
        uuid::Uuid::parse_str(argv[6].as_str().unwrap()).unwrap();
        assert_eq!(
            &argv[7..],
            &[
                json!("--model"),
                json!(config.model),
                json!("--max-context"),
                json!("4096"),
                json!("--max-output-tokens"),
                json!("0"),
                json!("--prefill-step-size"),
                json!("256"),
                json!("--output-batch-tokens"),
                json!("1"),
                json!("--memory-limit-bytes"),
                json!("4096"),
            ]
        );
        assert_eq!(current["device_imports"], json!([]));
        assert_eq!(source_digest(&schema), digest);
        for (path, bytes) in cache_paths.iter().zip(&cache_bytes) {
            assert_eq!(
                &std::fs::read(path).unwrap(),
                bytes,
                "Existing cache bytes must remain unchanged."
            );
        }
    }
}
