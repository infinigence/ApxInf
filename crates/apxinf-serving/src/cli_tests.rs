use super::*;

#[test]
fn vision_provider_requires_an_explicit_interpreter() {
    let args = Args::try_parse_from([
        "apxinf-serve",
        "--model",
        "/model",
        "--provider",
        "mlx-vlm-minicpmv47",
    ])
    .unwrap();
    assert!(args.python.is_none());
    let error = worker_paths(
        &WorkerRuntime::PythonMiniCpmVision,
        args.python,
        args.worker,
    )
    .unwrap_err();
    assert!(error.to_string().contains("explicit --python"));
    let args = Args::try_parse_from(["apxinf-serve", "--model", "/model"]).unwrap();
    assert_eq!(args.provider, "python-mlx-lm");
    let native = WorkerRuntime::NativeMiniCpm5 {
        executable: "/worker".into(),
        metal_library: "/metal".into(),
    };
    let (python, worker) = worker_paths(&native, None, None).unwrap();
    assert_eq!(
        python,
        PathBuf::from(".apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python")
    );
    assert_eq!(
        worker,
        PathBuf::from("python/apxinf/apxinf/serving/text_worker.py")
    );
}

#[test]
#[cfg(unix)]
fn vision_launch_preserves_the_final_virtual_environment_symlink() {
    let root = std::env::temp_dir().join(format!("apxinf-vision-cli-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(root.join("bin")).unwrap();
    let real = root.join("interpreter");
    std::fs::write(&real, b"interpreter fixture").unwrap();
    let link = root.join("bin/python");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let worker = root.join("worker.py");
    std::fs::write(&worker, b"worker fixture").unwrap();
    let (selected, _) = worker_paths(
        &WorkerRuntime::PythonMiniCpmVision,
        Some(link.clone()),
        Some(worker.clone()),
    )
    .unwrap();
    assert_eq!(selected.file_name().unwrap(), "python");
    assert_eq!(selected.parent().unwrap().file_name().unwrap(), "bin");
    assert!(selected
        .symlink_metadata()
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::canonicalize(selected).unwrap(),
        std::fs::canonicalize(&real).unwrap()
    );
    let (old, _) = worker_paths(&WorkerRuntime::Python, Some(link), Some(worker)).unwrap();
    assert_eq!(old, std::fs::canonicalize(real).unwrap());
    std::fs::remove_dir_all(root).unwrap();
}
