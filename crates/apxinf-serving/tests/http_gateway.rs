//! Real HTTP checks with an original CPU protocol fixture.
use std::{path::PathBuf, process::Command};

#[test]
fn real_gateway_with_cpu_worker() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("python3")
        .arg(root.join("tests/fixtures/serving/http_gateway_checks.py"))
        .arg("--binary")
        .arg(env!("CARGO_BIN_EXE_apxinf-serve"))
        .current_dir(&root)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("Python 3 is required for the HTTP fixture.");
    assert!(
        output.status.success(),
        "HTTP fixture failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    print!("{}", String::from_utf8_lossy(&output.stderr));
}
