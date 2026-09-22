//! Explicit native checkpoint integration test; run under the GPU allocator:
//! APXINF_QWEN_DRIVE_TEST_MODEL=/path/to/vlm cargo test -p apxinf-model \
//!   --features cuda --test qwen_drive_loading -- --ignored --test-threads=1
#![cfg(all(feature = "cuda", unix))]

use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{qwen_drive::QwenDriveModel, AutoModel, LoadOptions, LoadedModel};

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        // Only this test's create-new directory and symlinks, never the weights.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires CUDA and APXINF_QWEN_DRIVE_TEST_MODEL with a real VLM checkpoint"]
fn text_registry_ignores_adjacent_invalid_planner_but_explicit_loading_rejects_it() {
    let source = PathBuf::from(
        std::env::var_os("APXINF_QWEN_DRIVE_TEST_MODEL")
            .expect("set APXINF_QWEN_DRIVE_TEST_MODEL to the VLM checkpoint"),
    )
    .canonicalize()
    .unwrap();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let fixture = Fixture(std::env::temp_dir().join(format!(
        "apxinf-qwen-drive-loading-{}-{suffix}",
        std::process::id()
    )));
    std::fs::create_dir(&fixture.0).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let text = name.to_string_lossy();
        if text == "config.json"
            || text.ends_with(".safetensors")
            || text.ends_with(".safetensors.index.json")
        {
            std::os::unix::fs::symlink(entry.path(), fixture.0.join(name)).unwrap();
        }
    }
    let planner = fixture.0.join("planner-sft");
    std::fs::create_dir(&planner).unwrap();
    std::fs::write(
        planner.join("model.safetensors"),
        b"invalid planner checkpoint",
    )
    .unwrap();

    let loaded = AutoModel::load_model(Device::Cuda(0), &fixture.0, &LoadOptions::default())
        .expect("text loading must not inspect the adjacent planner");
    assert!(matches!(loaded, LoadedModel::Text { .. }));
    drop(loaded);

    let error = QwenDriveModel::load(&fixture.0, Some(&planner), Device::Cuda(0))
        .err()
        .expect("an explicitly requested invalid planner must fail");
    assert!(
        error.to_string().contains("load planner weights"),
        "unexpected error: {error}"
    );
}
