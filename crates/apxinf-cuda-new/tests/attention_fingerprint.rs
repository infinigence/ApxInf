#[path = "../build_support/attention_fingerprint.rs"]
mod attention_fingerprint;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apxinf-attention-fingerprint-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn write(&self, relative: &str, contents: &str) {
        let path = self.root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn fingerprint(&self, target: &str, nvcc_arch: &str, cutlass_arch: &str) -> String {
        attention_fingerprint::build_id(
            &self.root,
            target,
            std::iter::once((nvcc_arch, cutlass_arch)),
        )
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn attention_provider_vendor_compat_and_cutlass_inputs_invalidate_recipes() {
    let fixture = Fixture::new();
    let inputs = [
        "adapters/attention/providers/efficient.cu",
        "kernels/efficient_attention/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h",
        "kernels/efficient_compat/ATen/cuda/PhiloxUtils.cuh",
        "kernels/cutlass/include/cutlass/gemm/threadblock/default_mma.h",
    ];
    for path in inputs {
        fixture.write(path, "v1");
    }
    let original = fixture.fingerprint("aarch64-unknown-linux-gnu", "sm_110", "sm_110a");
    for path in inputs {
        fixture.write(path, "v2");
        assert_ne!(
            original,
            fixture.fingerprint("aarch64-unknown-linux-gnu", "sm_110", "sm_110a"),
            "{path}"
        );
        fixture.write(path, "v1");
    }
    fixture.write(
        "adapters/gemm/providers/unrelated.cu",
        "not-an-attention-input",
    );
    assert_eq!(
        original,
        fixture.fingerprint("aarch64-unknown-linux-gnu", "sm_110", "sm_110a")
    );
}
