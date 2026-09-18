#[path = "../build_support/gemm_fingerprint.rs"]
mod gemm_fingerprint;

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
            "apxinf-gemm-fingerprint-{}-{id}",
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
        gemm_fingerprint::build_id(
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

fn populated_fixture() -> Fixture {
    let fixture = Fixture::new();
    fixture.write("adapters/gemm/candidates.cu", "candidate-v1");
    fixture.write("include/apxinf_cuda/gemm.h", "gemm-abi-v1");
    fixture.write("kernels/custom/gemm.cuh", "custom-kernel-v1");
    fixture.write("kernels/cutlass/include/cutlass/cutlass.h", "cutlass-v1");
    fixture.write("adapters/runtime.cu", "unrelated-runtime-v1");
    fixture.write("adapters/attention/kernel.cu", "unrelated-adapter-v1");
    fixture
}

#[test]
fn only_gemm_native_inputs_change_the_build_id() {
    let fixture = populated_fixture();
    let original = fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a");

    fixture.write("adapters/runtime.cu", "unrelated-runtime-v2");
    fixture.write("adapters/attention/kernel.cu", "unrelated-adapter-v2");
    assert_eq!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a")
    );

    fixture.write("adapters/gemm/candidates.cu", "candidate-v2");
    assert_ne!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a")
    );
}

#[test]
fn gemm_abi_kernel_target_and_architectures_change_the_build_id() {
    let fixture = populated_fixture();
    let original = fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a");

    fixture.write("include/apxinf_cuda/gemm.h", "gemm-abi-v2");
    assert_ne!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a")
    );
    fixture.write("include/apxinf_cuda/gemm.h", "gemm-abi-v1");

    fixture.write("kernels/custom/gemm.cuh", "custom-kernel-v2");
    assert_ne!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a")
    );
    fixture.write("kernels/custom/gemm.cuh", "custom-kernel-v1");

    fixture.write("kernels/cutlass/include/cutlass/cutlass.h", "cutlass-v2");
    assert_ne!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_100", "sm_100a")
    );
    fixture.write("kernels/cutlass/include/cutlass/cutlass.h", "cutlass-v1");

    assert_ne!(
        original,
        fixture.fingerprint("aarch64-unknown-linux-gnu", "sm_100", "sm_100a")
    );
    assert_ne!(
        original,
        fixture.fingerprint("x86_64-unknown-linux-gnu", "sm_110", "sm_110a")
    );
}
