use std::path::{Path, PathBuf};

const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

// These are the native inputs that can change a GEMM recipe, candidate,
// kernel, or its ABI. In particular, unrelated adapters and runtime sources
// must not invalidate persisted GEMM recipes.
const GEMM_INPUT_TREES: &[&str] = &[
    "adapters/gemm",
    "framework",
    "kernels/cutlass/extensions",
    "kernels/cutlass/include",
    "kernels/cutlass/ops/gemm",
];

const GEMM_INPUT_FILES: &[&str] = &[
    "include/apxinf_cuda/gemm.h",
    "include/apxinf_cuda/gemm_types.h",
    "include/apxinf_cuda/status.h",
    "include/apxinf_cuda/tuning_types.h",
    "include/apxinf_cuda/types.h",
    "kernels/custom/gemm.cuh",
];

fn hash_bytes(hash: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u128::from(*byte);
        *hash = hash.wrapping_mul(FNV1A_128_PRIME);
    }
}

fn collect_tree(root: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_tree(&path, files);
        } else if path.extension().is_some_and(|extension| {
            matches!(
                extension.to_string_lossy().as_ref(),
                "cpp" | "cu" | "cuh" | "h" | "hh" | "hpp"
            )
        }) {
            files.push(path);
        }
    }
}

fn gemm_inputs(native: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for tree in GEMM_INPUT_TREES {
        collect_tree(&native.join(tree), &mut files);
    }
    files.extend(
        GEMM_INPUT_FILES
            .iter()
            .map(|relative| native.join(relative))
            .filter(|path| path.is_file()),
    );
    files.sort_unstable();
    files.dedup();
    files
}

pub fn build_id<'a>(
    native: &Path,
    target: &str,
    architectures: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    let mut hash = FNV1A_128_OFFSET;
    for value in ["apxinf-gemm-pilot-v1", env!("CARGO_PKG_VERSION"), target] {
        hash_bytes(&mut hash, value.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    for (nvcc_arch, cutlass_arch) in architectures {
        hash_bytes(&mut hash, nvcc_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
        hash_bytes(&mut hash, cutlass_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    for path in gemm_inputs(native) {
        hash_bytes(
            &mut hash,
            path.strip_prefix(native)
                .unwrap_or(&path)
                .to_string_lossy()
                .as_bytes(),
        );
        hash_bytes(&mut hash, &[0]);
        hash_bytes(
            &mut hash,
            &std::fs::read(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
        );
        hash_bytes(&mut hash, &[0xff]);
    }
    format!("gemm-kb1-{hash:032x}")
}
