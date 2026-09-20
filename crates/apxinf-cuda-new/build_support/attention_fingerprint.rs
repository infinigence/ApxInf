use std::path::{Path, PathBuf};

const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

const INPUT_TREES: &[&str] = &[
    "adapters/attention",
    "framework",
    "kernels/attention",
    "kernels/fa2",
    "kernels/fa2_compat",
    "kernels/cutlass/fmha",
    "kernels/cutlass/ops/attention",
    "patches",
];
const INPUT_FILES: &[&str] = &[
    "include/apxinf_cuda/attention.h",
    "include/apxinf_cuda/attention_types.h",
    "include/apxinf_cuda/status.h",
    "include/apxinf_cuda/tuning_types.h",
    "include/apxinf_cuda/types.h",
    "kernels/custom/attention.cuh",
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
                "cpp" | "cu" | "cuh" | "h" | "hh" | "hpp" | "patch"
            )
        }) {
            files.push(path);
        }
    }
}

pub fn build_id<'a>(
    native: &Path,
    target: &str,
    architectures: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    let mut hash = FNV1A_128_OFFSET;
    for value in ["apxinf-attention-v1", env!("CARGO_PKG_VERSION"), target] {
        hash_bytes(&mut hash, value.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    for (nvcc_arch, cutlass_arch) in architectures {
        hash_bytes(&mut hash, nvcc_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
        hash_bytes(&mut hash, cutlass_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    let mut files = Vec::new();
    for tree in INPUT_TREES {
        collect_tree(&native.join(tree), &mut files);
    }
    files.extend(
        INPUT_FILES
            .iter()
            .map(|relative| native.join(relative))
            .filter(|path| path.is_file()),
    );
    files.sort_unstable();
    files.dedup();
    for path in files {
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
    format!("attention-kb1-{hash:032x}")
}
