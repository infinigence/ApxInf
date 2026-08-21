//! Root-package build script.
//!
//! The CUDA kernel archive (`libapxinf_kernels.a`) is produced by
//! `crates/apxinf-cuda`'s build script. Rustc does not reliably propagate the
//! `cargo:rustc-link-lib=static` directive from a dependency crate through to
//! the final binary link, which left kernel C-ABI symbols undefined. Pin the
//! archive to the final link here, but only when it actually exists, so plain
//! CPU (`cargo build` / `cargo check`) builds keep working.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".to_string());
    let build_dir = manifest.join("target").join(&profile).join("build");

    let Ok(entries) = std::fs::read_dir(&build_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("apxinf-cuda-") {
            continue;
        }
        let out = entry.path().join("out");
        let archive = out.join("libapxinf_kernels.a");
        if archive.is_file() {
            println!("cargo:rustc-link-search=native={}", out.display());
            println!("cargo:rustc-link-lib=static=apxinf_kernels");
            println!("cargo:rustc-link-arg={}", archive.display());
            println!("cargo:rerun-if-changed={}", archive.display());
        }
    }
}

