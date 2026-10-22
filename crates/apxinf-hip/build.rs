//! Compile the HIP device code when a ROCm toolchain is present.
//!
//! With ROCm: `kernels/apxinf_hip.hip` is compiled by `hipcc` for one GPU
//! architecture, archived, and linked with `amdhip64` and `hipblas`, and the
//! `apxinf_hip_runtime` cfg is set.
//!
//! Without ROCm the crate still builds — macOS and CUDA-only hosts compile the
//! workspace with `--features hip` — but the runtime cfg is absent, the C ABI is
//! replaced by stubs, and `HipBackend::new` reports that the backend was built
//! without ROCm. That fails at the point someone asks for a HIP device, not at
//! link time on a machine that never meant to run HIP.
//!
//! Environment:
//! - `ROCM_PATH` — ROCm root, default `/opt/rocm`.
//! - `APXINF_HIP_ARCH` — target, e.g. `gfx1151`. Defaults to the first GPU
//!   `rocm_agent_enumerator` reports. Required when building on a host whose
//!   GPU differs from the deployment target, or that has no GPU.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(apxinf_hip_runtime)");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=APXINF_HIP_ARCH");
    println!("cargo:rerun-if-changed=kernels/apxinf_hip.hip");

    let rocm = PathBuf::from(env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into()));
    let hipcc = rocm.join("bin/hipcc");
    if !hipcc.is_file() {
        println!(
            "cargo:warning=apxinf-hip: no hipcc at {}; building without the HIP runtime",
            hipcc.display()
        );
        println!("cargo:rustc-env=APXINF_HIP_ARCH=none");
        return;
    }

    let arch = target_arch(&rocm);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let object = out_dir.join("apxinf_hip.o");
    let source = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels/apxinf_hip.hip");

    let output = Command::new(&hipcc)
        .args(["-c", "-O3", "-fPIC", "-std=c++17"])
        .arg(format!("--offload-arch={arch}"))
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output()
        .unwrap_or_else(|e| panic!("apxinf-hip: failed to run {}: {e}", hipcc.display()));
    if !output.status.success() {
        // ROCm's device linker needs libxml2.so.2; distributions that ship a
        // newer libxml2 report it here. LD_LIBRARY_PATH must reach a
        // compatible copy before the build starts.
        panic!(
            "apxinf-hip: hipcc failed for {arch}\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let archive = out_dir.join("libapxinf_hip_kernels.a");
    let _ = std::fs::remove_file(&archive);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("apxinf-hip: failed to run ar");
    assert!(status.success(), "apxinf-hip: ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apxinf_hip_kernels");
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=apxinf_hip_runtime");
    // The runtime refuses a device whose architecture differs from this one:
    // a mismatched code object only fails later, as "invalid device function".
    println!("cargo:rustc-env=APXINF_HIP_ARCH={arch}");
    println!("cargo:warning=apxinf-hip: HIP kernels built for {arch}");
}

fn target_arch(rocm: &Path) -> String {
    if let Ok(arch) = env::var("APXINF_HIP_ARCH") {
        let arch = arch.trim().to_string();
        assert!(
            arch.starts_with("gfx"),
            "apxinf-hip: APXINF_HIP_ARCH must name a gfx target such as gfx1151, got `{arch}`"
        );
        return arch;
    }
    // gfx000 is the CPU agent.
    Command::new(rocm.join("bin/rocm_agent_enumerator"))
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .find(|line| line.starts_with("gfx") && *line != "gfx000")
                .map(str::to_string)
        })
        .unwrap_or_else(|| {
            panic!(
                "apxinf-hip: found hipcc but no GPU to target; set APXINF_HIP_ARCH (e.g. gfx1151)"
            )
        })
}
