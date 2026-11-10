//! Compile the HIP device code when a ROCm toolchain is present.
//!
//! With ROCm: every translation unit in [`SOURCES`] is compiled by `hipcc` for
//! one GPU architecture, the objects are archived, the archive is linked with
//! `amdhip64` and `hipblas`, and the `apxinf_hip_runtime` cfg is set.
//!
//! hipBLASLt is optional. When its header is present the second GEMM provider
//! is compiled in and `hipblaslt` is linked; otherwise the provider's entry
//! points become stubs and `native::gemm` keeps using hipBLAS. That keeps the
//! crate buildable on ROCm installs that predate hipBLASLt.
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

/// Device translation units under `kernels/`. All of them include `common.h`.
const SOURCES: &[&str] = &[
    "apxinf_hip.hip",
    "norm.hip",
    "elementwise.hip",
    "native_attention.hip",
    "rope.hip",
];

/// Flags the vendored Composable Kernel units must be built with.
///
/// Copied from `kernels/ck/VENDOR.md`, which records them as the flags CK's
/// own CMake uses for these kernels and the ones the instances were validated
/// with. They apply to CK's units and to apxinf's wrapper over them, never to
/// the rest of `SOURCES`: `-enable-post-misched=0` in particular is a
/// different code generator, and the kernels in `SOURCES` are tuned and
/// measured without it.
const CK_FLAGS: &[&str] = &[
    "-O3",
    "-fPIC",
    "-std=c++20",
    "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
    "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
    "-mllvm",
    "--lsr-drop-solution=1",
    "-mllvm",
    "-enable-post-misched=0",
    "-fbracket-depth=1024",
    "-DCK_TILE_FMHA_FWD_SPLITKV_API=1",
    "-DCK_TILE_FMHA_FWD_APPENDKV_API=0",
    "-DCK_TILE_FMHA_FWD_PAGEDKV_API=0",
];

fn main() {
    println!("cargo:rustc-check-cfg=cfg(apxinf_hip_runtime)");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=APXINF_HIP_ARCH");
    println!("cargo:rerun-if-changed=kernels/common.h");
    for source in SOURCES {
        println!("cargo:rerun-if-changed=kernels/{source}");
    }
    println!("cargo:rerun-if-changed=kernels/ck_attention.hip");
    println!("cargo:rerun-if-changed=kernels/ck");

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
    let kernels = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");

    // Probed by header, not by ROCm version: a version table would go stale,
    // and a missing header is exactly what breaks the compile.
    let blaslt = rocm.join("include/hipblaslt/hipblaslt.h").is_file();
    if !blaslt {
        println!(
            "cargo:warning=apxinf-hip: no hipblaslt.h under {}; \
             building with hipBLAS as the only GEMM provider",
            rocm.display()
        );
    }

    // Probed before anything is spawned: the always-compiled units need to
    // know whether `ck_attention.hip` will define the CK entry points or they
    // must carry stubs.
    let ck = kernels.join("ck");
    let has_ck = ck.join("generated/fmha_fwd_api.cpp").is_file();

    // Every translation unit is compiled as its own `hipcc` process, and they
    // are spawned together rather than in sequence. CK contributes 20 units
    // that take about five minutes serially; in parallel the whole build is
    // bounded by the slowest one.
    let mut jobs = Vec::new();
    for source in SOURCES {
        let object = out_dir.join(Path::new(source).with_extension("o"));
        let child = Command::new(&hipcc)
            .args(["-c", "-O3", "-fPIC", "-std=c++17"])
            .arg(format!("-DAPXINF_HIP_HAS_BLASLT={}", blaslt as i32))
            .arg(format!("-DAPXINF_HIP_HAS_CK={}", has_ck as i32))
            .arg(format!("--offload-arch={arch}"))
            .arg(kernels.join(source))
            .arg("-o")
            .arg(&object)
            .spawn()
            .unwrap_or_else(|e| panic!("apxinf-hip: failed to run {}: {e}", hipcc.display()));
        jobs.push(((*source).to_string(), object, child));
    }

    // CK's FMHA, when the vendored copy is present. Absent, the entry points
    // are the stubs in `apxinf_hip.hip` and `native::attention` keeps using
    // apxinf's own kernel -- the same shape of fallback the hipBLASLt provider
    // has.
    if has_ck {
        let mut ck_sources: Vec<PathBuf> = std::fs::read_dir(ck.join("generated"))
            .expect("apxinf-hip: cannot read kernels/ck/generated")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension()? == "cpp").then_some(path)
            })
            .collect();
        ck_sources.sort();
        ck_sources.push(kernels.join("ck_attention.hip"));
        for source in ck_sources {
            let object = out_dir.join(format!(
                "ck_{}.o",
                source.file_stem().unwrap().to_string_lossy()
            ));
            let child = Command::new(&hipcc)
                .arg("-c")
                .args(CK_FLAGS)
                .arg(format!("--offload-arch={arch}"))
                .arg("-I")
                .arg(&kernels)
                .arg("-I")
                .arg(ck.join("include"))
                // CK's generated units include "fmha_fwd.hpp" by name, so the
                // example directory holding it has to be on the path too.
                .arg("-I")
                .arg(ck.join("example/ck_tile/01_fmha"))
                .arg(&source)
                .arg("-o")
                .arg(&object)
                .spawn()
                .unwrap_or_else(|e| panic!("apxinf-hip: failed to run {}: {e}", hipcc.display()));
            jobs.push((source.display().to_string(), object, child));
        }
    } else {
        println!(
            "cargo:warning=apxinf-hip: no vendored CK at {}; attention uses apxinf's \
             own kernel only",
            ck.display()
        );
    }

    let mut objects = Vec::with_capacity(jobs.len());
    for (name, object, child) in jobs {
        let output = child
            .wait_with_output()
            .unwrap_or_else(|e| panic!("apxinf-hip: {} failed: {e}", hipcc.display()));
        if !output.status.success() {
            // ROCm's device linker needs libxml2.so.2; distributions that ship a
            // newer libxml2 report it here. LD_LIBRARY_PATH must reach a
            // compatible copy before the build starts.
            panic!(
                "apxinf-hip: hipcc failed on {name} for {arch}\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        objects.push(object);
    }

    let archive = out_dir.join("libapxinf_hip_kernels.a");
    let _ = std::fs::remove_file(&archive);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("apxinf-hip: failed to run ar");
    assert!(status.success(), "apxinf-hip: ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apxinf_hip_kernels");
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    if blaslt {
        println!("cargo:rustc-link-lib=dylib=hipblaslt");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-check-cfg=cfg(apxinf_hip_ck)");
    if has_ck {
        println!("cargo:rustc-cfg=apxinf_hip_ck");
    }
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
