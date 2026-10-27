//! Compile the HIP device code when a ROCm toolchain is present.
//!
//! With ROCm: every translation unit in [`SOURCES`], plus the Composable Kernel
//! attention units (`ck_attention.hip` and `kernels/ck/generated/`, see
//! `kernels/ck/VENDOR.md`), is compiled by `hipcc` for one GPU architecture,
//! the objects are archived, the archive is linked with `amdhip64` and
//! `hipblas`, and the `apxinf_hip_runtime` cfg is set. Units compile in
//! parallel, and a unit whose object is newer than everything it depends on is
//! not recompiled: the CK units are slow and their inputs rarely change.
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
use std::time::SystemTime;

/// Device translation units under `kernels/`. All of them include `common.h`.
const SOURCES: &[&str] = &[
    "apxinf_hip.hip",
    "norm.hip",
    "elementwise.hip",
    "native_attention.hip",
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

    let ck = kernels.join("ck");
    let ck_include = ck.join("include");
    let ck_fmha = ck.join("example/ck_tile/01_fmha");
    let ck_generated = ck.join("generated");

    let common = newest(&[kernels.join("common.h")]);
    let mut units: Vec<Unit> = SOURCES
        .iter()
        .map(|source| Unit {
            source: kernels.join(source),
            object: out_dir.join(Path::new(source).with_extension("o")),
            ck: false,
            inputs_changed: common,
        })
        .collect();
    // CK units depend on the whole vendored tree, which only changes on a CK bump.
    let ck_inputs = newest_in(&ck).max(common);
    let mut generated: Vec<PathBuf> = std::fs::read_dir(&ck_generated)
        .unwrap_or_else(|e| panic!("apxinf-hip: read {}: {e}", ck_generated.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cpp"))
        .collect();
    generated.sort();
    for source in std::iter::once(kernels.join("ck_attention.hip")).chain(generated) {
        let name = source.file_stem().unwrap().to_string_lossy().into_owned();
        units.push(Unit {
            object: out_dir.join(format!("ck_{name}.o")),
            source,
            ck: true,
            inputs_changed: ck_inputs,
        });
    }

    let compile = |unit: &Unit| {
        if is_fresh(&unit.object, std::slice::from_ref(&unit.source), unit.inputs_changed) {
            return;
        }
        let mut command = Command::new(&hipcc);
        command
            .args(["-c", "-O3", "-fPIC"])
            .arg(format!("--offload-arch={arch}"));
        if unit.ck {
            // CK's own CMake flags for its FMHA forward kernels (ROCm >= 6.2);
            // recorded in kernels/ck/VENDOR.md.
            command
                .args(["-std=c++20", "-DCK_TILE_FMHA_FWD_FAST_EXP2=1"])
                .args([
                    "-fgpu-flush-denormals-to-zero",
                    "-fno-offload-uniform-block",
                ])
                .args([
                    "-mllvm",
                    "--lsr-drop-solution=1",
                    "-mllvm",
                    "-enable-post-misched=0",
                ])
                .arg("-fbracket-depth=1024")
                .args([
                    "-DCK_TILE_FMHA_FWD_SPLITKV_API=1",
                    "-DCK_TILE_FMHA_FWD_APPENDKV_API=0",
                ])
                .arg("-DCK_TILE_FMHA_FWD_PAGEDKV_API=0")
                .args(["-Wno-undefined-func-template", "-Wno-float-equal"])
                .arg("-I")
                .arg(&ck_include)
                .arg("-I")
                .arg(&ck_fmha)
                .arg("-I")
                .arg(&kernels);
        } else {
            command.arg("-std=c++17");
        }
        let output = command
            .arg(&unit.source)
            .arg("-o")
            .arg(&unit.object)
            .output()
            .unwrap_or_else(|e| panic!("apxinf-hip: failed to run {}: {e}", hipcc.display()));
        if !output.status.success() {
            // ROCm's device linker needs libxml2.so.2; distributions that ship a
            // newer libxml2 report it here. LD_LIBRARY_PATH must reach a
            // compatible copy before the build starts.
            panic!(
                "apxinf-hip: hipcc failed on {} for {arch}\n{}{}",
                unit.source.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    };
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(units.len()) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                match units.get(index) {
                    Some(unit) => compile(unit),
                    None => break,
                }
            });
        }
    });
    let objects: Vec<&PathBuf> = units.iter().map(|unit| &unit.object).collect();

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
    println!(
        "cargo:rustc-link-search=native={}",
        rocm.join("lib").display()
    );
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

struct Unit {
    source: PathBuf,
    object: PathBuf,
    /// Compiled with Composable Kernel's include paths and flags.
    ck: bool,
    /// Newest modification time among the unit's headers.
    inputs_changed: SystemTime,
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn newest(paths: &[PathBuf]) -> SystemTime {
    paths
        .iter()
        .filter_map(|p| modified(p))
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Newest modification time of any file under `dir`.
fn newest_in(dir: &Path) -> SystemTime {
    let mut newest = SystemTime::UNIX_EPOCH;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Some(time) = modified(&path) {
                newest = newest.max(time);
            }
        }
    }
    newest
}

/// An object is fresh when it is newer than its sources, its headers and this
/// build script (whose flags it was compiled with).
fn is_fresh(object: &Path, sources: &[PathBuf], inputs: SystemTime) -> bool {
    let Some(built) = modified(object) else {
        return false;
    };
    let script = modified(&Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("build.rs"));
    built > newest(sources) && built > inputs && script.is_none_or(|time| built > time)
}
