//! Build script for the MUSA backend.
//!
//! Mirrors `crates/apxinf-cuda/build.rs` with the CUDA toolchain replaced by
//! MUSA's: `MUSA_PATH` instead of `CUDA_PATH`, `mcc` instead of `nvcc`,
//! `--offload-arch=mp_<cc>` instead of `-arch=sm_<cc>`, and musart / mublas /
//! mublasLt instead of cudart / cublas / cublasLt.
//!
//! Like the CUDA build, a missing toolkit is a soft failure: the crate still
//! compiles so `cargo build --workspace` works on machines without MUSA, and
//! the backend reports "not available" at runtime instead of failing the build.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build_support/mcc.rs"]
mod mcc;

#[path = "build_support/musa_arch.rs"]
mod musa_arch;

use musa_arch::{ASSEMBLY_ARCH, VENDOR_FMHA_ARCH};

#[path = "build_support/host_math_shim.rs"]
mod host_math_shim;

#[path = "build_support/stream_device.rs"]
mod stream_device;

#[path = "build_support/mttx.rs"]
mod mttx;

/// FNV-1a over the kernel sources, so a MUSA recipe cache can never be
/// mistaken for a CUDA one. The prefix is what keeps the two apart.
const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

/// A tree of kernel inputs: how deep it is hashed, what `mcc` compiles out of
/// it, and what that compilation needs.
///
/// A tree is hashed *and* compiled from the same entry, so a source cannot be
/// compiled into the archive while staying out of the build id that names it.
struct KernelTree {
    /// Relative to the crate root.
    path: &'static str,
    /// Whether the hash and the rerun watch descend into subdirectories.
    recursive: bool,
    /// Extensions `mcc` compiles from this tree. Empty means the tree is hashed
    /// but not compiled — `kernels/asm` is assembled by `llvm-mc` instead.
    compiled: &'static [&'static str],
    /// Extra `-I` roots, relative to the crate root.
    includes: &'static [&'static str],
    /// Whether these translation units take the vendor's device flags.
    vendor_flags: bool,
    /// Files in this tree that are compiled only alongside the vendored FMHA.
    ///
    /// `mutlass_fa_adapter.mu` is the whole of that list: it exists to drive
    /// the frozen kernels, so on a build without them it would be an object
    /// whose only content is a reference to a symbol nothing defines. Leaving
    /// it in would work — a static archive is only searched for what is
    /// referenced — but the backend's own rule is that a target carries the
    /// kernels it can run and no others.
    only_with_vendor_fmha: &'static [&'static str],
}

/// Kernel sources and headers, in the CUDA crate's layout.
const KERNEL_INPUT_TREES: &[KernelTree] = &[
    KernelTree {
        path: "adapters",
        recursive: false,
        compiled: &["mu"],
        includes: &["kernels/mutlass_fmha/tools"],
        vendor_flags: false,
        only_with_vendor_fmha: &["mutlass_fa_adapter.mu"],
    },
    KernelTree {
        path: "kernels/custom",
        recursive: false,
        // **Nothing here is a `.mu` and nothing should be.** This tree holds
        // device code as `.cuh`, pulled into `adapters/custom_kernels.mu` by
        // `#include`; a `.mu` dropped in here would be compiled by `mcc` as its
        // own translation unit, included by no adapter, and archived as kernels
        // nothing can reach — while the build id stays correct, so nothing would
        // say so. `compiled` lists what this tree is expected to contain, and
        // the `kernels/asm` entry below it says "not compiled" the same way.
        compiled: &[],
        includes: &[],
        vendor_flags: false,
        only_with_vendor_fmha: &[],
    },
    // The MP22 attention assembly, which is not compiled by `mcc` but still has
    // to be in the hash: its bytes decide what the backend runs, so a change to
    // it must produce a different kernel build id.
    KernelTree {
        path: "kernels/asm",
        recursive: false,
        compiled: &[],
        includes: &[],
        vendor_flags: false,
        only_with_vendor_fmha: &[],
    },
    // The vendored MATE/MUTLASS FMHA. Recursive, because the headers it
    // includes live four levels down and a frozen `.mu` that changed while the
    // build id did not would be indistinguishable from a current one.
    KernelTree {
        path: "kernels/mutlass_fmha",
        recursive: true,
        compiled: &["mu"],
        includes: &[
            "kernels/mutlass_fmha/mate/data/include",
            "kernels/mutlass_fmha/mate/data/mutlass/include",
            "kernels/mutlass_fmha/tools",
        ],
        vendor_flags: true,
        only_with_vendor_fmha: &[],
    },
];

const KERNEL_INPUT_FILES: &[&str] = &["src/ffi/custom.rs", "src/ffi/mutlass_fa.rs"];

/// Inputs that must trigger a rebuild but must not enter the build id.
///
/// `build_support/host_math_shim/bits/math-vector.h` is what `mcc` includes in
/// place of the host `<bits/math-vector.h>` when the toolchain cannot compile
/// MUSA code without it. Its bytes decide whether that fallback works, not what
/// the kernel archive contains, so it is watched rather than hashed: hashing it
/// would change `musa-kb2-*` and detach every measurement keyed on that id.
///
/// The Rust sources beside it need no entry — build.rs compiles them through
/// `#[path]`, and Cargo rebuilds the script (and so re-runs it) from the
/// script's own dep-info.
const WATCH_ONLY_FILES: &[&str] = &["build_support/host_math_shim/bits/math-vector.h"];

/// The oldest toolkit the vendored FMHA is known to apply to, and why this is a
/// floor rather than a range.
///
/// MATE renders the kernel from a template whose device flags change below mcc
/// 5.2.0, and the frozen render was produced and measured on 5.2.0. Above that
/// the flags have not changed, so a newer toolkit is treated as compatible.
///
/// That asymmetry is deliberate. A ceiling would turn the next toolkit upgrade
/// into the loss of mp_31's only attention, punishing "not yet measured" as if
/// it were "measured wrong" — and the failure mode on the far side of a floor
/// is a missing operator, which reports itself by name, not a wrong number.
const VENDOR_FMHA_MCC_MIN: (u32, u32) = (5, 2);

/// MATE's own device flags for the vendored kernel, in MATE's order, from the
/// `cuda_cflags` of the `build.ninja` MATE wrote when it first compiled it.
///
/// These are **not** this repo's flags, and the difference is deliberate.
/// `-Od3` is device `-O3` and `-O2` host, so the repo's `-O3` is not added on
/// top; and `-fno-signed-zeros`, `-fmusa-flush-denormals-to-zero` and the two
/// `-mllvm` switches are device-codegen settings whose effect this repo has
/// measured on other kernels. Applying them to the whole kernel set would
/// invalidate every recorded replay result at once, so they are scoped to the
/// trees that were compiled with them.
const VENDOR_FMHA_FLAGS: &[&str] = &[
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
];

/// The MP22 attention and BF16 GEMM images, and their embedded C symbols.
///
/// Each is assembled and linked on its own: the three export the same kernel
/// symbol `mate_fa_d256_mp22`, so which one runs is decided by which image the
/// driver loaded, and they cannot share a translation unit.
const MUBIN_VARIANTS: &[(&str, &str)] = &[
    ("fa_d256_mp22", "kApxinfMusaFaD256Mp22Image"),
    ("fa_d256_mp22_decode", "kApxinfMusaFaD256Mp22DecodeImage"),
    ("fa_d256_mp22_splitkv", "kApxinfMusaFaD256Mp22SplitkvImage"),
    ("asm_bf16bf16gemm_nt_tc_128_128x128_epilogue", "kGemm0"),
    ("asm_bf16bf16gemm_nt_tc_128_128x64_epilogue", "kGemm1"),
    ("asm_bf16bf16gemm_nt_tc_128_32x128_epilogue", "kGemm2"),
    ("asm_bf16bf16gemm_nt_tc_256_256x128_epilogue", "kGemm3"),
    ("asm_bf16bf16gemm_nt_tc_256_256x64_epilogue", "kGemm4"),
    ("asm_bf16bf16gemm_nt_tc_512_256x256_epilogue", "kGemm5"),
    ("asm_bf16bf16gemm_nn_tc_128_128x128_epilogue", "kGemm6"),
    ("asm_bf16bf16gemm_nn_tc_128_128x64_epilogue", "kGemm7"),
    ("asm_bf16bf16gemm_nn_tc_128_32x128_epilogue", "kGemm8"),
    ("asm_bf16bf16gemm_nn_tc_256_256x128_epilogue", "kGemm9"),
    ("asm_bf16bf16gemm_nn_tc_256_256x64_epilogue", "kGemm10"),
    ("asm_bf16bf16gemm_nn_tc_512_256x256_epilogue", "kGemm11"),
    ("asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_wide_mfast", "kGemm12"),
    ("asm_bf16bf16gemm_nn_tc_256_128x256_epilogue_wide_mfast", "kGemm13"),
    ("asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_normal_mfast", "kGemm14"),
    ("asm_bf16bf16gemm_nn_tc_256_256x128_epilogue_wide_nfast", "kGemm15"),
];

/// The MP22 INT8 (w8a8) GEMM tiles, and the C symbol each becomes.
///
/// Rendered from `kernels/qint8_mma_tiles/templates/` in the NN layout with
/// `io_dtype=int8` and a local template patch that gives the epilogue a BF16
/// output (the upstream `output_dtype=half` configuration packed FP16), and
/// checked in beside the BF16 tiles. These are the muDNN QY2 `s8mma`
/// convolution templates configured as a 1x1 GEMM, so they consume `A[m,k]`
/// (INT8), `B[n,k]` (INT8, **output-major**) and per-row and per-column FP16
/// scales, and write `D[m,n]` in **BF16** -- the contract's output, no
/// widening pass. **The order is load-bearing**:
/// `adapters/qint8_gemm_adapter.mu`'s `kTiles` and
/// `src/kernels/gemm/qint8_tiles.rs`'s `QINT8_TILES` are indexed by the same
/// position.
const QINT8_ASM_VARIANTS: &[(&str, &str)] = &[
    ("qint8_256x128", "kApxinfMusaQint8Gemm256x128Image"),
    ("qint8_256x256", "kApxinfMusaQint8Gemm256x256Image"),
    ("qint8_128x128", "kApxinfMusaQint8Gemm128x128Image"),
    ("qint8_256x64", "kApxinfMusaQint8Gemm256x64Image"),
    ("qint8_32x128", "kApxinfMusaQint8Gemm32x128Image"),
    ("qint8_32x64", "kApxinfMusaQint8Gemm32x64Image"),
];

// Separate from the convolution tile ABI: these use the BF16 GEMM ABI with
// INT8 inputs, FP16 scales and a fused BF16 dequantization epilogue.
const QINT8_GEMM_ASM_VARIANTS: &[(&str, &str)] = &[
    ("qint8_gemm_nt_32x128", "kQint8GemmNt"),
    ("qint8_gemm_nn_32x128", "kQint8GemmNn"),
];

fn hash_bytes(hash: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u128::from(*byte);
        *hash = hash.wrapping_mul(FNV1A_128_PRIME);
    }
}

/// Extensions that cannot reach code generation, so a change to one cannot
/// change what the backend runs.
///
/// This is a deny-list, and deliberately so. An allow-list would drop any file
/// whose extension nobody had listed, and that failure is the silent one: the
/// kernel changed, the id did not, and every run after the edit measures the
/// object the old id names. A deny-list fails the other way — a documentation
/// edit retires a build it could not have changed — which costs a rebuild and
/// nothing else.
const NON_CODE_EXTENSIONS: &[&str] = &[
    "md", "rst", "txt", "json", "toml", "ini", "cfg", "yaml", "yml", "csv", "log", "patch",
    "diff", "py", "sh", "bash", "png", "jpg", "jpeg", "svg", "pdf",
];

/// Whether a file can reach code generation, and so whether its bytes belong in
/// the kernel build id.
///
/// The rule is not "everything in the tree": a citation correction in
/// `kernels/asm/README.md` would otherwise rewrite the kernel build id —
/// retiring the machine's tactic database and moving the last digits of the
/// recorded cosine — for a file the assembler never reads. A reader of a build
/// id is asking what device code this build produced; documentation is not part
/// of that answer.
///
/// An unrecognised extension stays in the hash. "Not classified" must not read
/// as "not code".
fn reaches_codegen(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if name.starts_with("LICENSE") || name.starts_with("NOTICE") || name.starts_with("README") {
        return false;
    }
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) => {
            let extension = extension.to_ascii_lowercase();
            !NON_CODE_EXTENSIONS.contains(&extension.as_str())
        }
        None => true,
    }
}

/// Collect every code-bearing file in `directory`, at any depth.
///
/// The vendored FMHA tree is four levels deep, and `read_dir` alone sees only
/// its top. A header or a frozen `.mu` that changed without the kernel build id
/// changing is the failure this avoids: the archive would keep the object the
/// old id named, and every run after the edit would be measuring it.
fn collect_tree(directory: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_tree(&path, files);
        } else if path.is_file() && reaches_codegen(&path) {
            files.push(path);
        }
    }
}

fn collect_kernel_inputs(root: &Path, files: &mut Vec<PathBuf>) {
    for tree in KERNEL_INPUT_TREES {
        let directory = root.join(tree.path);
        if tree.recursive {
            collect_tree(&directory, files);
        } else {
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && reaches_codegen(&path) {
                    files.push(path);
                }
            }
        }
    }
    for file in KERNEL_INPUT_FILES {
        let path = root.join(file);
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
}

/// Bumped whenever the composition of the kernel build id changes, so an id
/// computed under old rules cannot be mistaken for one computed under these.
///
/// This crate has already shipped ids that hashed absolute paths, ignored the
/// arch, and counted files that were not in the repository — three ways for two
/// different kernels to share an id. `v3` narrows the rule a fourth time, to
/// code-bearing files only (`reaches_codegen`); the visible `musa-kb2-` prefix
/// moves with it so an id says which rule produced it without anyone having to
/// reason about the hash.
const KERNEL_BUILD_ID_DOMAIN: &str = "apxinf-musa-kernel-build-v3";

fn computed_kernel_build_id(
    root: &Path,
    version: &str,
    arch: Option<&str>,
    vendor_fmha: bool,
) -> String {
    let mut files = Vec::new();
    collect_kernel_inputs(root, &mut files);
    let mut hash = FNV1A_128_OFFSET;
    hash_bytes(&mut hash, KERNEL_BUILD_ID_DOMAIN.as_bytes());
    hash_bytes(&mut hash, version.as_bytes());
    // The id names a binary, and two archs are two binaries: the same sources
    // compiled for mp_22 and mp_31 produce different device code, so they must
    // not produce the same name for it.
    hash_bytes(&mut hash, arch.unwrap_or("no-toolkit").as_bytes());
    // Likewise the vendor decision: whether the frozen FMHA is in the build is
    // not visible in the sources, because it is a gate on the target.
    hash_bytes(
        &mut hash,
        if vendor_fmha {
            b"vendor-fmha"
        } else {
            b"no-vendor-fmha"
        },
    );
    for path in &files {
        // Relative to the crate, so the same commit in a different directory is
        // the same build. CUDA's build script hashes relative paths for the same
        // reason; this one used to hash the absolute path a checkout happened to
        // have, which made the id a property of where the worktree sat.
        let relative = path.strip_prefix(root).unwrap_or(path);
        hash_bytes(&mut hash, relative.to_string_lossy().as_bytes());
        if let Ok(contents) = std::fs::read(path) {
            hash_bytes(&mut hash, &contents);
        }
    }
    format!("musa-kb2-{hash:032x}")
}

/// Watch every input, explicitly.
///
/// Cargo stops watching the package as soon as a build script emits any
/// `rerun-if-changed`, and directory watching is not documented as recursive.
/// So a recursive tree is named directory by directory and file by file rather
/// than left to a rule that may not hold: the vendored FMHA is four levels deep
/// and an edit to a header in it has to rebuild.
fn emit_rerun_if_changed_tree(root: &Path) {
    fn watch(directory: &Path) {
        println!("cargo:rerun-if-changed={}", directory.display());
        let Ok(entries) = std::fs::read_dir(directory) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                watch(&path);
            } else if path.is_file() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    for tree in KERNEL_INPUT_TREES {
        let directory = root.join(tree.path);
        if tree.recursive {
            watch(&directory);
        } else {
            println!("cargo:rerun-if-changed={}", directory.display());
        }
    }
    for file in KERNEL_INPUT_FILES {
        println!("cargo:rerun-if-changed={}", root.join(file).display());
    }
    for file in WATCH_ONLY_FILES {
        println!("cargo:rerun-if-changed={}", root.join(file).display());
    }
}

/// `link_dir` candidates, most specific first.
fn musa_library_directories(musa_path: &Path) -> Vec<PathBuf> {
    ["lib", "lib64"]
        .iter()
        .map(|name| musa_path.join(name))
        .filter(|path| path.is_dir())
        .collect()
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    for variable in [
        "APXINF_MUSA_ARCH",
        "APXINF_KERNEL_BUILD_ID",
        "MUSA_PATH",
        "MUSA_HOME",
        "APXINF_MTTX_SRC",
        "MTTX_SRC",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    println!("cargo:rustc-check-cfg=cfg(apxinf_musa_available)");
    println!("cargo:rustc-check-cfg=cfg(mttx_available)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_musa_mutlass_fmha)");
    emit_rerun_if_changed_tree(&manifest_dir);

    let musa_path = std::env::var("MUSA_PATH")
        .or_else(|_| std::env::var("MUSA_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/usr/local/musa"));

    let include_dir = musa_path.join("include");
    let library_directories = musa_library_directories(&musa_path);
    let mcc = musa_path.join("bin/mcc");

    let toolkit_found =
        include_dir.is_dir() && !library_directories.is_empty() && mcc.is_file();

    // The arch and the vendor decision are resolved before the build id, because
    // both are inputs to it. `select_musa_arch` only needs the compiler, so it is
    // safe to ask even when the rest of the toolkit is missing — and when it is
    // missing there is no arch to name, which the id records as such.
    let choice = if toolkit_found {
        // The shim's path is fixed, so the arch probe can use it before anything
        // has decided whether this toolchain needs it — which is what lets that
        // probe test the arch name rather than this host's glibc.
        let host_math_shim = mcc::shim(&manifest_dir);
        let explicit_arch = std::env::var("APXINF_MUSA_ARCH").ok();
        // `APXINF_MUSA_ARCH=` is "unset" spelled by a shell or a CI template,
        // not a request to build for the empty target. It is named rather than
        // dropped in silence: the value *was* set, and the reader is entitled to
        // know it did nothing.
        let explicit_arch = match explicit_arch {
            Some(value) if value.trim().is_empty() => {
                println!(
                    "cargo:warning=APXINF_MUSA_ARCH is set but empty; detecting the visible \
                     device instead. Unset the variable to say the same thing quietly."
                );
                None
            }
            other => other,
        };
        // A value that is neither blank nor a name is a mistake this build can
        // name. Falling through to detection would compile for a generation
        // nobody asked for, and the mismatch would only surface at run time, in
        // `MusaContext::new`, a long way from the setting that caused it.
        Some(
            musa_arch::select_musa_arch(explicit_arch.as_deref(), &mcc, host_math_shim.as_deref())
                .unwrap_or_else(|error| {
                    panic!(
                        "MUSA architecture selection failed: {error}\n\
                         Refusing to guess an offload target — see doc/musa-backend.md."
                    )
                }),
        )
    } else {
        None
    };
    let arch = choice.as_ref().map(|choice| choice.target());
    let mutlass_fmha = arch
        .as_ref()
        .is_some_and(|arch| vendor_fmha_decision(arch, &musa_path).is_ok());

    // `lib.rs` reads both of these with `env!`, so they are emitted on every
    // path — including the ones that give up on the toolkit.
    let version = musa_arch::musa_version(&musa_path).unwrap_or_else(|| "unknown".to_string());
    let build_id = std::env::var("APXINF_KERNEL_BUILD_ID").unwrap_or_else(|_| {
        computed_kernel_build_id(
            &manifest_dir,
            &version,
            arch.as_ref().map(|arch| arch.mcc_arch.as_str()),
            mutlass_fmha,
        )
    });
    println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={build_id}");

    // The architecture strings the Rust side compares against, so the constants
    // that decide what this binary launches have one definition shared by the
    // build script and the library rather than a copy in each.
    println!("cargo:rustc-env=APXINF_MUSA_ASSEMBLY_ARCH={ASSEMBLY_ARCH}");
    println!("cargo:rustc-env=APXINF_MUSA_VENDOR_FMHA_ARCH={VENDOR_FMHA_ARCH}");

    // The routing decision itself, not the inputs to it. `kernels/attention.rs`
    // reads this instead of re-deriving the rule from the two constants above —
    // the same answer, published once, so there is one rule rather than two
    // readings of one.
    let route = arch
        .as_ref()
        .map_or(musa_arch::D256Route::Unsupported, |arch| {
            musa_arch::d256_route(&arch.mcc_arch, mutlass_fmha)
        });
    println!("cargo:rustc-env=APXINF_MUSA_D256_ROUTE={}", route.as_str());

    let Some(arch) = arch else {
        println!(
            "cargo:warning=MUSA not found under {} — building without GPU support",
            musa_path.display()
        );
        println!("cargo:rustc-env=APXINF_MUSA_ARCH=unknown");
        return;
    };

    if let Some(warning) = choice.as_ref().and_then(|choice| choice.warning()) {
        println!("cargo:warning={warning}");
    }
    println!(
        "cargo:warning=ApxInf MUSA architecture: {} ({})",
        arch.mcc_arch, arch.source
    );

    println!("cargo:rustc-env=APXINF_MUSA_ARCH={}", arch.mcc_arch);
    println!("cargo:rustc-cfg=apxinf_musa_available");

    for directory in &library_directories {
        println!("cargo:rustc-link-search=native={}", directory.display());
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mutlass_fmha = vendor_fmha_available(&arch, &musa_path);
    emit_mubin_images(&musa_path, &arch, &manifest_dir, &out_dir);
    compile_kernels(
        &mcc,
        &include_dir,
        &arch,
        &manifest_dir,
        &out_dir,
        mutlass_fmha,
    );

    // Keep the vendor math DSOs *after* the static archive. GNU ld's
    // --as-needed drops a shared library that nothing references at the point
    // it is scanned, and the archive's objects are what reference mublas.
    // mublasLt is here for `ffi::musa::mublasLtGetVersion` -- the only version
    // query the vendor exposes -- which Rust calls directly rather than through
    // the archive. Same ordering, same reason, as crates/apxinf-cuda/build.rs.
    //
    // The order within the group matters too. `mublasLt` before `mublas`,
    // because the former is built on the latter; and `musa` last of the three,
    // because `libmublasLt.so` and `libmublas.so` both leave the driver's
    // `muTensorDescriptorEncode` undefined and neither lists a NEEDED entry for
    // it -- it lives in the driver runtime `libmusa.so.1`, whose name appears
    // in neither DSO. Its absence reports a bare undefined-reference with no
    // hint at which library is missing.
    println!("cargo:rustc-link-lib=mublasLt");
    println!("cargo:rustc-link-lib=mublas");
    // muDNN's C++ API (`musa::dnn`, which `adapters/mudnn_adapter.mu` drives)
    // sits in its own `libmudnncxx.so` on 5.1 and inside `libmudnn.so` on
    // 4.3.5. Asking for a library that is not there is a link error, so link
    // the separate DSO only when this toolkit ships it.
    if library_directories
        .iter()
        .any(|directory| directory.join("libmudnncxx.so").is_file())
    {
        println!("cargo:rustc-link-lib=mudnncxx");
    }
    println!("cargo:rustc-link-lib=mudnn");
    println!("cargo:rustc-link-lib=musart");
    println!("cargo:rustc-link-lib=musa");
    println!("cargo:rustc-link-lib=stdc++");

    build_mttx_client(&musa_path, &manifest_dir, &out_dir);
}

/// Compile the MTTX marker client, when the `mttx` feature asks for it.
///
/// Runs after the toolkit check above rather than before it: without a MUSA
/// toolkit the backend is off and there is no host for a marker, so compiling
/// one would only produce an object nothing can reach.
///
/// The client needs the toolkit for nothing, but it does need a
/// `mt-tools-extension` checkout, and the two ways that can go wrong are not
/// the same. No checkout is a build without markers — the crate's no-op stubs
/// take over and the build says what it lost. A checkout that does not compile
/// is this crate's own source being wrong, and since a marker API that cannot
/// work reports nothing at run time, that one stops the build.
fn build_mttx_client(musa_path: &Path, manifest_dir: &Path, out_dir: &Path) {
    if std::env::var_os("CARGO_FEATURE_MTTX").is_none() {
        return;
    }
    let include_dir = match mttx::resolve(musa_path) {
        Ok(include_dir) => include_dir,
        Err(reason) => {
            println!("cargo:warning={reason}");
            println!("cargo:warning=ApxInf MUSA: building without MTTX range markers");
            return;
        }
    };
    match mttx::build(&include_dir, manifest_dir, out_dir) {
        Ok(()) => {
            println!(
                "cargo:warning=ApxInf MUSA: MTTX range markers from {}",
                include_dir.display()
            );
            println!("cargo:rustc-cfg=mttx_available");
        }
        Err(error) => panic!("cannot build the MTTX marker client: {error}"),
    }
}

/// Whether this build compiles the vendored mp_31 FMHA.
///
/// Both halves of the condition are about what the frozen render was validated
/// against, not about what would be convenient: the `.mu` files are mp_31
/// device code, and the toolkit's device flags change below the version MATE
/// produced them with. A build that fails either does not get a wrong answer —
/// `kernels::attention` reports the operator as a kernel gap by name — so the
/// gate costs a missing operator rather than a miscompile.
///
/// Emits the cfg the Rust side gates on, because a caller that only learned the
/// answer could still disagree with the compiler about it.
///
/// The reason a target did not get the vendored tree is collected here and
/// reported in **one** message, and only when this target has no D256 route at
/// all — `musa_arch::d256_route`, the same function the Rust side routes with.
/// A build that announces a gap it does not have is as misleading as one that
/// hides a gap it does: mp_22 says nothing here, because its own assembly serves
/// the operator.
/// Whether this build may compile the vendored mp_31 FMHA, or why not.
///
/// Pure, and separate from [`vendor_fmha_available`] for a reason the build id
/// made concrete: `build.rs` needs this answer **twice** — once as an input to
/// `computed_kernel_build_id` and once to decide what to compile — and the two
/// were served by one function that emitted `rustc-cfg` and the "this target has
/// no D256 kernel at all" warning as side effects. Cargo dedupes the cfg, so the
/// only symptom was the warning printing twice, which reads as two missing
/// kernels rather than one. A predicate that answers and an announcer that
/// speaks keep the count honest.
fn vendor_fmha_decision(
    arch: &musa_arch::MusaArch,
    musa_path: &Path,
) -> std::result::Result<(), String> {
    if arch.mcc_arch != VENDOR_FMHA_ARCH {
        return Err(format!(
            "the vendored FMHA is {VENDOR_FMHA_ARCH} device code and the vendored D256 \
             assembly is {ASSEMBLY_ARCH} ISA; this build targets {}",
            arch.mcc_arch
        ));
    }
    match musa_arch::musa_version_number(musa_path) {
        Some(version) if version >= VENDOR_FMHA_MCC_MIN => Ok(()),
        Some((major, minor)) => Err(format!(
            "the vendored FMHA was rendered and measured on MUSA toolkit {}.{}.x; this \
             toolkit reports {major}.{minor}",
            VENDOR_FMHA_MCC_MIN.0, VENDOR_FMHA_MCC_MIN.1
        )),
        None => Err(format!(
            "cannot read the MUSA toolkit version from {}; without it the vendored FMHA \
             ({VENDOR_FMHA_ARCH}, toolkit >= {}.{}.0) cannot be shown to apply",
            musa_path.display(),
            VENDOR_FMHA_MCC_MIN.0,
            VENDOR_FMHA_MCC_MIN.1
        )),
    }
}

fn vendor_fmha_available(arch: &musa_arch::MusaArch, musa_path: &Path) -> bool {
    let Err(reason) = vendor_fmha_decision(arch, musa_path) else {
        println!("cargo:rustc-cfg=apxinf_musa_mutlass_fmha");
        return true;
    };
    if musa_arch::d256_route(&arch.mcc_arch, false) == musa_arch::D256Route::Unsupported {
        println!(
            "cargo:warning=ApxInf MUSA: {reason}, so attention::mqa_bf16 will report itself as \
             a kernel gap"
        );
    }
    false
}

/// Native kernel compilation.
///
/// Each tree brings its own extension list, include roots and flags, because
/// the vendored FMHA is not built the way this repo builds its own kernels —
/// see `VENDOR_FMHA_FLAGS`. `include_vendor_fmha` is the arch and toolkit gate:
/// when it is false the tree is skipped, and `kernels/attention.rs` answers the
/// operator with a named error instead of a link failure.
fn compile_kernels(
    mcc: &Path,
    include_dir: &Path,
    arch: &musa_arch::MusaArch,
    manifest_dir: &Path,
    out_dir: &Path,
    include_vendor_fmha: bool,
) {
    let mut sources = Vec::new();
    // The same trees the kernel-build-id hashes, so a source cannot be compiled
    // into the archive while staying out of the hash that names it.
    for tree in KERNEL_INPUT_TREES {
        // Only the tree whose units were built with the vendor's flags is
        // skipped whole; `adapters` mixes the FMHA launcher with the operators
        // every target has, so it is filtered per file below.
        if tree.vendor_flags && !include_vendor_fmha {
            continue;
        }
        let directory = manifest_dir.join(tree.path);
        // The same depth rule as the hash and the watch: a tree that is
        // recursive has its sources where its headers are, not in one flat
        // directory. The vendored FMHA's `.mu` files are one level down from
        // the tree root, and a flat scan silently produced an archive whose
        // launcher had no kernel behind it.
        let mut candidates = Vec::new();
        if tree.recursive {
            collect_tree(&directory, &mut candidates);
        } else {
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    candidates.push(path);
                }
            }
        }
        for path in candidates {
            let name = path.file_name().unwrap_or_default();
            if !include_vendor_fmha
                && tree
                    .only_with_vendor_fmha
                    .iter()
                    .any(|skipped| name == std::ffi::OsStr::new(skipped))
            {
                continue;
            }
            let matches = path.extension().is_some_and(|extension| {
                tree.compiled
                    .iter()
                    .any(|wanted| extension == std::ffi::OsStr::new(wanted))
            });
            if matches {
                sources.push((tree, path));
            }
        }
    }
    sources.sort_by(|left, right| left.1.cmp(&right.1));

    if sources.is_empty() {
        println!("cargo:warning=ApxInf MUSA: no native kernels yet — all operators run through the eager bridge");
        return;
    }

    // A toolchain that cannot compile MUSA code at all is a broken build, not a
    // build without a GPU backend: `Ok(None)` is "this toolchain needs no shim",
    // and conflating the two is what turns a broken toolchain into a silently
    // disabled backend. The strength matches how a single adapter failing to
    // compile is handled below.
    let shim = match host_math_shim::resolve(mcc, arch, include_dir, manifest_dir) {
        Ok(shim) => shim,
        Err(reason) => panic!("MUSA kernel build cannot start: {reason}"),
    };
    if let Some(shim) = &shim {
        println!("cargo:warning={}", shim.reason);
    }
    // One probe for the whole adapter tree: `adapter_common.h` uses it to
    // decide whether a stream can name its own device. A toolkit that cannot
    // compile the call keeps the thread's selection, which is what the header
    // did before — the warning says which one this build took.
    let stream_device = stream_device::available(mcc, arch, include_dir, shim.as_ref());
    if !stream_device {
        println!(
            "cargo:warning=ApxInf MUSA: musaStreamGetDevice does not compile against this \
             toolkit; stream-keyed caches fall back to the thread's current device"
        );
    }
    let mut objects = Vec::new();
    for (tree, source) in &sources {
        let object = out_dir.join(format!(
            "{}.o",
            source.file_stem().unwrap().to_string_lossy()
        ));
        let mut command = Command::new(mcc);
        // `mcc` picks the language from the extension, and `.mu` is not a name
        // it infers MUSA from; `-x musa` says which language this is.
        command
            .arg("-x")
            .arg("musa")
            .arg(format!("--offload-arch={}", arch.mcc_arch))
            .arg("-c")
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg(format!("-I{}", include_dir.display()))
            .arg(format!("-I{}", manifest_dir.join("kernels").display()))
            // For the generated `mubin_images.h` the attention adapter includes.
            .arg(format!("-I{}", out_dir.display()))
            .arg(format!("-DAPXINF_KERNEL_BUILD_ID=\"{}\"", env_build_id()))
            .arg(source)
            .arg("-o")
            .arg(&object);
        if stream_device {
            command.arg("-DAPXINF_MUSA_HAS_STREAM_DEVICE");
        }
        if tree.vendor_flags {
            command.args(VENDOR_FMHA_FLAGS);
        } else {
            command.arg("-O3");
        }
        for include in tree.includes {
            command.arg(format!("-I{}", manifest_dir.join(include).display()));
        }
        // muDNN's C++ headers moved between toolkits, together with the library
        // that carries them: 5.1 keeps them in `include/mudnncxx/`, 4.3.5 in
        // `include/` itself. Point at the subdirectory only where it exists.
        let mudnncxx = include_dir.join("mudnncxx");
        if mudnncxx.is_dir() {
            command.arg(format!("-I{}", mudnncxx.display()));
        }
        // Only when the probe above showed this toolchain needs it, so a fixed
        // mcc never has a host header shadowed out from under it.
        if let Some(shim) = &shim {
            command.arg("-isystem").arg(&shim.include_dir);
        }
        let status = command.status();
        match status {
            Ok(status) if status.success() => objects.push(object),
            Ok(status) => {
                panic!("MUSA kernel build failed for {} ({status}); refusing to produce an unlinked native backend", source.display());
            }
            Err(error) => {
                panic!("cannot run mcc for {}: {error}", source.display());
            }
        }
    }

    let archive = out_dir.join("libapxinf_musa_kernels.a");
    // Start from nothing. `ar rcs` adds and replaces members but never removes
    // one, so an object whose source was deleted or renamed would stay in the
    // library forever — and the linker takes whichever member it meets first,
    // so the stale one can silently win. This is not hypothetical: renaming
    // `mublaslt_adapter.mu` to `mublas_adapter.mu` left the old object in the
    // archive ahead of the new one, and every measurement of the new adapter
    // was really measuring the old.
    if let Err(error) = std::fs::remove_file(&archive) {
        if error.kind() != std::io::ErrorKind::NotFound {
            println!("cargo:warning=cannot remove the stale kernel archive: {error}");
        }
    }
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&archive);
    for object in &objects {
        ar.arg(object);
    }
    if !ar.status().map(|status| status.success()).unwrap_or(false) {
        println!("cargo:warning=failed to archive MUSA kernels");
        println!("cargo:warning=ApxInf MUSA: compiling without GPU support");
        return;
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apxinf_musa_kernels");
}

/// Assemble the MP22 attention kernels and emit them as C byte arrays.
///
/// `mcc` cannot take assembly, and the driver wants a loadable module rather
/// than a linked symbol, so these go through the toolkit's own `llvm-mc` and
/// `lld` — the same two commands the assembly's upstream JIT runs. The images
/// are emitted as a generated header rather than kept as files so nothing has
/// to find them at run time; `adapters/mubin_fa_adapter.mu` includes the header
/// and hands the bytes to `muModuleLoadData`.
///
/// A missing assembler or a failed assembly emits empty arrays and a warning
/// instead of failing the build: the crate still compiles everywhere the rest
/// of the toolkit does, and the adapter reports the gap by name when the
/// operator is actually called. Failing the build here would take out the
/// whole backend over one operator.
fn emit_mubin_images(
    musa_path: &Path,
    arch: &musa_arch::MusaArch,
    manifest_dir: &Path,
    out_dir: &Path,
) {
    let asm_dir = manifest_dir.join("kernels/asm");
    let assembler = musa_path.join("bin/llvm-mc");
    let linker = musa_path.join("bin/lld");

    let mut header = String::from(
        "/* Generated by crates/apxinf-musa/build.rs from kernels/asm sources. Do not edit. */\n\
         #ifndef APXINF_MUSA_MUBIN_IMAGES_H\n\
         #define APXINF_MUSA_MUBIN_IMAGES_H\n\n",
    );
    // Whether to assemble is a property of the target, decided before the
    // assembler is asked anything. The `.asm` files are MP22 ISA and
    // `llvm-mc` rejects their scheduling annotations on every other generation,
    // so on a non-mp_22 target the only thing running it produces is a failure
    // message that says "assembling failed" about a build that was never going
    // to assemble anything — a warning that trains its readers to skip warnings,
    // and that reads like a broken toolchain rather than a different ISA.
    //
    // The arrays are still emitted, empty, because `mubin_fa_adapter.mu` is
    // compiled on every target: `ffi::mubin::apxinf_musa_fa_d256` is reachable
    // whenever the crate is, and the adapter answers "the image is empty" by
    // name if anything reaches it.
    let assembles_here = arch.mcc_arch == ASSEMBLY_ARCH;
    let mut usable = assembles_here && assembler.is_file() && linker.is_file();
    if assembles_here && !usable {
        println!(
            "cargo:warning=MUSA assembly: {} or {} is missing — \
             MP22 attention/GEMM images are unavailable",
            assembler.display(),
            linker.display()
        );
    }

    for (name, symbol) in MUBIN_VARIANTS
        .iter()
        .chain(QINT8_ASM_VARIANTS.iter())
        .chain(QINT8_GEMM_ASM_VARIANTS.iter())
    {
        let source = asm_dir.join(format!("{name}.asm"));
        let object = out_dir.join(format!("{name}.o"));
        let image = out_dir.join(format!("{name}.mubin"));
        let mut bytes = Vec::new();

        if usable {
            // `-mcpu` takes the same arch string as `--offload-arch`, so the
            // assembly follows APXINF_MUSA_ARCH rather than assuming mp_22.
            let assembled = Command::new(&assembler)
                .args(["--arch=mtgpu", "--filetype=obj"])
                .arg(format!("-mcpu={}", arch.mcc_arch))
                .arg(&source)
                .arg("-o")
                .arg(&object)
                .status();
            let linked = assembled
                .as_ref()
                .ok()
                .filter(|status| status.success())
                .map(|_| {
                    Command::new(&linker)
                        .args(["-flavor", "gnu", "--no-undefined", "-shared"])
                        .arg(&object)
                        .arg("-o")
                        .arg(&image)
                        .status()
                });
            match linked {
                Some(Ok(status)) if status.success() => match std::fs::read(&image) {
                    Ok(read) => bytes = read,
                    Err(error) => {
                        println!("cargo:warning=cannot read {}: {error}", image.display())
                    }
                },
                _ => println!(
                    "cargo:warning=MUSA assembly: assembling {} failed",
                    source.display()
                ),
            }
        }

        if bytes.is_empty() {
            usable = false;
            header.push_str(&format!("static const unsigned char {symbol}[] = {{0}};\n"));
        } else {
            header.push_str(&format!("static const unsigned char {symbol}[] = {{"));
            for (index, byte) in bytes.iter().enumerate() {
                if index % 16 == 0 {
                    header.push_str("\n    ");
                }
                header.push_str(&format!("0x{byte:02x},"));
            }
            header.push_str("\n};\n");
        }
        header.push_str(&format!(
            "static const unsigned long {symbol}Size = {};\n\n",
            bytes.len()
        ));
    }

    header.push_str("#endif\n");
    let path = out_dir.join("mubin_images.h");
    if let Err(error) = std::fs::write(&path, header) {
        println!("cargo:warning=cannot write {}: {error}", path.display());
    }
}

fn env_build_id() -> String {
    std::env::var("APXINF_KERNEL_BUILD_ID").unwrap_or_else(|_| "musa-kb2-unknown".to_string())
}
