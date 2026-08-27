use std::env;

#[path = "build_support/cuda_arch.rs"]
mod cuda_arch;

use cuda_arch::{is_cutlass_sm100_family, select_cuda_arch, ArchSource};

const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

fn hash_bytes(hash: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u128::from(*byte);
        *hash = hash.wrapping_mul(FNV1A_128_PRIME);
    }
}

fn collect_kernel_inputs(root: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_kernel_inputs(&path, files);
            continue;
        }
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("._"))
        {
            continue;
        }
        if path.extension().is_some_and(|extension| {
            matches!(
                extension.to_string_lossy().as_ref(),
                "cu" | "cuh" | "h" | "hh" | "hpp"
            )
        }) {
            files.push(path);
        }
    }
}

fn computed_kernel_build_id(
    cuda_sources_root: &std::path::Path,
    target_arch: &str,
    nvcc_arch: Option<&str>,
    cutlass_arch: Option<&str>,
) -> String {
    let mut hash = FNV1A_128_OFFSET;
    for value in [
        "apxinf-cuda-kernel-build-v1",
        env!("CARGO_PKG_VERSION"),
        target_arch,
        nvcc_arch.unwrap_or("native"),
        cutlass_arch.unwrap_or("native"),
    ] {
        hash_bytes(&mut hash, value.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }

    let mut files = Vec::new();
    collect_kernel_inputs(cuda_sources_root, &mut files);
    files.sort_unstable();
    for path in files {
        let relative = path.strip_prefix(cuda_sources_root).unwrap_or(&path);
        hash_bytes(&mut hash, relative.to_string_lossy().as_bytes());
        hash_bytes(&mut hash, &[0]);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read kernel build input {}: {error}", path.display()));
        hash_bytes(&mut hash, &bytes);
        hash_bytes(&mut hash, &[0xff]);
    }

    format!("kb1-{hash:032x}")
}

fn emit_kernel_build_id(build_id: &str) {
    assert!(
        !build_id.is_empty() && !build_id.contains('\n') && !build_id.contains('\r'),
        "APXINF_KERNEL_BUILD_ID must be non-empty and single-line"
    );
    println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={build_id}");
}

fn emit_rerun_if_changed_tree(root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            emit_rerun_if_changed_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn is_fa2_sm80_family(arch: &str) -> bool {
    matches!(arch, "sm_80" | "sm_86" | "sm_87" | "sm_89")
}

fn is_cutlass_sm89_family(arch: &str) -> bool {
    matches!(arch, "sm_89")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(nvtx_v2)");
    println!("cargo:rustc-check-cfg=cfg(nvtx_v3)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_fmha)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_gemm)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_bf16_sm89)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_int8_sm80)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_sm80)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_f16_sm100)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_direct_e4m3_sm100)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_marlin_sm89)");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH_CUTLASS");
    println!("cargo:rerun-if-env-changed=APXINF_KERNEL_BUILD_ID");
    println!("cargo:rerun-if-env-changed=CUDA_VISIBLE_DEVICES");
    println!("cargo:rerun-if-env-changed=NVIDIA_VISIBLE_DEVICES");
    println!("cargo:rerun-if-changed=build_support/cuda_arch.rs");
    // Only try to link CUDA if the toolkit is available.
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Candidate lib directories. Desktop CUDA uses lib64/; embedded / cross
    // layouts split libs across several subdirs. Drive OS 7 puts libcublas
    // under `thor/targets/aarch64-linux/lib/` while libnvToolsExt lives in
    // the top-level lib64/. Add every candidate that exists so -lcudart,
    // -lcublas, and -lnvtx all resolve regardless of where each landed.
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let candidate_lib_dirs = [
        format!("{cuda_path}/lib64"),
        format!("{cuda_path}/lib"),
        format!("{cuda_path}/targets/{arch}/lib"),
        format!("{cuda_path}/targets/aarch64-linux/lib"),
        format!("{cuda_path}/targets/x86_64-linux/lib"),
        format!("{cuda_path}/thor/targets/aarch64-linux/lib"),
    ];
    let lib_dirs: Vec<String> = candidate_lib_dirs
        .iter()
        .filter(|d| std::path::Path::new(d).exists())
        .cloned()
        .collect();

    // Check if CUDA is available
    let has_cuda = !lib_dirs.is_empty();

    if has_cuda {
        for d in &lib_dirs {
            println!("cargo:rustc-link-search=native={d}");
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");

        // NVTX naming split: desktop CUDA 12 ships `libnvtx3interop.so`, but
        // Drive OS 7 (and older embedded CUDA runtimes) ship the classic
        // `libnvToolsExt.so` (NVTX v2). Both export `nvtxRangePushA` /
        // `nvtxRangePop` — we just have to pick the right `-l` name. Emit a
        // cfg the `nvtx` module reads. Search across all lib dirs since the
        // NVTX lib may sit in a different subdir than cublas (Drive OS does
        // exactly this).
        if std::env::var("CARGO_FEATURE_NVTX").is_ok() {
            let has_v3 = lib_dirs.iter().any(|d| {
                std::path::Path::new(&format!("{d}/libnvtx3interop.so")).exists()
                    || std::path::Path::new(&format!("{d}/libnvtx3interop.so.1")).exists()
            });
            let has_v2 = lib_dirs.iter().any(|d| {
                std::path::Path::new(&format!("{d}/libnvToolsExt.so")).exists()
                    || std::path::Path::new(&format!("{d}/libnvToolsExt.so.1")).exists()
            });
            if has_v3 {
                // Desktop CUDA 12.x. `nvtx.rs` will use `#[link(name = "nvtx3interop")]`.
                println!("cargo:rustc-cfg=nvtx_v3");
            } else if has_v2 {
                // Drive OS / embedded CUDA. `nvtx.rs` falls back to `-lnvToolsExt`.
                println!("cargo:rustc-cfg=nvtx_v2");
            } else {
                println!("cargo:warning=NVTX feature enabled but neither libnvtx3interop nor libnvToolsExt found in any of {lib_dirs:?} — build will fail at link time");
                println!("cargo:rustc-cfg=nvtx_v3"); // preserve prior behavior
            }
            // Suppress the "unexpected cfg" warning under Rust 1.80+ check-cfg.
        }

        // Compile CUDA kernels if nvcc is available
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let kernels_dir = format!("{manifest_dir}/kernels");
        let adapters_dir = format!("{manifest_dir}/adapters");
        if std::path::Path::new(&kernels_dir).exists() {
            // Track the directory as well as its current members so adding a
            // new top-level .cu file invalidates an existing Cargo build.
            println!("cargo:rerun-if-changed={kernels_dir}");
            println!("cargo:rerun-if-changed={adapters_dir}");
            // The build ID hashes every CUDA/CUTLASS source below kernels/.
            // Track the same complete tree so an in-place header edit cannot
            // leave Cargo using a stale build ID or stale device object.
            emit_rerun_if_changed_tree(std::path::Path::new(&kernels_dir));
            emit_rerun_if_changed_tree(std::path::Path::new(&adapters_dir));
            // Pick a target arch: explicit override > native CUDA device
            // detection. Never infer a GPU architecture from the CPU target:
            // Orin, Thor-U, and Thor are all aarch64 but require different
            // device code. Cross-compilation therefore requires an override.
            let arch_for_target = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
            let out_dir = env::var("OUT_DIR").unwrap();
            let nvcc = {
                let bundled = format!("{cuda_path}/bin/nvcc");
                if std::path::Path::new(&bundled).exists() {
                    bundled
                } else {
                    "nvcc".to_string()
                }
            };
            let host = env::var("HOST").unwrap_or_default();
            let target = env::var("TARGET").unwrap_or_default();
            let selection = select_cuda_arch(
                env::var("APXINF_CUDA_ARCH").ok(),
                env::var("APXINF_CUDA_ARCH_CUTLASS").ok(),
                &host,
                &target,
                std::path::Path::new(&nvcc),
                std::path::Path::new(&out_dir),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "CUDA architecture selection failed: {error}\n\
                     Set APXINF_CUDA_ARCH explicitly when auto-detection is unavailable \
                     (Orin: sm_87, Thor-U: sm_101, Thor: sm_110)."
                )
            });
            match &selection.source {
                ArchSource::Explicit => println!(
                    "cargo:warning=ApxInf CUDA architecture: {} (explicit), CUTLASS: {}",
                    selection.nvcc_arch, selection.cutlass_arch
                ),
                ArchSource::Detected { device_count } => println!(
                    "cargo:warning=ApxInf CUDA architecture: {} (auto-detected from {device_count} visible GPU(s)), CUTLASS: {}",
                    selection.nvcc_arch, selection.cutlass_arch
                ),
            }
            let nvcc_arch = Some(selection.nvcc_arch);
            let cutlass_arch = Some(selection.cutlass_arch);
            let kernel_build_id = env::var("APXINF_KERNEL_BUILD_ID").unwrap_or_else(|_| {
                computed_kernel_build_id(
                    std::path::Path::new(&manifest_dir),
                    &arch_for_target,
                    nvcc_arch.as_deref(),
                    cutlass_arch.as_deref(),
                )
            });
            emit_kernel_build_id(&kernel_build_id);

            // Compile host adapters from an explicit list. Template-heavy
            // CUTLASS operators are added below as their own translation
            // units; adapters include declaration-only headers and own only
            // the stable C ABI.
            let mut kernel_files = vec![
                std::path::Path::new(&adapters_dir).join("core_kernels_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("sampling_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("static_bf16_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("w8a8_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("custom_kernels.cu"),
                std::path::Path::new(&adapters_dir).join("cublas_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("cublaslt_adapter.cu"),
            ];
            assert!(
                kernel_files.iter().all(|path| path.is_file()),
                "one or more required CUDA adapters are missing under {adapters_dir}"
            );

            let cutlass_root = std::path::Path::new(&kernels_dir).join("cutlass");
            let cutlass_fmha_operator = cutlass_root.join("fmha_sm100.cu");
            let cutlass_gemm_operator = cutlass_root.join("fp8_gemm_sm100.cu");
            let cutlass_fp8_dual_operator = cutlass_root.join("fp8_dual_geglu_sm100.cu");
            let cutlass_fp8_header = cutlass_root.join("fp8_operators_sm100.h");
            let cutlass_bf16_operator = cutlass_root.join("bf16_gemm_sm100.cu");
            let cutlass_bf16_dual_operator = cutlass_root.join("bf16_dual_geglu_sm100.cu");
            let cutlass_bf16_header = cutlass_root.join("bf16_operators_sm100.h");
            let cutlass_bf16_sm89_operator = cutlass_root.join("bf16_gemm_sm89.cu");
            let cutlass_bf16_sm89_header = cutlass_root.join("bf16_operators_sm89.h");
            let cutlass_int8_operator = cutlass_root.join("w8a8_gemm_sm80.cu");
            let cutlass_fmha = std::path::Path::new(&adapters_dir).join("cutlass_fmha_adapter.cu");
            let cutlass_gemm = std::path::Path::new(&adapters_dir).join("cutlass_fp8_adapter.cu");
            let cutlass_bf16 = std::path::Path::new(&adapters_dir).join("cutlass_bf16_adapter.cu");
            let cutlass_bf16_sm89 =
                std::path::Path::new(&adapters_dir).join("cutlass_bf16_sm89_adapter.cu");
            let cutlass_int8 = std::path::Path::new(&adapters_dir).join("cutlass_w8a8_adapter.cu");
            let marlin_adapter = std::path::Path::new(&adapters_dir).join("marlin_adapter.cu");
            let mut cutlass_includes = Vec::new();
            if cutlass_arch.as_deref().is_some_and(is_cutlass_sm100_family) {
                let fmha = cutlass_root.join("fmha");
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                assert!(
                    fmha.is_dir() && cutlass.is_dir() && cutlass_utils.is_dir(),
                    "vendored CUTLASS/FMHA headers are incomplete under {}",
                    cutlass_root.display()
                );
                assert!(
                    cutlass_gemm_operator.is_file()
                        && cutlass_fp8_dual_operator.is_file()
                        && cutlass_fp8_header.is_file()
                        && cutlass_bf16_operator.is_file()
                        && cutlass_bf16_dual_operator.is_file()
                        && cutlass_bf16_header.is_file()
                        && cutlass_fmha_operator.is_file()
                        && cutlass_gemm.is_file()
                        && cutlass_bf16.is_file()
                        && cutlass_fmha.is_file(),
                    "CUTLASS operators or native C ABI adapters are missing"
                );
                cutlass_includes.extend([fmha, cutlass, cutlass_utils]);
                kernel_files.extend([
                    cutlass_gemm_operator.clone(),
                    cutlass_fp8_dual_operator.clone(),
                    cutlass_bf16_operator.clone(),
                    cutlass_bf16_dual_operator.clone(),
                    cutlass_gemm.clone(),
                    cutlass_bf16.clone(),
                ]);
                kernel_files.push(cutlass_fmha.clone());
                println!("cargo:rustc-cfg=apxinf_cutlass_gemm");
                println!("cargo:rustc-cfg=apxinf_cutlass_fmha");
                emit_rerun_if_changed_tree(&cutlass_root);
            }

            if cutlass_arch.as_deref().is_some_and(is_cutlass_sm89_family) {
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                assert!(
                    cutlass_bf16_sm89_operator.is_file()
                        && cutlass_bf16_sm89_header.is_file()
                        && cutlass_bf16_sm89.is_file()
                        && cutlass.is_dir()
                        && cutlass_utils.is_dir(),
                    "CUTLASS BF16 SM89 adapter or headers are missing under {}",
                    cutlass_root.display()
                );
                cutlass_includes.extend([cutlass_root.clone(), cutlass, cutlass_utils]);
                kernel_files.extend([
                    cutlass_bf16_sm89_operator.clone(),
                    cutlass_bf16_sm89.clone(),
                ]);
                println!("cargo:rustc-cfg=apxinf_cutlass_bf16_sm89");
                emit_rerun_if_changed_tree(&cutlass_root);
                emit_rerun_if_changed_tree(std::path::Path::new(&adapters_dir));
            }

            let mut cutlass_int8_includes = Vec::new();
            if nvcc_arch.as_deref().is_some_and(is_fa2_sm80_family) {
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                let extensions = cutlass_root.join("extensions");
                assert!(
                    cutlass_int8_operator.is_file()
                        && cutlass_int8.is_file()
                        && cutlass.is_dir()
                        && cutlass_utils.is_dir()
                        && extensions
                            .join("epilogue/epilogue_per_row_per_col_scale.h")
                            .is_file()
                        && extensions
                            .join("gemm/gemm_universal_base_compat.h")
                            .is_file()
                        && extensions
                            .join("gemm/gemm_with_epilogue_visitor.h")
                            .is_file(),
                    "vendored SM80 INT8 CUTLASS sources are incomplete under {}",
                    cutlass_root.display()
                );
                cutlass_int8_includes.extend([cutlass_root.clone(), cutlass, cutlass_utils]);
                kernel_files.push(cutlass_int8.clone());
                println!("cargo:rustc-cfg=apxinf_cutlass_int8_sm80");
                emit_rerun_if_changed_tree(&extensions);
            }

            let fa2_root = cutlass_root.join("fa2");
            let fa2_operator = cutlass_root.join("fa2_bf16_sm80.cu");
            let fa2_wrapper = std::path::Path::new(&adapters_dir).join("fa2_adapter.cu");
            let mut fa2_sources = Vec::new();
            let mut fa2_direct_e4m3_sources = Vec::new();
            let mut fa2_includes = Vec::new();
            let fa2_sm80 = nvcc_arch.as_deref().is_some_and(is_fa2_sm80_family);
            let fa2_f16_sm100 = nvcc_arch.as_deref().is_some_and(is_cutlass_sm100_family);
            if fa2_sm80 || fa2_f16_sm100 {
                let fa2_hdim96 = fa2_root.join("flash_attn/flash_fwd_hdim96_bf16_sm80.cu");
                let fa2_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_bf16_sm80.cu");
                let fa2_f16_hdim96 = fa2_root.join("flash_attn/flash_fwd_hdim96_fp16.cu");
                let fa2_f16_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_fp16.cu");
                let fa2_cutlass = fa2_root.join("cutlass/include");
                assert!(
                    fa2_operator.is_file()
                        && fa2_wrapper.is_file()
                        && fa2_hdim96.is_file()
                        && fa2_hdim256.is_file()
                        && fa2_f16_hdim96.is_file()
                        && fa2_f16_hdim256.is_file()
                        && fa2_cutlass.is_dir(),
                    "vendored FlashAttention-2 sources are incomplete under {}",
                    fa2_root.display()
                );
                fa2_sources.extend([
                    fa2_wrapper.clone(),
                    fa2_hdim96,
                    fa2_hdim256,
                    fa2_f16_hdim96,
                    fa2_f16_hdim256,
                ]);
                if fa2_sm80 {
                    let fa2_split_hdim256 =
                        fa2_root.join("flash_attn/flash_fwd_split_hdim256_bf16_sm80.cu");
                    assert!(
                        fa2_split_hdim256.is_file(),
                        "vendored FlashAttention-2 split-KV source is incomplete under {}",
                        fa2_root.display()
                    );
                    fa2_sources.push(fa2_split_hdim256);
                }
                if fa2_f16_sm100 {
                    println!("cargo:rustc-cfg=apxinf_fa2_f16_sm100");
                    let direct_operator = cutlass_root.join("fa2_f16_e4m3_sm100.cu");
                    let direct_wrapper =
                        std::path::Path::new(&adapters_dir).join("fa2_direct_e4m3_adapter.cu");
                    let direct_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_fp16_e4m3.cu");
                    assert!(
                        direct_operator.is_file()
                            && direct_wrapper.is_file()
                            && direct_hdim256.is_file(),
                        "FA2 direct-E4M3 FA2 sources are incomplete"
                    );
                    fa2_direct_e4m3_sources.extend([direct_wrapper, direct_hdim256]);
                    kernel_files.extend(fa2_direct_e4m3_sources.iter().cloned());
                    println!("cargo:rustc-cfg=apxinf_fa2_direct_e4m3_sm100");
                }
                fa2_includes.extend([fa2_root.clone(), fa2_cutlass]);
                kernel_files.extend(fa2_sources.iter().cloned());
                if fa2_sm80 {
                    println!("cargo:rustc-cfg=apxinf_fa2_sm80");
                }
                emit_rerun_if_changed_tree(&fa2_root);
            }

            if nvcc_arch.as_deref() == Some("sm_89") {
                let marlin_root = std::path::Path::new(&kernels_dir).join("marlin");
                assert!(
                    marlin_adapter.is_file()
                        && marlin_root.join("kernel.h").is_file()
                        && marlin_root.join("marlin_template.h").is_file()
                        && marlin_root.join("core/scalar_type.hpp").is_file(),
                    "vendored SM89 Marlin sources are incomplete under {}",
                    marlin_root.display()
                );
                kernel_files.push(marlin_adapter.clone());
                println!("cargo:rustc-cfg=apxinf_marlin_sm89");
                emit_rerun_if_changed_tree(&marlin_root);
            }

            if !kernel_files.is_empty() {
                let target_include_dirs = [
                    format!("{cuda_path}/include"),
                    format!("{cuda_path}/targets/{arch}/include"),
                    format!("{cuda_path}/targets/aarch64-linux/include"),
                    format!("{cuda_path}/thor/targets/aarch64-linux/include"),
                ];
                for entry in &kernel_files {
                    println!("cargo:rerun-if-changed={}", entry.display());
                    let stem = entry.file_stem().unwrap().to_string_lossy().to_string();
                    let obj = format!("{out_dir}/{stem}.o");

                    let mut cmd = std::process::Command::new(&nvcc);
                    cmd.args([
                        "-c",
                        &entry.to_string_lossy(),
                        "-o",
                        &obj,
                        "--compiler-options",
                        "-fPIC",
                        "-O3",
                        "-std=c++17",
                    ]);
                    let selected_arch = if entry == &cutlass_fmha
                        || entry == &cutlass_gemm_operator
                        || entry == &cutlass_fp8_dual_operator
                        || entry == &cutlass_bf16_operator
                        || entry == &cutlass_bf16_dual_operator
                        || entry == &cutlass_bf16_sm89_operator
                        || entry == &cutlass_bf16_sm89
                        || entry == &cutlass_gemm
                        || entry == &cutlass_bf16
                    {
                        cutlass_arch.as_ref()
                    } else {
                        nvcc_arch.as_ref()
                    };
                    if let Some(selected_arch) = selected_arch {
                        cmd.args([format!("-arch={selected_arch}")]);
                    }
                    for include in &target_include_dirs {
                        if std::path::Path::new(include).exists() {
                            cmd.arg(format!("-I{include}"));
                        }
                    }
                    if entry == &cutlass_fmha
                        || entry == &cutlass_gemm_operator
                        || entry == &cutlass_fp8_dual_operator
                        || entry == &cutlass_bf16_operator
                        || entry == &cutlass_bf16_dual_operator
                        || entry == &cutlass_bf16_sm89_operator
                        || entry == &cutlass_bf16_sm89
                        || entry == &cutlass_gemm
                        || entry == &cutlass_bf16
                    {
                        cmd.arg("--expt-relaxed-constexpr");
                        cmd.arg("--expt-extended-lambda");
                        for include in &cutlass_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if entry == &cutlass_fp8_dual_operator {
                        cmd.arg("-DAPXINF_FP8_DUAL_GEGLU_PRODUCTION=1");
                    }
                    if entry == &cutlass_bf16_dual_operator {
                        cmd.arg("-DAPXINF_BF16_DUAL_GEGLU_PRODUCTION=1");
                    }
                    if entry == &cutlass_int8 {
                        cmd.arg("--expt-relaxed-constexpr");
                        cmd.arg("--expt-extended-lambda");
                        for include in &cutlass_int8_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if fa2_sources.contains(entry) {
                        cmd.args([
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_HALF2_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-DFLASH_NAMESPACE=apxinf_fa2",
                        ]);
                        if fa2_sm80 {
                            cmd.arg("-DAPXINF_FA2_SM80=1");
                        }
                        for include in &fa2_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if fa2_direct_e4m3_sources.contains(entry) {
                        cmd.args([
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_HALF2_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-DFLASH_NAMESPACE=apxinf_fa2_direct_e4m3",
                            "-DAPXINF_FA2_DIRECT_E4M3=1",
                        ]);
                        for include in &fa2_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if entry == &marlin_adapter {
                        cmd.arg("--expt-relaxed-constexpr");
                    }
                    let status = cmd.status().expect("failed to run nvcc");

                    assert!(status.success(), "nvcc failed for {}", entry.display());
                }

                // Create a static library from all kernel objects
                let objs: Vec<String> = kernel_files
                    .iter()
                    .map(|e| {
                        let stem = e.file_stem().unwrap().to_string_lossy().to_string();
                        format!("{out_dir}/{stem}.o")
                    })
                    .collect();

                let lib_path = format!("{out_dir}/libapxinf_kernels.a");
                // `ar rcs` updates/replaces named members but does not remove
                // objects that disappeared from the source list. Recreate the
                // archive so a renamed adapter (for example static_fp8.cu)
                // cannot leave duplicate, stale C ABI symbols behind.
                if let Err(error) = std::fs::remove_file(&lib_path) {
                    assert_eq!(
                        error.kind(),
                        std::io::ErrorKind::NotFound,
                        "remove stale CUDA archive {lib_path}: {error}"
                    );
                }
                let status = std::process::Command::new("ar")
                    .arg("rcs")
                    .arg(&lib_path)
                    .args(&objs)
                    .status()
                    .expect("failed to run ar");

                assert!(status.success(), "ar failed");

                println!("cargo:rustc-link-search=native={out_dir}");
                println!("cargo:rustc-link-lib=static=apxinf_kernels");
                // Keep CUDA math DSOs after the static archive. GNU ld's
                // --as-needed otherwise discards cublasLt before it sees the
                // static inference archive's references.
                println!("cargo:rustc-link-lib=cublasLt");
                println!("cargo:rustc-link-lib=cublas");
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-link-lib=stdc++");
            }
        }

        println!("cargo:rustc-cfg=feature=\"cuda\"");
    } else {
        emit_kernel_build_id(
            &env::var("APXINF_KERNEL_BUILD_ID")
                .unwrap_or_else(|_| format!("no-cuda-{}", env!("CARGO_PKG_VERSION"))),
        );
        println!("cargo:warning=CUDA not found — building without GPU support");
        println!("cargo:rustc-cfg=feature=\"no_cuda\"");
    }
}
