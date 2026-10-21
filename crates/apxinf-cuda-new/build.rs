use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../apxinf-cuda/build_support/nvcc_build.rs"]
mod nvcc_build;
#[path = "build_support/attention_fingerprint.rs"]
mod attention_fingerprint;
#[path = "build_support/cuda_arch.rs"]
mod cuda_arch;
#[path = "build_support/gemm_fingerprint.rs"]
mod gemm_fingerprint;

use cuda_arch::{
    gencode_args, is_cutlass_sm100_family, select_cuda_arch, target_features, ArchSelection,
    ArchSource,
};

fn write_arch_header(out: &Path, selection: &ArchSelection) -> PathBuf {
    let path = out.join("apxinf_cuda_arches.h");
    let mut header = String::from(
        "#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace apxinf::gemm {\n\
         constexpr uint64_t kDeviceFeatureNativeFp8 = UINT64_C(1) << 0;\n\
         constexpr uint64_t kDeviceFeatureCutlassSm100 = UINT64_C(1) << 1;\n\
         constexpr uint64_t kDeviceFeatureFa2 = UINT64_C(1) << 2;\n\
         constexpr uint64_t kDeviceFeatureCutlassSm89Bf16Geglu = UINT64_C(1) << 3;\n\
         constexpr uint64_t kDeviceFeatureCutlassSm87W8A8 = UINT64_C(1) << 4;\n\
         struct CompiledTarget { int sm; uint64_t features; };\n\
         constexpr CompiledTarget kCompiledTargets[] = {\n",
    );
    for target in &selection.targets {
        let features = target_features(target);
        writeln!(header, "  {{{}, UINT64_C({features})}},", target.sm()).unwrap();
    }
    header.push_str(
        "};\n\
         inline const CompiledTarget* compiled_target(int sm) {\n\
           for (const auto& target : kCompiledTargets) {\n\
             if (target.sm == sm) return &target;\n\
           }\n\
           return nullptr;\n\
         }\n\
         }  // namespace apxinf::gemm\n",
    );
    // Rewrite only when the content actually changed: every adapter that
    // includes this header is a dependency in its `.d` file, so bumping the
    // mtime on an unchanged header would force the object cache to rebuild all
    // of them on every run.
    if std::fs::read_to_string(&path).ok().as_deref() != Some(header.as_str()) {
        std::fs::write(&path, header)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    }
    path
}

fn rerun_tree(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn rerun_paths(root: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rerun_paths(&path, files);
        } else {
            files.push(path);
        }
    }
}

fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination)
        .unwrap_or_else(|error| panic!("create {}: {error}", destination.display()));
    for entry in std::fs::read_dir(source)
        .unwrap_or_else(|error| panic!("read {}: {error}", source.display()))
    {
        let entry = entry.unwrap_or_else(|error| panic!("read directory entry: {error}"));
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &destination_path);
        } else {
            std::fs::copy(&source_path, &destination_path).unwrap_or_else(|error| {
                panic!(
                    "copy {} to {}: {error}",
                    source_path.display(),
                    destination_path.display()
                )
            });
        }
    }
}

fn stage_patched_fa2(native: &Path, fa2_root: &Path, out: &Path) -> PathBuf {
    let staged = out.join("fa2-direct-e4m3-patched");
    let patch = native
        .join("patches")
        .join("fa2-direct-e4m3-output.patch");
    // Re-staging rewrites every file in the tree, which bumps the mtime of the
    // patched sources the E4M3 translation units depend on. Skip it when the
    // exact source tree and patch that produced the current staging are
    // unchanged, so the object cache survives a rebuild that only touched Rust.
    let stamp = out.join("fa2-direct-e4m3-patched.stamp");
    let mut fingerprint = 0x6c62272e07bb014262b821756295c58du128;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            fingerprint ^= u128::from(*byte);
            fingerprint = fingerprint.wrapping_mul(0x0000000001000000000000000000013b);
        }
    };
    let mut files = Vec::new();
    rerun_paths(&fa2_root.join("flash_attn"), &mut files);
    files.sort_unstable();
    for file in &files {
        feed(file.to_string_lossy().as_bytes());
        feed(&std::fs::read(file).unwrap_or_else(|error| {
            panic!("read FA2 staging input {}: {error}", file.display())
        }));
    }
    feed(&std::fs::read(&patch).unwrap_or_else(|error| {
        panic!("read FA2 patch {}: {error}", patch.display())
    }));
    let key = format!("{fingerprint:032x}");
    if staged.is_dir()
        && std::fs::read_to_string(&stamp).ok().as_deref() == Some(key.as_str())
    {
        return staged;
    }
    if staged.exists() {
        std::fs::remove_dir_all(&staged)
            .unwrap_or_else(|error| panic!("remove {}: {error}", staged.display()));
    }
    copy_tree(&fa2_root.join("flash_attn"), &staged.join("flash_attn"));
    let mut command = Command::new("patch");
    command
        .current_dir(&staged)
        .args(["--batch", "--forward", "-p0", "-i"])
        .arg(&patch);
    run(&mut command, "apply FA2 direct-E4M3 patch");
    std::fs::write(&stamp, &key)
        .unwrap_or_else(|error| panic!("write {}: {error}", stamp.display()));
    staged
}

fn cuda_library_directories(cuda: &str, cpu_arch: &str) -> Vec<PathBuf> {
    [
        format!("{cuda}/lib64"),
        format!("{cuda}/lib"),
        format!("{cuda}/targets/{cpu_arch}/lib"),
        format!("{cuda}/targets/aarch64-linux/lib"),
        format!("{cuda}/targets/x86_64-linux/lib"),
        format!("{cuda}/thor/targets/aarch64-linux/lib"),
    ]
    .into_iter()
    .map(PathBuf::from)
    .filter(|path| path.is_dir())
    .collect()
}

fn run(command: &mut Command, action: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("{action}: {error}"));
    assert!(status.success(), "{action} failed with {status}");
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(apxinf_nvfp4_gemm_swiglu)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_decode)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH_CUTLASS");
    println!("cargo:rerun-if-env-changed=APXINF_KERNEL_BUILD_ID");
    println!("cargo:rerun-if-env-changed=CUDA_VISIBLE_DEVICES");
    println!("cargo:rerun-if-changed=build_support/cuda_arch.rs");
    println!("cargo:rerun-if-changed=build_support/attention_fingerprint.rs");
    println!("cargo:rerun-if-changed=build_support/gemm_fingerprint.rs");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native = manifest.join("native");
    rerun_tree(&native);

    let cuda = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    let cpu_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let library_directories = cuda_library_directories(&cuda, &cpu_arch);
    let bundled_nvcc = PathBuf::from(format!("{cuda}/bin/nvcc"));
    if library_directories.is_empty() || !bundled_nvcc.is_file() {
        let id = env::var("APXINF_KERNEL_BUILD_ID")
            .unwrap_or_else(|_| format!("gemm-no-cuda-{}", env!("CARGO_PKG_VERSION")));
        println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={id}");
        println!("cargo:warning=CUDA toolkit not found; native GEMM library was not built");
        return;
    }

    for directory in &library_directories {
        println!("cargo:rustc-link-search=native={}", directory.display());
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let selection = select_cuda_arch(
        env::var("APXINF_CUDA_ARCH").ok(),
        env::var("APXINF_CUDA_ARCH_CUTLASS").ok(),
        &host,
        &target,
        &bundled_nvcc,
        &out,
    )
    .unwrap_or_else(|error| {
        panic!(
            "CUDA architecture selection failed: {error}\nSet APXINF_CUDA_ARCH explicitly when cross-compiling"
        )
    });
    match &selection.source {
        ArchSource::Explicit => println!("cargo:warning=GEMM targets selected explicitly"),
        ArchSource::Detected { device } => {
            println!("cargo:warning=GEMM target detected from current CUDA device {device}")
        }
    }
    let target_summary = selection
        .targets
        .iter()
        .map(|target| format!("{} (CUTLASS {})", target.nvcc_arch, target.cutlass_arch))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo:warning=GEMM targets: {target_summary}");

    let native_sm110 = selection.targets.iter().any(|target| target.sm() == 110)
        && selection
            .targets
            .iter()
            .any(|target| is_cutlass_sm100_family(&target.cutlass_arch));
    let nvfp4_gemm_swiglu = native_sm110;
    if nvfp4_gemm_swiglu {
        println!("cargo:rustc-cfg=apxinf_nvfp4_gemm_swiglu");
    }

    let id = env::var("APXINF_KERNEL_BUILD_ID").unwrap_or_else(|_| {
        gemm_fingerprint::build_id(
            &native,
            &target,
            selection
                .targets
                .iter()
                .map(|arch| (arch.nvcc_arch.as_str(), arch.cutlass_arch.as_str())),
        )
    });
    let attention_id = attention_fingerprint::build_id(
        &native,
        &target,
        selection
            .targets
            .iter()
            .map(|arch| (arch.nvcc_arch.as_str(), arch.cutlass_arch.as_str())),
    );
    assert!(
        !id.chars()
            .any(|character| matches!(character, '\n' | '\r' | '"')),
        "invalid kernel build ID"
    );
    println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={id}");
    write_arch_header(&out, &selection);

    let adapters = native.join("adapters");
    let mut generic_sources = [
        "runtime.cpp",
        "gemm/candidates.cpp",
        "gemm/tuning_key.cpp",
        "../framework/tuning_db.cpp",
        "gemm/autotune.cpp",
        "gemm/reference.cpp",
        "gemm/execution.cpp",
        "gemm/providers/cublas.cu",
        "gemm/providers/cublaslt.cu",
        "gemm/providers/cutlass.cu",
        "gemm/providers/custom.cu",
        "attention/candidates.cpp",
        "attention/tuning_key.cpp",
        "attention/autotune.cpp",
        "attention/execution.cpp",
        "attention/providers/custom.cu",
        "elementwise/execution.cpp",
        "linear_attention/execution.cpp",
        "attn_helpers/execution.cpp",
        "reduction/execution.cpp",
        "sampling.cu",
        "norm.cu",
        "pointwise.cu",
        "rope.cu",
        "gather.cu",
        "quantization.cu",
    ]
    .map(|source| adapters.join(source))
    .to_vec();
    generic_sources.push(native.join("kernels/custom/mlp_ops.cu"));
    generic_sources.push(native.join("kernels/custom/gdn_ops.cu"));
    generic_sources.push(native.join("kernels/flashinfer_gdn/flashinfer_gdn_tma.cpp"));
    generic_sources.push(native.join("kernels/custom/attn_ops.cu"));
    generic_sources.push(native.join("kernels/custom/model_ops.cu"));
    generic_sources.push(native.join("tests/framework_backend.cu"));
    let cutlass_root = native.join("kernels/cutlass");
    let fa2_root = native.join("kernels/fa2");
    let fa2_compat_root = native.join("kernels/fa2_compat");
    let attention_kernel_root = native.join("kernels/attention");
    let mut cutlass_sources = Vec::new();
    let mut sm110_sources = Vec::new();
    let mut cutlass_sm87_sources = Vec::new();
    let mut cutlass_sm89_sources = Vec::new();
    if selection
        .targets
        .iter()
        .any(|target| target.cutlass_arch == "sm_87")
    {
        cutlass_sm87_sources.push(cutlass_root.join("ops/gemm").join("gemm_i8_bf16_sm80.cu"));
    }
    if selection
        .targets
        .iter()
        .any(|target| target.cutlass_arch == "sm_89")
    {
        cutlass_sm89_sources.push(
            cutlass_root
                .join("ops/gemm")
                .join("gemm_bf16_geglu_sm89.cu"),
        );
    }
    if selection
        .targets
        .iter()
        .any(|target| is_cutlass_sm100_family(&target.cutlass_arch))
    {
        let operators = cutlass_root.join("ops/gemm");
        cutlass_sources.extend(
            [
                "gemm_e4m3_f16_sm100.cu",
                "gemm_e4m3_geglu_interleaved_sm100.cu",
                "gemm_bf16_geglu_sm100.cu",
                "gemm_bf16_geglu_interleaved_sm100.cu",
                "gemm_nvfp4_sm100.cu",
            ]
            .map(|source| operators.join(source)),
        );
        if native_sm110 {
            sm110_sources.push(operators.join("gemm_nvfp4_swiglu_sm100.cu"));
            let attention = cutlass_root.join("ops/attention");
            sm110_sources.extend(
                [
                    "fmha_d256_causal_sm110.cu",
                    "fmha_d256_split_batch_sm110.cu",
                    "fmha_strided_groups_sm110.cu",
                ]
                .map(|source| attention.join(source)),
            );
            sm110_sources.extend([
                native.join("kernels/custom/gdn_bf16_prefill_prologue.cu"),
                native.join(
                    "adapters/linear_attention/providers/gdn_bf16_prefill_sm110.cu",
                ),
                native.join("kernels/flashinfer_gdn/gdn_bf16_prefill_sm110_launch.cu"),
            ]);
        }
        cutlass_sources.push(adapters.join("attention/providers/cutlass.cu"));
        // Vendored FlashInfer Cake GDN prefill: plain CUDA C++, but it needs
        // the same compute_110a codegen as the CUTLASS group because it emits
        // tcgen05 and TMA. See native/kernels/flashinfer_gdn/README.md.
        cutlass_sources.push(native.join("kernels/flashinfer_gdn/flashinfer_gdn_launch.cu"));
    } else {
        // The descriptor-side wrapper is part of the stable ABI on every CUDA
        // target, but the generated Cake kernel itself requires Blackwell.
        // Keep non-Blackwell builds linkable and report unsupported at launch.
        generic_sources.push(native.join("kernels/flashinfer_gdn/flashinfer_gdn_stub.cpp"));
    }
    let has_fa2 = selection.targets.iter().any(|target| target.sm() >= 80);
    // The allocation-free FA2 decode ABI is a stub that returns
    // APXINF_STATUS_UNSUPPORTED unless FA2 was compiled in; surface that so a
    // dependent crate can select the fallback without a runtime failure path.
    if has_fa2 {
        println!("cargo:rustc-cfg=apxinf_fa2_decode");
    }
    let mut fa2_sources = Vec::new();
    if has_fa2 {
        generic_sources.push(adapters.join("attention/providers/fa2.cpp"));
        fa2_sources.extend(
            [
                "fa2.cu",
                "fa2_decode_splitkv.cu",
                "flash_attn/flash_fwd_hdim128_bf16_sm80.cu",
                "flash_attn/flash_fwd_hdim256_bf16_sm80.cu",
                "flash_attn/flash_fwd_split_hdim256_bf16_sm80.cu",
            ]
            .map(|source| fa2_root.join(source)),
        );
        fa2_sources.extend(
            [
                "fa2_fwd_hdim96_f16.cu",
                "fa2_fwd_hdim96_bf16.cu",
                "fa2_fwd_hdim128_extra.cu",
                "fa2_fwd_hdim256_extra.cu",
                "fa2_fwd_split_hdim256_f16.cu",
            ]
            .map(|source| attention_kernel_root.join(source)),
        );
    }
    let mut fa2_e4m3_sources = Vec::new();
    if selection
        .targets
        .iter()
        .any(|target| is_cutlass_sm100_family(&target.cutlass_arch))
    {
        fa2_e4m3_sources.extend(
            ["fa2_f16_e4m3_522.cu", "fa2_fwd_hdim256_e4m3.cu"]
                .map(|source| attention_kernel_root.join(source)),
        );
    }
    let patched_fa2_root =
        (!fa2_e4m3_sources.is_empty()).then(|| stage_patched_fa2(&native, &fa2_root, &out));
    assert!(
        generic_sources
            .iter()
            .chain(&cutlass_sources)
            .chain(&sm110_sources)
            .chain(&cutlass_sm87_sources)
            .chain(&cutlass_sm89_sources)
            .chain(&fa2_sources)
            .chain(&fa2_e4m3_sources)
            .all(|path| path.is_file()),
        "native build source is missing"
    );

    let cuda_includes = [
        PathBuf::from(format!("{cuda}/include")),
        PathBuf::from(format!("{cuda}/targets/{cpu_arch}/include")),
        PathBuf::from(format!("{cuda}/targets/aarch64-linux/include")),
        PathBuf::from(format!("{cuda}/thor/targets/aarch64-linux/include")),
    ];
    let cutlass_includes = [
        cutlass_root.clone(),
        cutlass_root.join("fmha"),
        cutlass_root.join("include"),
        cutlass_root.join("tools/util/include"),
    ];
    let has_cutlass = !cutlass_sources.is_empty();
    let has_cutlass_sm87 = !cutlass_sm87_sources.is_empty();
    let has_cutlass_sm89 = !cutlass_sm89_sources.is_empty();
    let generic_codegen = gencode_args(
        selection
            .targets
            .iter()
            .map(|target| target.nvcc_arch.clone()),
    );
    let cutlass_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| is_cutlass_sm100_family(&target.cutlass_arch))
            .map(|target| target.cutlass_arch.clone()),
    );
    let sm110_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| target.sm() == 110)
            .map(|target| target.cutlass_arch.clone()),
    );
    let cutlass_sm89_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| target.cutlass_arch == "sm_89")
            .map(|target| target.cutlass_arch.clone()),
    );
    let cutlass_sm87_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| target.cutlass_arch == "sm_87")
            .map(|target| target.cutlass_arch.clone()),
    );
    let fa2_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| target.sm() >= 80)
            .map(|target| target.nvcc_arch.clone()),
    );
    let mut objects = Vec::new();
    let workers = nvcc_build::parallelism();
    let mut jobs: Vec<nvcc_build::CompileJob> = Vec::new();
    for (index, source) in generic_sources
        .drain(..)
        .map(|source| (source, false, false, false, false, false, false))
        .chain(
            cutlass_sources
                .into_iter()
                .map(|source| (source, true, false, false, false, false, false)),
        )
        .chain(
            sm110_sources
                .into_iter()
                .map(|source| (source, false, true, false, false, false, false)),
        )
        .chain(
            cutlass_sm87_sources
                .into_iter()
                .map(|source| (source, false, false, true, false, false, false)),
        )
        .chain(
            cutlass_sm89_sources
                .into_iter()
                .map(|source| (source, false, false, false, true, false, false)),
        )
        .chain(
            fa2_sources
                .into_iter()
                .map(|source| (source, false, false, false, false, true, false)),
        )
        .chain(
            fa2_e4m3_sources
                .into_iter()
                .map(|source| (source, false, false, false, false, false, true)),
        )
        .enumerate()
    {
        let (
            source,
            is_cutlass,
            is_sm110,
            is_cutlass_sm87,
            is_cutlass_sm89,
            is_fa2,
            is_fa2_e4m3,
        ) = source;
        let object = out.join(format!(
            "gemm-{index}-{}.o",
            source.file_stem().unwrap().to_string_lossy()
        ));
        let mut command = Command::new(&bundled_nvcc);
        command
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .args(["--compiler-options", "-fPIC", "-O3", "-std=c++17"])
            .arg(format!("-I{}", native.join("include").display()))
            .arg(format!("-I{}", out.display()));
        // The build IDs fold a whole-tree fingerprint, so they change on any
        // kernel edit. Only the two tuning-key translation units read them;
        // putting them on every command line would invalidate the whole object
        // cache whenever one source file moves.
        if source.file_name().is_some_and(|name| name == "tuning_key.cpp") {
            if source.to_string_lossy().contains("/gemm/") {
                command.arg(format!("-DAPXINF_GEMM_BUILD_ID=\"{id}\""));
            } else {
                command.arg(format!("-DAPXINF_ATTENTION_BUILD_ID=\"{attention_id}\""));
            }
        }
        command.args(if is_sm110 {
            &sm110_codegen
        } else if is_cutlass || is_fa2_e4m3 {
            &cutlass_codegen
        } else if is_cutlass_sm87 {
            &cutlass_sm87_codegen
        } else if is_cutlass_sm89 {
            &cutlass_sm89_codegen
        } else if is_fa2 {
            &fa2_codegen
        } else {
            &generic_codegen
        });
        for include in cuda_includes.iter().filter(|path| path.is_dir()) {
            command.arg(format!("-I{}", include.display()));
        }
        if has_cutlass {
            command.arg("-DAPXINF_GEMM_CUTLASS=1");
            command.arg("-DAPXINF_ATTENTION_CUTLASS=1");
        }
        if nvfp4_gemm_swiglu {
            command.arg("-DAPXINF_NVFP4_GEMM_SWIGLU=1");
        }
        if has_cutlass_sm89 {
            command.arg("-DAPXINF_GEMM_CUTLASS_SM89=1");
        }
        if has_cutlass_sm87 {
            command.arg("-DAPXINF_GEMM_CUTLASS_SM87_W8A8=1");
        }
        if has_fa2 {
            command.arg("-DAPXINF_ATTENTION_FA2=1");
        }
        if !cutlass_codegen.is_empty() {
            command.arg("-DAPXINF_ATTENTION_FA2_E4M3=1");
        }
        if is_cutlass || is_sm110 || is_cutlass_sm87 || is_cutlass_sm89 {
            command.args(["--expt-relaxed-constexpr", "--expt-extended-lambda"]);
            // `nvcc -O3` does not imply `-DNDEBUG`, so every `CUTLASS_ASSERT`
            // survives into device code (97 `BPT.TRAP` instructions in the
            // FMHA family before this). Scoped to the CUTLASS translation
            // units so the other adapters keep their asserts.
            command.arg("-DNDEBUG");
            for include in &cutlass_includes {
                command.arg(format!("-I{}", include.display()));
            }
            if source
                .file_name()
                .is_some_and(|name| name == "gemm_e4m3_geglu_interleaved_sm100.cu")
            {
                command.arg("-DAPXINF_FP8_DUAL_GEGLU_PRODUCTION=1");
            }
            if source
                .file_name()
                .is_some_and(|name| name == "gemm_bf16_geglu_interleaved_sm100.cu")
            {
                command.arg("-DAPXINF_BF16_DUAL_GEGLU_PRODUCTION=1");
            }
            let file_name = source.file_name().and_then(|name| name.to_str());
            const D256_COMMON: &[&str] = &[
                "-DAPXINF_EX2_APPROX=1",
                "-DAPXINF_EX2_FTZ=1",
                "-DAPXINF_DEBUG_SHOW_SO=0",
                "-DAPXINF_FUSE_ROWSUM=1",
                "-DAPXINF_P_FLUSH_HALF=1",
                "-DAPXINF_CORR_RING2=1",
                "-DAPXINF_LOAD_SHAPE_PER_MODE=1",
            ];
            if file_name == Some("fmha_d256_causal_sm110.cu") {
                command.args(D256_COMMON).args([
                    "-DAPXINF_EX2_EMU_FREQ=14",
                    "-DAPXINF_EX2_EMU_RES=6",
                    "-DAPXINF_EX2_EMU_START_FRG=0",
                    "-DAPXINF_RESCALE_THRESHOLD=0.0f",
                    "-DAPXINF_KV_DESCENDING=0",
                    "-DAPXINF_ROWSUM_QUARTERS=1",
                    "-DAPXINF_ACC_SCALE_HALF=1",
                    "-DAPXINF_ROWSUM_RESCALE_RN=1",
                    "-DAPXINF_FINAL_NORM=3",
                    "-DAPXINF_S_PIPE_STAGES=2",
                ]);
            }
            if file_name == Some("fmha_d256_split_batch_sm110.cu") {
                command.args(D256_COMMON).args([
                    "-DAPXINF_EX2_EMU_RES=0",
                    "-DAPXINF_RESCALE_THRESHOLD=8.0f",
                    "-DAPXINF_KV_DESCENDING=1",
                    "-DAPXINF_ROWSUM_QUARTERS=0",
                    "-DAPXINF_ACC_SCALE_HALF=0",
                    "-DAPXINF_ROWSUM_RESCALE_RN=0",
                    "-DAPXINF_FINAL_NORM=2",
                    "-DAPXINF_LSE_FA4=1",
                ]);
            }
            if file_name == Some("fmha_strided_groups_sm110.cu") {
                command.args([
                    "-DAPXINF_MASK_R2P=1",
                    "-DAPXINF_KV_DESCENDING=1",
                    "-DAPXINF_RESCALE_THRESHOLD=8.0f",
                    "-DAPXINF_EX2_APPROX=1",
                    "-DAPXINF_EX2_FTZ=1",
                    "-DAPXINF_DEBUG_SHOW_SO=0",
                    "-DAPXINF_FUSE_ROWSUM=1",
                    "-DAPXINF_EX2_EMU_FREQ=16",
                    "-DAPXINF_EX2_EMU_RES=4",
                    "-DAPXINF_EX2_EMU_START_FRG=1",
                    "-DAPXINF_ACC_SCALE_HALF=0",
                    "-DAPXINF_FINAL_NORM=1",
                    "-DAPXINF_NO_LSE=1",
                    "-DAPXINF_PERSISTENT=1",
                    "-DAPXINF_NUM_REGS_SOFTMAX=200",
                    "-DAPXINF_NUM_REGS_CORRECTION=64",
                    "-DAPXINF_NUM_REGS_OTHER=48",
                    "-DAPXINF_FMHA_KV_STAGES=4",
                    "-DAPXINF_SKIP_PAD_QSTAGE=1",
                    "-DAPXINF_P_CHUNKS16=1",
                    "-DAPXINF_ROWSUM_RESCALE_RN=1",
                ]);
            }
            if file_name == Some("gdn_bf16_prefill_sm110_launch.cu") {
                command.args([
                    "-DHEAD_GROUP_LOG2=1",
                    "-DNUM_O_HEADS_LOG2=5",
                    "-DIS_GQA=0",
                ]);
            }
        }
        if is_fa2 || is_fa2_e4m3 {
            command.args([
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ]);
            // The runtime never enables these FA2 feature axes. Keep the
            // legacy compile-time pruning so every compiled specialization
            // matches the contract and split-KV does not instantiate an
            // unused combinatorial kernel set. The shared constant makes the
            // exact compile policy part of the Attention build fingerprint.
            command.args(attention_fingerprint::FA2_FIXED_FEATURE_DEFINES);
            command.arg(if is_fa2_e4m3 {
                "-DFLASH_NAMESPACE=apxinf_cuda_new_fa2_direct_e4m3"
            } else {
                "-DFLASH_NAMESPACE=apxinf_cuda_new_fa2"
            });
            if is_fa2_e4m3 {
                command.arg("-DAPXINF_FA2_DIRECT_E4M3=1");
            }
            command.arg(format!("-I{}", fa2_compat_root.display()));
            if is_fa2_e4m3 {
                command.arg(format!(
                    "-I{}",
                    patched_fa2_root
                        .as_ref()
                        .expect("E4M3 FA2 staging must exist")
                        .display()
                ));
            }
            command.arg(format!("-I{}", fa2_root.display()));
            command.arg(format!("-I{}", cutlass_root.join("include").display()));
        }
        // Skip the compile when a previous run produced a current object: the
        // recorded command line is identical and every dependency is older.
        // The misses join the worker pool after the loop.
        let stem = source.file_stem().unwrap().to_string_lossy().to_string();
        let deps = out.join(format!("gemm-{index}-{stem}.d"));
        let stamp = out.join(format!("gemm-{index}-{stem}.cmdline"));
        let cmdline = nvcc_build::describe_command(&command);
        if nvcc_build::object_is_current(&object, &deps, &stamp, &cmdline) {
            objects.push(object);
            continue;
        }
        command.arg("-MD").arg("-MF").arg(&deps);
        jobs.push(nvcc_build::CompileJob {
            command,
            action: format!("compile {}", source.display()),
            stamp,
            cmdline,
        });
        objects.push(object);
    }
    nvcc_build::run_parallel(jobs, workers);

    let archive = out.join("libapxinf_gemm_native.a");
    let _ = std::fs::remove_file(&archive);
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&archive).args(&objects);
    run(&mut ar, "archive GEMM native objects");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=apxinf_gemm_native");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cudart");
    // The vendored FlashInfer GDN prefill builds its TMA descriptors with
    // cuTensorMapEncodeTiled, which lives in the driver library rather than
    // the runtime.
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=stdc++");
}
