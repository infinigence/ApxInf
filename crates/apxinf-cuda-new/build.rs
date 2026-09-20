use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    std::fs::write(&path, header)
        .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
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
    if staged.exists() {
        std::fs::remove_dir_all(&staged)
            .unwrap_or_else(|error| panic!("remove {}: {error}", staged.display()));
    }
    copy_tree(&fa2_root.join("flash_attn"), &staged.join("flash_attn"));
    let patch = native
        .join("patches")
        .join("fa2-direct-e4m3-output.patch");
    let mut command = Command::new("patch");
    command
        .current_dir(&staged)
        .args(["--batch", "--forward", "-p0", "-i"])
        .arg(&patch);
    run(&mut command, "apply FA2 direct-E4M3 patch");
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
    ]
    .map(|source| adapters.join(source))
    .to_vec();
    generic_sources.push(native.join("tests/framework_backend.cu"));
    let cutlass_root = native.join("kernels/cutlass");
    let fa2_root = native.join("kernels/fa2");
    let fa2_compat_root = native.join("kernels/fa2_compat");
    let attention_kernel_root = native.join("kernels/attention");
    let mut cutlass_sources = Vec::new();
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
            ]
            .map(|source| operators.join(source)),
        );
        cutlass_sources.push(adapters.join("attention/providers/cutlass.cu"));
    }
    let has_fa2 = selection.targets.iter().any(|target| target.sm() >= 80);
    let mut fa2_sources = Vec::new();
    if has_fa2 {
        generic_sources.push(adapters.join("attention/providers/fa2.cpp"));
        fa2_sources.extend(
            [
                "fa2.cu",
                "flash_attn/flash_fwd_hdim128_bf16_sm80.cu",
                "flash_attn/flash_fwd_hdim256_bf16_sm80.cu",
            ]
            .map(|source| fa2_root.join(source)),
        );
        fa2_sources.extend(
            ["fa2_fwd_hdim128_extra.cu", "fa2_fwd_hdim256_extra.cu"]
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
    let patched_fa2_root = (!fa2_e4m3_sources.is_empty())
        .then(|| stage_patched_fa2(&native, &fa2_root, &out));
    assert!(
        generic_sources
            .iter()
            .chain(&cutlass_sources)
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
    let fa2_codegen = gencode_args(
        selection
            .targets
            .iter()
            .filter(|target| target.sm() >= 80)
            .map(|target| target.nvcc_arch.clone()),
    );
    let mut objects = Vec::new();
    for (index, source) in generic_sources
        .drain(..)
        .map(|source| (source, false, false, false))
        .chain(
            cutlass_sources
                .into_iter()
                .map(|source| (source, true, false, false)),
        )
        .chain(
            fa2_sources
                .into_iter()
                .map(|source| (source, false, true, false)),
        )
        .chain(
            fa2_e4m3_sources
                .into_iter()
                .map(|source| (source, false, false, true)),
        )
        .enumerate()
    {
        let (source, is_cutlass, is_fa2, is_fa2_e4m3) = source;
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
            .arg(format!("-I{}", out.display()))
            .arg(format!("-DAPXINF_GEMM_BUILD_ID=\"{id}\""))
            .arg(format!("-DAPXINF_ATTENTION_BUILD_ID=\"{attention_id}\""));
        command.args(if is_cutlass || is_fa2_e4m3 {
            &cutlass_codegen
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
        if has_fa2 {
            command.arg("-DAPXINF_ATTENTION_FA2=1");
        }
        if !cutlass_codegen.is_empty() {
            command.arg("-DAPXINF_ATTENTION_FA2_E4M3=1");
        }
        if is_cutlass {
            command.args(["--expt-relaxed-constexpr", "--expt-extended-lambda"]);
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
            command.arg(if is_fa2_e4m3 {
                "-DFLASH_NAMESPACE=apxinf_fa2_direct_e4m3"
            } else {
                "-DFLASH_NAMESPACE=apxinf_fa2"
            });
            if is_fa2_e4m3 {
                command.args([
                    "-DAPXINF_FA2_DIRECT_E4M3=1",
                    "-DFLASHATTENTION_DISABLE_DROPOUT",
                    "-DFLASHATTENTION_DISABLE_ALIBI",
                    "-DFLASHATTENTION_DISABLE_SOFTCAP",
                    "-DFLASHATTENTION_DISABLE_LOCAL",
                ]);
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
        run(&mut command, &format!("compile {}", source.display()));
        objects.push(object);
    }

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
    println!("cargo:rustc-link-lib=stdc++");
}
