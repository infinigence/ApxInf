use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build_support/cuda_arch.rs"]
mod cuda_arch;

use cuda_arch::{
    gencode_args, is_cutlass_sm100_family, select_cuda_arch, target_features, ArchSelection,
    ArchSource,
};

const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

fn hash_bytes(hash: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u128::from(*byte);
        *hash = hash.wrapping_mul(FNV1A_128_PRIME);
    }
}

fn collect_inputs(root: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_inputs(&path, files);
        } else if path.extension().is_some_and(|extension| {
            matches!(
                extension.to_string_lossy().as_ref(),
                "cu" | "cuh" | "h" | "hh" | "hpp"
            )
        }) {
            files.push(path);
        }
    }
}

fn build_id(native: &Path, target: &str, selection: &ArchSelection) -> String {
    let mut hash = FNV1A_128_OFFSET;
    for value in ["apxinf-gemm-pilot-v1", env!("CARGO_PKG_VERSION"), target] {
        hash_bytes(&mut hash, value.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    for arch in &selection.targets {
        hash_bytes(&mut hash, arch.nvcc_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
        hash_bytes(&mut hash, arch.cutlass_arch.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }
    let mut files = Vec::new();
    collect_inputs(native, &mut files);
    files.sort_unstable();
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
    format!("gemm-kb1-{hash:032x}")
}

fn write_arch_header(out: &Path, selection: &ArchSelection) -> PathBuf {
    let path = out.join("apxinf_cuda_arches.h");
    let mut header = String::from(
        "#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace apxinf::gemm {\n\
         constexpr uint64_t kDeviceFeatureNativeFp8 = UINT64_C(1) << 0;\n\
         constexpr uint64_t kDeviceFeatureCutlassSm100 = UINT64_C(1) << 1;\n\
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

    let id = env::var("APXINF_KERNEL_BUILD_ID")
        .unwrap_or_else(|_| build_id(&native, &target, &selection));
    assert!(
        !id.chars()
            .any(|character| matches!(character, '\n' | '\r' | '"')),
        "invalid kernel build ID"
    );
    println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={id}");
    write_arch_header(&out, &selection);

    let adapters = native.join("adapters");
    let mut generic_sources = [
        "runtime.cu",
        "gemm/registry.cu",
        "gemm/tuning_key.cu",
        "gemm/tuning_db.cu",
        "gemm/autotune.cu",
        "gemm/reference.cu",
        "gemm/execution.cu",
        "gemm/providers/cublas.cu",
        "gemm/providers/cublaslt.cu",
        "gemm/providers/cutlass.cu",
        "gemm/providers/custom.cu",
    ]
    .map(|source| adapters.join(source))
    .to_vec();
    let cutlass_root = native.join("kernels/cutlass");
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
    }
    assert!(
        generic_sources
            .iter()
            .chain(&cutlass_sources)
            .all(|path| path.is_file()),
        "GEMM build source is missing"
    );

    let cuda_includes = [
        PathBuf::from(format!("{cuda}/include")),
        PathBuf::from(format!("{cuda}/targets/{cpu_arch}/include")),
        PathBuf::from(format!("{cuda}/targets/aarch64-linux/include")),
        PathBuf::from(format!("{cuda}/thor/targets/aarch64-linux/include")),
    ];
    let cutlass_includes = [
        cutlass_root.clone(),
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
    let mut objects = Vec::new();
    for (index, source) in generic_sources
        .drain(..)
        .map(|source| (source, false))
        .chain(cutlass_sources.into_iter().map(|source| (source, true)))
        .enumerate()
    {
        let (source, is_cutlass) = source;
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
            .arg(format!("-DAPXINF_GEMM_BUILD_ID=\"{id}\""));
        command.args(if is_cutlass {
            &cutlass_codegen
        } else {
            &generic_codegen
        });
        for include in cuda_includes.iter().filter(|path| path.is_dir()) {
            command.arg(format!("-I{}", include.display()));
        }
        if has_cutlass {
            command.arg("-DAPXINF_GEMM_CUTLASS=1");
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
