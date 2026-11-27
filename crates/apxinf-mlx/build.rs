use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_ROOT");
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    println!("cargo:rerun-if-changed=native/bridge.h");
    println!("cargo:rerun-if-changed=native/signpost.c");
    if env::var_os("CARGO_FEATURE_SIGNPOST").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
    {
        cc::Build::new()
            .file("native/signpost.c")
            .flag("-fvisibility=hidden")
            .compile("apxinf_mlx_signpost");
    }
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    // Feature unification on Linux/CUDA must not require an Apple SDK.
    // Rust modules/registrations use the same platform guard.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    let root = PathBuf::from(
        env::var_os("MLX_ROOT")
            .expect("set MLX_ROOT to the MLX 0.32.4 C++ SDK root (include/ and lib/)"),
    );
    println!("cargo:rerun-if-changed={}", root.join("include/mlx/version.h").display());
    let version = fs::read_to_string(root.join("include/mlx/version.h"))
        .expect("MLX_ROOT lacks include/mlx/version.h");
    let macro_value = |name: &str| {
        version.lines().find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("#define") && fields.next() == Some(name))
                .then(|| fields.next())
                .flatten()
        })
    };
    assert!(
        macro_value("MLX_VERSION_MAJOR") == Some("0")
            && macro_value("MLX_VERSION_MINOR") == Some("32")
            && macro_value("MLX_VERSION_PATCH") == Some("4"),
        "apxinf-mlx pins MLX 0.32.4 headers and library"
    );
    assert!(
        root.join("lib/libmlx.dylib").is_file(),
        "MLX_ROOT lacks lib/libmlx.dylib"
    );
    // Inline MLX's steel NAX attention headers into OUT_DIR for Metal JIT kernels
    // that do not receive an include search path at runtime.
    let include_root = root.join("include");
    let mut seen = std::collections::HashSet::new();
    let mut inlined = String::new();
    fn inline_header(
        include_root: &std::path::Path,
        rel: &str,
        seen: &mut std::collections::HashSet<String>,
        out: &mut String,
    ) {
        if !seen.insert(rel.to_string()) {
            return;
        }
        let path = include_root.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read MLX header {}: {e}", path.display()));
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "#pragma once" {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let rest = rest.trim();
                if let Some(inner) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                    inline_header(include_root, inner, seen, out);
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    inline_header(
        &include_root,
        "mlx/backend/metal/kernels/steel/attn/nax.h",
        &mut seen,
        &mut inlined,
    );
    inline_header(
        &include_root,
        "mlx/backend/metal/kernels/steel/utils.h",
        &mut seen,
        &mut inlined,
    );
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    fs::write(out_dir.join("steel_attn_nax_header.metal"), inlined)
        .expect("failed to write steel_attn_nax_header.metal");

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("native/bridge.cpp")
        .include(root.join("include"))
        .flag("-fvisibility=hidden")
        .compile("apxinf_mlx_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    // This rpath applies only to this crate's tests/examples. Downstream
    // executables still need DYLD_LIBRARY_PATH as documented in the README.
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        root.join("lib").display()
    );
}
