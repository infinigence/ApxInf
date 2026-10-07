use std::{env, fs, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=MLX_ROOT");
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    println!("cargo:rerun-if-changed=native/bridge.h");
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "macos",
        "apxinf-mlx requires Apple Silicon macOS"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        "aarch64",
        "apxinf-mlx requires Apple Silicon"
    );
    let root = PathBuf::from(
        env::var_os("MLX_ROOT")
            .expect("set MLX_ROOT to the MLX 0.31.2 C++ SDK root (include/ and lib/)"),
    );
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
            && macro_value("MLX_VERSION_MINOR") == Some("31")
            && macro_value("MLX_VERSION_PATCH") == Some("2"),
        "apxinf-mlx pins MLX 0.31.2 headers and library"
    );
    assert!(
        root.join("lib/libmlx.dylib").is_file(),
        "MLX_ROOT lacks lib/libmlx.dylib"
    );
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
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        root.join("lib").display()
    );
}
