use std::env;
fn main() {
    println!("cargo:rerun-if-changed=native/bridge.mm");
    println!("cargo:rerun-if-changed=native/bridge.h");
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }
    // Feature unification on Linux/CUDA must not require an Apple SDK; the
    // runtime module uses the same platform guard.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("native/bridge.mm")
        .flag("-fobjc-arc")
        .flag("-fvisibility=hidden")
        .compile("apxinf_coreml_bridge");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=dylib=c++");
}
