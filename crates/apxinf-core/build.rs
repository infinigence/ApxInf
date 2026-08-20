fn main() {
    // When the `accelerate` feature is enabled, link Apple's Accelerate framework
    // which provides the CBLAS implementation on macOS.
    if std::env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // When the `openblas` feature is enabled, link the system OpenBLAS library.
    // Requires libopenblas-dev (or equivalent) to be installed.
    if std::env::var("CARGO_FEATURE_OPENBLAS").is_ok() {
        let system_paths = [
            "/usr/lib/x86_64-linux-gnu/openblas-pthread",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib",
            "/usr/local/lib",
        ];
        for path in &system_paths {
            if std::path::Path::new(&format!("{path}/libopenblas.so")).exists()
                || std::path::Path::new(&format!("{path}/libopenblas.a")).exists()
            {
                println!("cargo:rustc-link-search={path}");
            }
        }
        println!("cargo:rustc-link-lib=openblas");
    }
}
