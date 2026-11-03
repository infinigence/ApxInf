//! Build script for apxinf-ascend.
//!
//! Links against the CANN ACL runtime libraries. The CANN install root is
//! discovered via the `ASCEND_HOME_PATH` environment variable (set by
//! `set_env.sh` / CANN's `env.sh`). If unset, falls back to the standard
//! `/usr/local/Ascend/cann-9.1.0/x86_64-linux` path on the 950PR.
//!
//! Three libraries are linked:
//! - `ascendcl`  — ACL Runtime (aclInit, aclrt*, aclmdl*)
//! - `nnopbase`  — aclnn base types (aclTensor, aclScalar, aclOpExecutor)
//! - `opapi`     — aclnn operator implementations (aclnnRmsNorm, aclnnMatmul, …)

fn main() {
    let cann_root = std::env::var("ASCEND_HOME_PATH")
        .unwrap_or_else(|_| "/usr/local/Ascend/cann-9.1.0/x86_64-linux".to_string());

    let lib_dir = format!("{cann_root}/lib64");

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=ascendcl");
    println!("cargo:rustc-link-lib=nnopbase");
    println!("cargo:rustc-link-lib=opapi");

    // Re-run if the CANN install changes.
    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
}
