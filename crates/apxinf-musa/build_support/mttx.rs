//! The MTTX client: where its headers are, and building the one translation
//! unit that speaks it.
//!
//! MUSA's profiler marker API is MTTX (`mt-tools-extension`), and it is
//! header-only — `mtToolsExt.h` carries the whole client, including the
//! `dlopen($MTTX_INJECTION64_PATH)` handshake that attaches a profiler tool.
//! This crate ships none of it: the headers live in a repository of their own,
//! so the build looks for a checkout and compiles `src/mttx_client.c` against
//! it.
//!
//! **Not finding a checkout is a soft failure.** `mttx.rs` falls back to no-op
//! stubs, the backend still builds and runs, and the build says so on stdout —
//! the same policy as a missing MUSA toolkit. **Failing to compile once the
//! headers were found is not soft**: that means this crate's own source is
//! wrong, and the failure mode of the marker API is silence, so a build that
//! reports success while the labels cannot work is worse than one that stops.
//!
//! `src/mttx_client.c` sits outside `adapters/`, `kernels/custom/` and
//! `kernels/asm/` on purpose. Those trees are hashed into
//! `APXINF_KERNEL_BUILD_ID`, which names the device code this build produces;
//! a host-side marker client changes no device code, and letting it move the id
//! would retire a valid device cache and bury a false difference in every later
//! "the id changed, so the kernels changed" reading.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::mcc;

/// Environment variables naming a `mt-tools-extension` checkout, most specific
/// first.
const SOURCE_VARIABLES: &[&str] = &["APXINF_MTTX_SRC", "MTTX_SRC"];

/// Directories inside a checkout that hold the client headers, since the
/// variable may name either the repository root or its `c/` directory.
const HEADER_SUBDIRS: &[&str] = &["c/include", "include"];

/// The header that has to be reachable, relative to an include directory.
const CLIENT_HEADER: &str = "mttx/mtToolsExt.h";

/// The translation unit that re-exports the client's C ABI for Rust.
const CLIENT_SOURCE: &str = "src/mttx_client.c";

/// The include directory holding the MTTX client headers, or why there is none.
///
/// The environment wins so a build can name its own checkout; the fallback is a
/// checkout beside the toolkit, and the message names every location tried
/// rather than only the variable.
pub fn resolve(musa_path: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    for variable in SOURCE_VARIABLES {
        if let Ok(value) = std::env::var(variable) {
            if !value.is_empty() {
                candidates.push(PathBuf::from(value));
            }
        }
    }
    candidates.push(musa_path.join("mt-tools-extension"));

    for candidate in &candidates {
        for subdir in HEADER_SUBDIRS {
            let include_dir = candidate.join(subdir);
            if include_dir.join(CLIENT_HEADER).is_file() {
                return Ok(include_dir);
            }
        }
    }

    Err(format!(
        "no MTTX client headers: none of {} holds {CLIENT_HEADER} (tried {}). \
         Point MTTX_SRC at a mt-tools-extension checkout",
        SOURCE_VARIABLES.join(" / "),
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Compile the client and hand the linker an archive holding it.
///
/// Returns the compiler's stderr when it fails: this is the one build step
/// here that reports a hard error, so the message is the whole diagnosis.
pub fn build(include_dir: &Path, manifest_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let source = manifest_dir.join(CLIENT_SOURCE);
    let object = out_dir.join("mttx_client.o");
    let archive = out_dir.join("libapxinf_musa_mttx.a");

    // File-level rerun triggers. The directory-level ones the kernel trees use
    // are enough for a tree that only ever gains files, but cargo watches a
    // directory's mtime, so editing this source in place would not rebuild it.
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-env-changed=CC");

    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-fPIC")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .arg(format!("-I{}", include_dir.display()))
        .output()
        .map_err(|error| format!("cannot run {compiler}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{compiler} failed on {}:\n{}",
            source.display(),
            mcc::first_line(&String::from_utf8_lossy(&output.stderr))
        ));
    }

    // `ar rcs` adds and replaces members but never removes one; this archive has
    // a single member that is always rewritten, so there is nothing stale to
    // clear first.
    let archived = Command::new("ar")
        .arg("rcs")
        .arg(&archive)
        .arg(&object)
        .status()
        .map_err(|error| format!("cannot run ar: {error}"))?;
    if !archived.success() {
        return Err(format!("ar failed to archive {}", object.display()));
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apxinf_musa_mttx");
    // The client reaches `dlopen` through the header's DLLOPEN macro. glibc 2.34
    // folded libdl into libc, where this is a no-op, but the symbol is real on
    // anything older.
    println!("cargo:rustc-link-lib=dl");
    Ok(())
}
