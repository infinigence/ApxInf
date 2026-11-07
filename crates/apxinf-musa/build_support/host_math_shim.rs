//! Whether MUSA's `mcc` can get through this host's glibc math headers.
//!
//! `mcc` force-includes a runtime wrapper that reaches `<cmath>`, and glibc's
//! aarch64 `<bits/math-vector.h>`, which that pulls in, emits NEON and SVE
//! vector typedefs on a compiler-version test alone — without checking the
//! target's features. `mp_22` has neither, so every `-x musa` translation unit
//! fails against this host, including one with no includes of its own.
//! `host_math_shim/bits/math-vector.h` is a minimal replacement that omits those
//! typedefs, which is the outcome glibc's own header describes for a compiler
//! without vector types.
//!
//! This module answers the narrow question — *does this toolchain need the
//! shim* — by compiling rather than by inspecting a version, because the failure
//! depends on the host glibc as much as on the SDK. It runs only when there are
//! kernels to compile, so a build with none pays nothing.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::mcc;
use crate::musa_arch::MusaArch;

/// The shim, and why this toolchain needs it.
pub struct HostMathShim {
    /// The directory to hand to `-isystem`.
    pub include_dir: PathBuf,
    /// What went wrong without it, for the build log.
    pub reason: String,
}

/// The shim to add to the include path, or `None` when this toolchain compiles
/// without it.
///
/// Tries the compile without the shim first, so a toolchain that does not need
/// the workaround does not get it. `Err` means neither attempt compiled: a
/// kernel build that cannot start is a different thing from one that needs a
/// workaround, and conflating the two is what turns a broken toolchain into a
/// silently disabled backend.
pub fn resolve(
    mcc_path: &Path,
    arch: &MusaArch,
    include_dir_musa: &Path,
    manifest_dir: &Path,
) -> Result<Option<HostMathShim>, String> {
    let scratch = mcc::Scratch::new("host-math");
    let source = scratch.source(
        "probe.mu",
        "__global__ void apxinf_host_math_probe(float* p) { p[0] = 1.0f; }\n",
    )?;
    let object = scratch.object("probe");
    let extra = [
        OsString::from(format!("--offload-arch={}", arch.mcc_arch)),
        OsString::from(format!("-I{}", include_dir_musa.display())),
    ];

    let plain = mcc::compile(mcc_path, &extra, None, &source, &object)?;
    let plain_error = match plain {
        Ok(()) => return Ok(None),
        Err(error) => error,
    };

    let shim = mcc::shim(manifest_dir).ok_or_else(|| {
        format!(
            "mcc cannot compile MUSA code and this crate ships no shim to retry with: {}",
            mcc::first_line(&plain_error)
        )
    })?;
    match mcc::compile(mcc_path, &extra, Some(&shim), &source, &object)? {
        Ok(()) => Ok(Some(HostMathShim {
            include_dir: shim,
            reason: format!(
                "mcc cannot compile MUSA code against this host's glibc without the bundled \
                 <bits/math-vector.h> shim; compiling with it. First failure: {}",
                mcc::first_line(&plain_error)
            ),
        })),
        Err(error) => Err(format!(
            "mcc cannot compile MUSA code, with or without the bundled math-vector shim.\n\
             without it: {}\n\
             with it: {}",
            mcc::first_line(&plain_error),
            mcc::first_line(&error)
        )),
    }
}
