//! Whether `musaStreamGetDevice` is callable from this toolkit.
//!
//! `adapter_common.h` needs the device a *stream* belongs to, not the device
//! the host thread last selected, and the two can disagree. The header's
//! `MUSART_VERSION` cannot answer which one the toolkit offers: a 5.1 toolkit
//! defines `MUSART_VERSION` as 50100 while declaring and exporting
//! `musaStreamGetDevice`, so a version test is a false negative on exactly the
//! toolkits this port runs on. Compiling one call is the answer that matches
//! what the adapters will do with it.

use std::ffi::OsString;
use std::path::Path;

use super::host_math_shim::HostMathShim;
use super::mcc;
use super::musa_arch::MusaArch;

// Called from a host function on purpose: `musaStreamGetDevice` is `__host__`,
// so a kernel-body probe would fail to compile on every toolkit.
const PROBE: &str = r#"
#include <musa_runtime.h>

int apxinf_stream_device_probe(musaStream_t stream) {
    int device = -1;
    return musaStreamGetDevice(stream, &device) == musaSuccess ? device : -1;
}
"#;

/// Compile a call to `musaStreamGetDevice`; `false` means the adapters must
/// fall back to the thread's current selection.
pub fn available(
    mcc_path: &Path,
    arch: &MusaArch,
    include_dir_musa: &Path,
    shim: Option<&HostMathShim>,
) -> bool {
    let scratch = mcc::Scratch::new("stream-device");
    let source = match scratch.source("probe.mu", PROBE) {
        Ok(source) => source,
        Err(_) => return false,
    };
    let object = scratch.object("probe");
    let extra = [
        OsString::from(format!("--offload-arch={}", arch.mcc_arch)),
        OsString::from(format!("-I{}", include_dir_musa.display())),
        OsString::from("-std=c++17"),
    ];
    match mcc::compile(
        mcc_path,
        &extra,
        shim.map(|shim| shim.include_dir.as_path()),
        &source,
        &object,
    ) {
        Ok(Ok(())) => true,
        // A compiler that cannot run at all is reported by the shim probe that
        // runs before this one; here it only means "do not take the new path".
        Ok(Err(_)) | Err(_) => false,
    }
}
