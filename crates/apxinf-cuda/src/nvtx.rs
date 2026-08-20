//! NVTX (NVIDIA Tools Extension) range markers for nsys profiling.
//!
//! Wrap a code region in `let _g = nvtx::range("name");` and it appears as a
//! labeled range in the Nsight Systems timeline. The guard pops on drop.
//!
//! The lib name varies by CUDA distribution:
//!   * Desktop CUDA 12.x ships `libnvtx3interop.so` (NVTX v3 interop shim).
//!   * Drive OS 7 / embedded CUDA ships the classic `libnvToolsExt.so`
//!     (NVTX v2). Both export `nvtxRangePushA` / `nvtxRangePop`.
//! `build.rs` probes `$CUDA_PATH/lib64` and emits `--cfg nvtx_v3` or
//! `--cfg nvtx_v2` accordingly. Disable the `nvtx` cargo feature entirely
//! to compile out the FFI (no-op stubs) — useful when no NVTX lib is
//! available at all.

use std::ffi::CString;

#[cfg(all(feature = "nvtx", nvtx_v3))]
mod imp {
    use std::os::raw::c_char;

    #[link(name = "nvtx3interop")]
    extern "C" {
        pub fn nvtxRangePushA(name: *const c_char) -> i32;
        pub fn nvtxRangePop() -> i32;
    }
}

#[cfg(all(feature = "nvtx", nvtx_v2, not(nvtx_v3)))]
mod imp {
    use std::os::raw::c_char;

    #[link(name = "nvToolsExt")]
    extern "C" {
        pub fn nvtxRangePushA(name: *const c_char) -> i32;
        pub fn nvtxRangePop() -> i32;
    }
}

// Feature off, OR feature on but build.rs found no lib. In the latter case
// build.rs already warned; here we substitute no-op stubs so the crate
// still compiles and profiling markers become free.
#[cfg(any(not(feature = "nvtx"), all(not(nvtx_v3), not(nvtx_v2))))]
mod imp {
    use std::os::raw::c_char;

    /// No-op stub used when the `nvtx` cargo feature is disabled or no NVTX
    /// library was found on the target's CUDA path.
    pub unsafe fn nvtxRangePushA(_name: *const c_char) -> i32 {
        0
    }
    /// No-op stub — see `nvtxRangePushA`.
    pub unsafe fn nvtxRangePop() -> i32 {
        0
    }
}

/// RAII guard: pushes a range on construction, pops on drop.
pub struct Range {
    _name: CString, // kept alive for the lifetime of the guard
}

impl Range {
    pub fn new(name: &str) -> Self {
        let c = CString::new(name).unwrap_or_else(|_| CString::new("nvtx").unwrap());
        unsafe {
            imp::nvtxRangePushA(c.as_ptr());
        }
        Self { _name: c }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe {
            imp::nvtxRangePop();
        }
    }
}

/// Convenience: `let _g = nvtx::range("name");` returns a guard that pops
/// on drop.
pub fn range(name: &str) -> Range {
    Range::new(name)
}
