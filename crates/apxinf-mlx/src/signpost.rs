//! os_signpost range markers for Instruments.
//!
//! Wrap a code region in `let _g = signpost::range("name");` and it appears as
//! an interval in Instruments' Points of Interest lane (subsystem `com.apxinf`,
//! signpost name `range`, start message = `name`). The guard ends the interval
//! on drop. This is the Apple counterpart of `apxinf-cuda-new`'s `nvtx` and
//! `apxinf-musa`'s `mttx`; model code reaches all three through
//! `profiling::trace::range`.
//!
//! Points of Interest signposts are recorded whether or not a tool is
//! attached, so every range costs an FFI call and a log write. That is why the
//! backend sits behind the `signpost` feature. On targets other than macOS the
//! feature compiles to no-op stubs.

#[cfg(target_os = "macos")]
mod imp {
    use std::os::raw::{c_char, c_int};

    extern "C" {
        pub fn apxinf_signpost_begin(name: *const c_char, len: c_int) -> u64;
        pub fn apxinf_signpost_end(id: u64);
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::os::raw::{c_char, c_int};

    pub unsafe fn apxinf_signpost_begin(_name: *const c_char, _len: c_int) -> u64 {
        0
    }
    pub unsafe fn apxinf_signpost_end(_id: u64) {}
}

/// A named interval, open from construction to drop.
///
/// Intervals are matched by a per-range id rather than a per-thread stack.
pub struct Range {
    id: u64,
}

impl Range {
    pub fn new(name: &str) -> Self {
        // The shim prints at most `len` bytes and never reads past them, so the
        // name needs no NUL terminator and no copy. An interior NUL ends the
        // printed name early.
        let len = name.len().min(i32::MAX as usize) as i32;
        let id = unsafe { imp::apxinf_signpost_begin(name.as_ptr().cast(), len) };
        Self { id }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe { imp::apxinf_signpost_end(self.id) }
    }
}

pub fn range(name: &str) -> Range {
    Range::new(name)
}
