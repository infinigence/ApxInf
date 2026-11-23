//! MTTX (Moore Threads Tools Extension) range markers for msys profiling.
//!
//! Wrap a code region in `let _g = mttx::range("name");` and it appears as a
//! labeled range in the msys timeline. The guard pops on drop. This is the
//! MUSA counterpart of `apxinf-cuda`'s `nvtx` module, and the model layer
//! reaches both through the one neutral name `profiling::trace::range`.
//!
//! MTTX is NVTX's client protocol under a different name: a header-only client
//! `dlopen`s `$MTTX_INJECTION64_PATH` on its first API call and hands the
//! process's export table to whatever is behind it. With no tool attached every
//! call is a no-op **for the life of the process** — the initialisation runs
//! once and never retries — so a capture taken without the injection library
//! silently carries no labels at all. Nothing here reports that: the answer is
//! the capture itself, where a report with no `type=3` events has no ranges in
//! it whatever the program pushed.
//!
//! The first range in a process is what runs that initialisation, so it is the
//! one that cannot be skipped. Once it has answered, a process with nothing
//! attached pays one relaxed atomic load per range and nothing else — no
//! allocation, no FFI — which is what keeps the op-level labels, thousands of
//! them per inference, free when nobody is listening.
//!
//! Disable the `mttx` cargo feature, or build where no `mt-tools-extension`
//! checkout was found (see `build_support/mttx.rs`), and `imp` becomes no-op
//! stubs: the surface stays, the cost and the C object both go.

use std::ffi::CString;
use std::sync::atomic::{AtomicI32, Ordering};

/// The C ABI `src/mttx_client.c` re-exports. The client's own API is `inline
/// static` with hidden visibility, so it has no symbols a linker could resolve
/// and this shim is the only way to reach it from Rust.
#[cfg(all(feature = "mttx", mttx_available))]
mod imp {
    use std::os::raw::c_char;

    extern "C" {
        /// Opens a default-domain range. Non-zero when a tool took it.
        pub fn apxinf_mttx_push(name: *const c_char) -> i32;
        /// Closes the innermost range on this thread.
        pub fn apxinf_mttx_pop() -> i32;
        /// `-1` not initialised, `0` no tool, `1` tool attached. Pure read:
        /// it inspects the callback the tool installed rather than asking the
        /// client to install one.
        pub fn apxinf_mttx_state() -> i32;
    }
}

/// No-op stubs for a build without the feature, or without MTTX headers to
/// compile the client against.
#[cfg(not(all(feature = "mttx", mttx_available)))]
mod imp {
    use std::os::raw::c_char;

    /// No-op stub: nothing here can reach a profiler.
    pub unsafe fn apxinf_mttx_push(_name: *const c_char) -> i32 {
        0
    }
    /// No-op stub — see `apxinf_mttx_push`.
    pub unsafe fn apxinf_mttx_pop() -> i32 {
        0
    }
    /// No-op stub: a build without the client has nothing to be attached.
    pub unsafe fn apxinf_mttx_state() -> i32 {
        0
    }
}

/// What the client last said: `-1` before the first range of the process has
/// run the client's one-shot initialisation, then `0` for no tool or `1` for a
/// tool attached. Final once known — the client never retries — so a process
/// that has ruled a tool out never asks again.
static ATTACHED: AtomicI32 = AtomicI32::new(-1);

/// A named range, live from construction to drop.
///
/// The MTTX range stack is **per thread**, so this guard is deliberately
/// `!Send`: dropping it on another thread would pop whatever that thread had
/// open instead of the range this guard pushed.
pub struct Range {
    /// `Some` only while a range is actually open. The name is kept alive
    /// because the client reads it during the push and keeps no copy; a guard
    /// whose push was refused holds `None` so its drop cannot pop a range it
    /// never opened, which would close the enclosing one instead.
    _name: Option<CString>,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Range {
    pub fn new(name: &str) -> Self {
        // Only skip once a tool has been ruled out. The first range of the
        // process is what rules it out, so it must go through.
        if ATTACHED.load(Ordering::Relaxed) == 0 {
            return Self {
                _name: None,
                _not_send: std::marker::PhantomData,
            };
        }
        // A range name with an interior NUL cannot cross the C boundary; label
        // it rather than dropping the marker, since the tool is attached.
        let c = CString::new(name).unwrap_or_else(|_| c"mttx".to_owned());
        // An attached tool can still refuse the push, by having push/pop
        // tracking turned off.
        let pushed = unsafe { imp::apxinf_mttx_push(c.as_ptr()) } != 0;
        if ATTACHED.load(Ordering::Relaxed) < 0 {
            // That push ran the initialisation, so the client can be asked now.
            ATTACHED.store(unsafe { imp::apxinf_mttx_state() }, Ordering::Relaxed);
        }
        Self {
            _name: pushed.then_some(c),
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        if self._name.is_some() {
            unsafe {
                imp::apxinf_mttx_pop();
            }
        }
    }
}

/// Convenience: `let _g = mttx::range("name");` returns a guard that pops on
/// drop.
pub fn range(name: &str) -> Range {
    Range::new(name)
}
