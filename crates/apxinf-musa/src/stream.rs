//! A MUSA stream, owned by the context that created it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

use apxinf_core::{Error, Result};

use crate::ffi;

/// The handle is atomic only because the context lends the stream out as
/// `&MusaStream` and one state — an invalidated capture — has to be left by
/// taking the handle away. Nothing here is used concurrently: the same
/// thread-affinity that lets this type claim `Send`/`Sync` keeps the swap
/// between two calls rather than inside one.
#[derive(Debug)]
pub struct MusaStream {
    handle: AtomicPtr<c_void>,
}

// SAFETY: a MUSA stream handle is an opaque driver object; the runtime
// serialises access to it, and ApxInf only ever uses it from the thread that
// owns the context.
unsafe impl Send for MusaStream {}
unsafe impl Sync for MusaStream {}

impl MusaStream {
    /// Create the driver stream on whichever device is current; callers bind
    /// the device first.
    fn create() -> Result<*mut c_void> {
        let mut handle = std::ptr::null_mut();
        unsafe { ffi::check_runtime(ffi::musa::musaStreamCreate(&mut handle)) }
            .map_err(Error::Other)?;
        Ok(handle)
    }

    pub fn new() -> Result<Self> {
        Ok(Self {
            handle: AtomicPtr::new(Self::create()?),
        })
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle.load(Ordering::Acquire)
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { ffi::check_runtime(ffi::musa::musaStreamSynchronize(self.handle())) }
            .map_err(Error::Other)
    }

    /// Continue on a fresh stream, releasing this one exactly as [`Drop`] does.
    ///
    /// For the one state a stream cannot leave by itself: a capture the runtime
    /// invalidated. `musaStreamEndCapture` then answers `901` and leaves the
    /// stream in `INVALIDATED`, where every later launch is refused and even
    /// `musaDeviceSynchronize` fails for as long as it exists. Nothing clears
    /// that stream state — not retrying the end, not emptying the error slot —
    /// and `musaStreamDestroy` on it neither fails nor waits, so replacement is
    /// the only way back to a usable context. The slot does have to be emptied
    /// *as well*: the failure left a status in it, and every adapter here
    /// reports `musaGetLastError()`. That is the half CUDA's
    /// `clear_capture_error` covers; on MUSA it is necessary but not enough.
    /// Measured on M1000.
    ///
    /// Nothing prepared for the old stream is lost with it. The vendor adapters
    /// key their handles by stream and rebuild them on first use, driver
    /// allocations outlive the stream that made them, and the reuse list for
    /// this handle is dropped with it. Blocks *dropped* after the swap go back
    /// onto a list keyed to the retired handle, which can only be handed out
    /// again if the driver recycles the number — and the capture that forced
    /// this synchronised before it opened, so the retired stream never executed
    /// the record it was building.
    pub fn replace(&self) -> Result<()> {
        let fresh = Self::create()?;
        self.release(self.handle.swap(fresh, Ordering::AcqRel));
        // What the failed end left in the slot, and whatever the destroy added,
        // describe a stream that no longer exists; the next launch on the
        // replacement would inherit it as its own verdict.
        unsafe {
            ffi::musa::musaGetLastError();
        }
        Ok(())
    }

    /// Release everything this stream owns, in the order that keeps the driver
    /// from recycling the handle underneath the caches that are keyed on it.
    fn release(&self, handle: *mut c_void) {
        if handle.is_null() {
            return;
        }
        // Before the stream goes: the free list is keyed on this handle, and
        // the driver recycles handles, so blocks left keyed to it would be
        // handed to whatever stream gets the number next.
        crate::buffer::cache_drop_stream(handle as usize);
        unsafe {
            #[cfg(apxinf_musa_available)]
            ffi::custom::apxinf_musa_mublas_release_stream(handle);
            #[cfg(apxinf_musa_available)]
            ffi::custom::apxinf_musa_mudnn_release_stream(handle);
            let _ = ffi::musa::musaStreamDestroy(handle);
        }
    }
}

impl Drop for MusaStream {
    fn drop(&mut self) {
        let handle = *self.handle.get_mut();
        self.release(handle);
    }
}
