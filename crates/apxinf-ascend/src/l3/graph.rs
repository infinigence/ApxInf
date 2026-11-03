//! Graph capture for Ascend NPU.
//!
//! Wraps `aclmdlRICaptureBegin` / `aclmdlRICaptureEnd` / `aclmdlRIExecuteAsync`.
//!
//! During capture, the workspace session is activated so that `run_op` calls
//! sub-allocate from the pool and skip per-op sync. The captured graph
//! encodes the full execution sequence; replay invokes a single
//! `aclmdlRIExecuteAsync` instead of ~4761 per-op dispatches.

use apxinf_core::{Error, Graph, Result};

use crate::l3::context::AscendNewContext;
use crate::l3::workspace::ExecutionSession;

/// A captured Ascend execution graph.
pub struct CapturedGraph {
    model_ri: crate::ffi::acl::aclmdlRI,
    stream: crate::ffi::acl::aclrtStream,
}

impl CapturedGraph {
    /// Replay the captured graph asynchronously on the capture stream.
    pub fn replay(&self) -> Result<()> {
        let ret = unsafe {
            crate::ffi::acl::aclmdlRIExecuteAsync(self.model_ri, self.stream)
        };
        crate::ffi::acl::check(ret, "aclmdlRIExecuteAsync")
            .map_err(|e| Error::Other(e))
    }
}

impl Graph for CapturedGraph {
    fn replay(&self) -> Result<()> {
        Self::replay(self)
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        if !self.model_ri.is_null() {
            unsafe {
                let _ = crate::ffi::acl::aclmdlRIDestroy(self.model_ri);
            }
        }
    }
}

/// Capture a computation graph.
///
/// Activates the workspace session (pool + skip-sync), begins RI capture,
/// runs `operation`, then ends capture to produce a `CapturedGraph`.
///
/// Returns both the captured graph and the operation's output. The output
/// tensor's device buffer is baked into the graph — after `replay()`, it
/// contains the new computation result.
pub fn capture<T>(
    ctx: &AscendNewContext,
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(CapturedGraph, T)> {
    let stream = ctx.raw_stream();

    // Activate session so run_op uses pool + skip sync during capture.
    let _session_guard = unsafe { crate::enter_session(session.pool()) };
    // Mark the thread as capturing so pool-overflow outputs are registered
    // with the pool (freed with the graph) instead of owned by their tensors.
    crate::set_capture_active(true);

    let ret = unsafe { crate::ffi::acl::aclmdlRICaptureBegin(stream, 0) };
    if ret != 0 {
        // Capture did not start — restore eager (tensor-owned) overflow
        // semantics for any subsequent forwarded computations.
        crate::set_capture_active(false);
    }
    crate::ffi::acl::check(ret, "aclmdlRICaptureBegin")
        .map_err(|e| Error::Other(e))?;

    let result = operation();

    // Note: Do NOT call aclrtSynchronizeStream here — it is forbidden
    // during RI capture (causes error 107027). The session's skip-sync
    // flag ensures all ops are async, and aclmdlRICaptureEnd handles
    // stream synchronization internally.

    let mut model_ri: crate::ffi::acl::aclmdlRI = std::ptr::null_mut();
    let ret = unsafe {
        crate::ffi::acl::aclmdlRICaptureEnd(stream, &mut model_ri)
    };
    // Capture has ended one way or another — restore eager overflow semantics.
    crate::set_capture_active(false);
    crate::ffi::acl::check(ret, "aclmdlRICaptureEnd")
        .map_err(|e| Error::Other(e))?;

    let output = result?;

    if crate::diag_mem_enabled() {
        let (ov_cnt, ov_bytes) = crate::output_overflow_stats();
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[diag-mem] capture: pool.used={:.1}MB capacity={:.1}MB | overflow.allocs={ov_cnt} bytes={:.1}MB",
            mb(session.used()),
            mb(session.capacity()),
            mb(ov_bytes as usize),
        );
    }

    Ok((CapturedGraph { model_ri, stream }, output))
}
