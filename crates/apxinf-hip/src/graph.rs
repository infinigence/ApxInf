//! Capture a sequence of device work once and replay it as one launch.
//!
//! The model's forward issues about 2131 kernels per inference. Measured, the
//! GPU is busy 79.4 ms of an 81.2 ms inference, so the host-side launch time a
//! graph can remove is the 1.8 ms that is not hidden behind kernels. A graph
//! does not make any kernel faster: the small norms in GR00T's DiT run 41
//! blocks over 20 compute units and are bound by wave dispatch, and a replayed
//! dispatch is the same dispatch.
//!
//! What a graph needs from the work it records is strict, and the failure mode
//! when it is not met is silent rather than loud -- a replay reads addresses
//! recorded at capture time, so a buffer that moved produces a plausible wrong
//! answer. See [`HipGraph::capture`].

use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard, OnceLock};

use apxinf_core::{Error, Result};

use crate::ffi::{self, check};
use crate::runtime::HipContext;

/// Serializes capture against everything else on the device.
///
/// Capture is not confined to its stream. `hipMalloc` is a device-wide call
/// and fails while a capture is open, whatever stream opened it and whatever
/// capture mode was used -- so a thread allocating on an unrelated context
/// fails for the duration. This surfaced as a test that passed alone and
/// failed in the suite, which is the mild version; the severe version is an
/// allocation failing inside a concurrent inference.
///
/// Held for the whole capture, which also makes nested capture impossible.
///
/// What it does **not** do is stop a thread that never calls through here.
/// Any `hipMalloc` anywhere in the process -- another context, another
/// backend, a tensor allocated by unrelated code -- fails while a capture is
/// open. So capture is only safe when the caller knows nothing else is
/// allocating: in this crate that is the model's own `prepare`/`run` sequence,
/// which is single-threaded by construction.
///
/// `HipContext::new` also takes this lock while it initializes hipBLAS and
/// hipBLASLt: their init kernels run on the legacy default stream, and a
/// legacy-stream operation while a capture is open fails with error 906 --
/// which hipblaslt answers by exiting the process. Under the lock that race
/// blocks instead of dying. This also means constructing a `HipContext`
/// inside `HipGraph::capture`'s `work` would deadlock; the capture contract
/// already forbids allocating inside `work`, and a context is the largest
/// allocation there is.
pub(crate) fn capture_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Whether the calling thread is between `hipStreamBeginCapture` and
// `hipStreamEndCapture`. Thread-local, so a thread that is capturing can be
// distinguished from a thread that is merely waiting for another thread's
// capture to close.
thread_local! {
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Take the capture lock for a device allocation, or refuse.
///
/// `hipMalloc` fails device-wide while a capture is open (900). When another
/// thread is capturing, waiting for it to close is the correct behaviour --
/// the allocation is legal, just mistimed. When the *calling* thread is
/// capturing, the allocation is the contract violation `capture` documents,
/// and blocking would deadlock: the lock is already held by this thread, and
/// the capture will not end until this call returns. That case stays a loud
/// error.
pub(crate) fn allocation_exclusive() -> Result<MutexGuard<'static, ()>> {
    let capturing = CAPTURING.with(std::cell::Cell::get);
    if capturing {
        return Err(Error::Other(
            "apxinf-hip: device allocation inside a capture. A captured replay \
             reads addresses fixed at capture time, so `work` must not allocate; \
             see `HipGraph::capture`'s contract."
                .into(),
        ));
    }
    Ok(capture_lock())
}

/// An instantiated HIP graph, replayable on the context's stream.
pub struct HipGraph {
    exec: *mut c_void,
    nodes: usize,
}

// SAFETY: an executable graph is an opaque handle, usable from any host thread
// that owns it; this crate issues from one thread onto one stream.
unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl Drop for HipGraph {
    fn drop(&mut self) {
        // Nothing useful can be done with a failure while dropping.
        unsafe { ffi::apxinf_hip_graph_destroy(self.exec) };
    }
}

impl HipGraph {
    /// Record everything `work` issues, and return it as one launch.
    ///
    /// # What the caller must guarantee
    ///
    /// Capture records addresses and launch parameters, not intent. A replay
    /// repeats exactly what was recorded, so `work` must issue the same
    /// kernels against the same addresses every time it would have run:
    ///
    /// - **No device allocation.** `hipMalloc` during capture fails outright,
    ///   which is the loud case. The quiet case is an allocation that
    ///   succeeded at capture and whose address the graph now holds for work
    ///   that will read it after it has been freed and reused.
    /// - **No copy from host memory that is rebuilt per call.** The graph
    ///   records the source pointer. A `Vec` built inside `work` is gone by
    ///   the first replay.
    /// - **No host round-trip.** A download inside `work` reads a buffer the
    ///   replay has not written yet, and a synchronize inside a capture is an
    ///   error.
    /// - **The same arena layout.** `work` must run against an arena that was
    ///   reset to the same base, so every `tensor()` returns the address the
    ///   graph recorded. This is what a shape-keyed plan already gives.
    ///
    /// None of these can be checked here, which is why this is `unsafe`. What
    /// *is* checked is that the capture was not empty: an empty capture
    /// instantiates and replays successfully, leaving whatever was in the
    /// output buffer, and that is indistinguishable from a correct run on the
    /// first inference after a real one.
    ///
    /// # Safety
    ///
    /// The caller guarantees the four properties above for every replay of the
    /// returned graph.
    pub unsafe fn capture<T>(
        ctx: &HipContext,
        work: impl FnOnce() -> Result<T>,
    ) -> Result<(Self, T)> {
        let _exclusive = capture_lock();
        ctx.bind()?;
        check("hipStreamBeginCapture", unsafe {
            ffi::apxinf_hip_graph_capture_begin(ctx.stream())
        })?;

        // A failure inside `work` leaves the stream in capture mode, which
        // poisons every later launch on it. End the capture before returning
        // the error, and discard whatever was recorded.
        //
        // A panic inside `work` must end the capture too: the capture lock is
        // released while the stream is still capturing, and while a capture is
        // open `hipMalloc` fails device-wide and every launch on this stream
        // fails, so an unguarded panic converts one panicking inference into a
        // process-wide hang. Unwind the panic, end the capture, discard the
        // record, then resume the panic so the caller still sees it.
        //
        // Ending it does not make the stream usable again: the failure that
        // broke `work` latches an error on the stream itself, and clearing the
        // thread-local last error does not reach it. Measured -- a forward with
        // a stray `synchronize` fails the capture and then fails its own
        // device-to-host copy with "a previous error during capture". So the
        // caller must treat this as fatal for the stream rather than as a
        // reason to retry eagerly on it.
        let unwound = {
            CAPTURING.with(|flag| flag.set(true));
            let unwound = panic::catch_unwind(AssertUnwindSafe(work));
            CAPTURING.with(|flag| flag.set(false));
            unwound
        };
        let value = match unwound {
            Ok(Ok(value)) => value,
            outcome => {
                let mut exec = std::ptr::null_mut();
                let mut nodes = 0i32;
                unsafe {
                    ffi::apxinf_hip_graph_capture_end(ctx.stream(), &mut exec, &mut nodes);
                    if !exec.is_null() {
                        ffi::apxinf_hip_graph_destroy(exec);
                    }
                }
                match outcome {
                    Ok(Err(error)) => return Err(error),
                    Err(panic) => panic::resume_unwind(panic),
                    Ok(Ok(_)) => unreachable!(),
                }
            }
        };

        // Ended unconditionally, and its status read afterwards: a stream left
        // in capture mode poisons every later launch and every allocation on
        // the device, so returning early on an error here would trade a
        // reportable failure for an unexplainable one.
        let mut exec = std::ptr::null_mut();
        let mut nodes = 0i32;
        let ended =
            unsafe { ffi::apxinf_hip_graph_capture_end(ctx.stream(), &mut exec, &mut nodes) };
        check("hipStreamEndCapture", ended)?;
        if nodes <= 0 {
            unsafe { ffi::apxinf_hip_graph_destroy(exec) };
            return Err(Error::Other(
                "apxinf-hip: the capture recorded no work. A graph that records nothing \
                 instantiates and replays without error, leaving the previous contents of \
                 the output buffer, so this is rejected rather than returned."
                    .into(),
            ));
        }
        Ok((
            Self {
                exec,
                nodes: nodes as usize,
            },
            value,
        ))
    }

    /// Kernels and copies the graph holds. Fixed at capture.
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// Replay, asynchronously on the context's stream.
    pub fn launch(&self, ctx: &HipContext) -> Result<()> {
        ctx.bind()?;
        check("hipGraphLaunch", unsafe {
            ffi::apxinf_hip_graph_launch(self.exec, ctx.stream())
        })
    }
}
