//! Execution session — real session with workspace pooling and async execution.
//!
//! When active (thread-local), all `run_op` calls skip per-op
//! `aclrtSynchronizeStream` and sub-allocate workspace from a pre-allocated
//! pool. A single `aclrtSynchronizeStream` at the end of the forward pass
//! replaces ~9600 individual syncs.

use std::rc::Rc;

use apxinf_core::Result;

use crate::WorkspacePool;

/// An execution session that owns a workspace pool.
///
/// `prepare` runs the forward pass once to warm up (allocations land in
/// the pool deterministically). `run` executes the real forward pass
/// with async ops and no per-op sync.
#[derive(Clone)]
pub struct ExecutionSession {
    pool: Rc<WorkspacePool>,
}

impl ExecutionSession {
    /// Create a session with a workspace pool of `capacity_bytes`.
    pub fn with_capacity(capacity_bytes: usize, device: usize) -> Result<Self> {
        let cap = if capacity_bytes == 0 {
            1 * 1024 * 1024 // minimum 1MB default
        } else {
            capacity_bytes
        };
        let pool = WorkspacePool::new(cap, device as i32)
            .map_err(|e| apxinf_core::Error::Other(format!("workspace pool alloc: {e}")))?;
        Ok(Self { pool: Rc::new(pool) })
    }

    /// Workspace pool capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.pool.capacity()
    }

    /// Current workspace pool usage in bytes.
    pub fn used(&self) -> usize {
        self.pool.used()
    }

    /// Borrow the underlying workspace pool (for graph capture integration).
    pub fn pool(&self) -> &WorkspacePool {
        &self.pool
    }

    /// Report pool + device memory state on a forward boundary.
    ///
    /// Gated on `APXINF_DIAG_MEM=1`; prints pool high-water mark vs
    /// capacity, device free/total, and A1 pool-overflow totals.
    pub fn report_mem_diag(&self, phase: &str) {
        if !crate::diag_mem_enabled() {
            return;
        }
        let (mut free_opt, mut total_opt) = (None, None);
        for attr in [
            crate::ffi::acl::aclrtMemAttr::ACL_HBM_MEM,
            crate::ffi::acl::aclrtMemAttr::ACL_DDR_MEM,
        ] {
            let (f, t) = crate::get_mem_info(attr);
            if f.is_some() {
                free_opt = f;
                total_opt = t;
                break;
            }
        }
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        let fmt = |v: Option<usize>| {
            v.map(|x| format!("{:.1}", mb(x))).unwrap_or_else(|| "n/a".into())
        };
        let (ov_cnt, ov_bytes) = crate::output_overflow_stats();
        eprintln!(
            "[diag-mem] {phase}: pool.used={:.1}MB capacity={:.1}MB | device.free={}MB total={}MB | overflow.allocs={ov_cnt} bytes={:.1}MB",
            mb(self.used()),
            mb(self.capacity()),
            fmt(free_opt),
            fmt(total_opt),
            mb(ov_bytes as usize),
        );
    }

    /// Prepare phase: run the closure with the session active.
    pub fn prepare<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        prepare_with_session(self, operation)
    }

    /// Run phase: run the closure with the session active.
    /// The caller must call `synchronize` after this returns.
    pub fn run<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        with_session(self, operation)
    }
}

/// Run a closure with an active session (thread-local).
///
/// Sets the thread-local workspace pool and skip-sync flag so that all
/// `run_op` calls within the closure sub-allocate from the pool and
/// skip per-op synchronization. The pool is reset before execution.
/// The caller must call `synchronize` after this returns.
pub fn with_session<T>(session: &ExecutionSession, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    let _guard = unsafe { crate::enter_session(&session.pool) };
    let result = operation();
    session.report_mem_diag("run");
    result
}

/// Prepare with an active session (thread-local).
///
/// Identical to `with_session` — in Ascend eager mode, prepare and run
/// both execute the closure with async ops and pool allocation.
pub fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _guard = unsafe { crate::enter_session(&session.pool) };
    let result = operation();
    session.report_mem_diag("prepare");
    result
}
