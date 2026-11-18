//! Safe wrappers for aclTensor, aclScalar, and the two-phase aclnn API.
//!
//! Every aclnn operator follows the same pattern:
//! 1. Create `aclTensor` descriptors for inputs and outputs
//! 2. Call `aclnnXxxGetWorkspaceSize` → get `workspace_size` and `executor`
//! 3. Allocate workspace on device (if needed)
//! 4. Call `aclnnXxx(workspace, workspace_size, executor, stream)`
//! 5. Synchronize stream (only in bare eager mode — skipped inside a session)
//! 6. Destroy tensors, scalars, and executor

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use apxinf_core::{DType, Shape};

use crate::ffi::acl::{self, aclrtStream};
use crate::ffi::aclnn;

// ── DType mapping ─────────────────────────────────────────────────────

/// Convert `apxinf_core::DType` to `aclDataType`.
pub fn acl_dtype(dtype: DType) -> aclnn::aclDataType {
    match dtype {
        DType::F32 => aclnn::aclDataType::ACL_FLOAT,
        DType::F16 => aclnn::aclDataType::ACL_FLOAT16,
        DType::BF16 => aclnn::aclDataType::ACL_BF16,
        DType::I8 => aclnn::aclDataType::ACL_INT8,
        DType::I32 => aclnn::aclDataType::ACL_INT32,
        DType::F8E4M3 => aclnn::aclDataType::ACL_FLOAT8_E4M3FN,
        DType::E2M1Pair => aclnn::aclDataType::ACL_FLOAT4_E2M1,
    }
}

// ── aclTensor wrapper ─────────────────────────────────────────────────

/// RAII wrapper for `aclTensor`. Created from a device pointer + shape +
/// dtype. Automatically destroyed on drop.
pub struct AclTensor {
    raw: *mut aclnn::aclTensor,
}

impl AclTensor {
    /// Create an aclTensor from a device pointer, shape, and dtype.
    ///
    /// Strides are computed as row-major (C-contiguous). Format is ND.
    pub fn from_device(ptr: usize, shape: &Shape, dtype: DType) -> Self {
        let dims: Vec<i64> = shape.dims().iter().map(|&d| d as i64).collect();
        let strides: Vec<i64> = shape.strides().iter().map(|&s| s as i64).collect();
        let acl_dt = acl_dtype(dtype);

        let raw = unsafe {
            aclnn::aclCreateTensor(
                dims.as_ptr(),
                dims.len() as u64,
                acl_dt,
                strides.as_ptr(),
                0,
                aclnn::aclFormat::ACL_FORMAT_ND,
                ptr::null(),
                0,
                ptr as *mut c_void,
            )
        };
        Self { raw }
    }

    /// Create an aclTensor with explicit storage dimensions.
    ///
    /// `storage_dims` specifies the actual size of the underlying storage.
    /// This is needed when the tensor's logical shape is smaller than its
    /// storage (e.g., a [1, cols] view into a [total_cols] buffer for
    /// broadcasting — CANN needs to know the storage size to validate
    /// that the broadcast source data is accessible).
    pub fn from_device_with_storage(
        ptr: usize,
        shape: &Shape,
        dtype: DType,
        storage_dims: &[i64],
    ) -> Self {
        let dims: Vec<i64> = shape.dims().iter().map(|&d| d as i64).collect();
        let strides: Vec<i64> = shape.strides().iter().map(|&s| s as i64).collect();
        let acl_dt = acl_dtype(dtype);

        let raw = unsafe {
            aclnn::aclCreateTensor(
                dims.as_ptr(),
                dims.len() as u64,
                acl_dt,
                strides.as_ptr(),
                0,
                aclnn::aclFormat::ACL_FORMAT_ND,
                storage_dims.as_ptr(),
                storage_dims.len() as u64,
                ptr as *mut c_void,
            )
        };
        Self { raw }
    }

    pub fn raw(&self) -> *mut aclnn::aclTensor {
        self.raw
    }
}

impl Drop for AclTensor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { aclnn::aclDestroyTensor(self.raw) };
        }
    }
}

// ── aclScalar wrapper ─────────────────────────────────────────────────

/// RAII wrapper for `aclScalar`.
pub struct AclScalar {
    raw: *mut aclnn::aclScalar,
}

impl AclScalar {
    /// Create a float32 scalar.
    pub fn f32(val: f32) -> Self {
        let mut v = val;
        let raw = unsafe {
            aclnn::aclCreateScalar(
                &mut v as *mut f32 as *mut c_void,
                aclnn::aclDataType::ACL_FLOAT,
            )
        };
        Self { raw }
    }

    /// Create a float64 scalar (used by aclnnRmsNorm epsilon).
    pub fn f64(val: f64) -> Self {
        let mut v = val;
        let raw = unsafe {
            aclnn::aclCreateScalar(
                &mut v as *mut f64 as *mut c_void,
                aclnn::aclDataType::ACL_FLOAT,
            )
        };
        Self { raw }
    }

    pub fn raw(&self) -> *mut aclnn::aclScalar {
        self.raw
    }
}

impl Drop for AclScalar {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { aclnn::aclDestroyScalar(self.raw) };
        }
    }
}

// ── aclOpExecutor wrapper ─────────────────────────────────────────────

/// RAII wrapper for `aclOpExecutor`.
///
/// Note: CANN 9.1.0's `aclnnXxx` execute functions consume the executor —
/// calling `aclDestroyAclOpExecutor` after execute causes a double free.
/// We therefore do NOT destroy the executor on drop.
pub struct AclExecutor {
    raw: *mut aclnn::aclOpExecutor,
}

impl AclExecutor {
    pub fn null() -> *mut *mut aclnn::aclOpExecutor {
        ptr::null_mut()
    }

    pub fn from_raw(raw: *mut aclnn::aclOpExecutor) -> Self {
        Self { raw }
    }

    pub fn raw(&self) -> *mut aclnn::aclOpExecutor {
        self.raw
    }
}

// ── aclTensorList wrapper ─────────────────────────────────────────────

/// RAII wrapper for `aclTensorList`.
pub struct AclTensorList {
    raw: *mut aclnn::aclTensorList,
}

impl AclTensorList {
    /// Create an aclTensorList from a slice of aclTensor raw pointers.
    pub fn new(tensors: &[*const aclnn::aclTensor]) -> Self {
        let raw = unsafe {
            aclnn::aclCreateTensorList(tensors.as_ptr(), tensors.len() as u64)
        };
        Self { raw }
    }

    pub fn raw(&self) -> *mut aclnn::aclTensorList {
        self.raw
    }
}

impl Drop for AclTensorList {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { aclnn::aclDestroyTensorList(self.raw) };
        }
    }
}

// ── aclIntArray wrapper ───────────────────────────────────────────────

/// RAII wrapper for `aclIntArray`.
pub struct AclIntArray {
    raw: *mut aclnn::aclIntArray,
}

impl AclIntArray {
    /// Create an aclIntArray from a slice of i64 values.
    pub fn new(values: &[i64]) -> Self {
        let raw = unsafe { aclnn::aclCreateIntArray(values.as_ptr(), values.len() as u64) };
        Self { raw }
    }

    pub fn raw(&self) -> *mut aclnn::aclIntArray {
        self.raw
    }
}

impl Drop for AclIntArray {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { aclnn::aclDestroyIntArray(self.raw) };
        }
    }
}

// ── Workspace pool ───────────────────────────────────────────────────

const WORKSPACE_ALIGNMENT: usize = 256;

/// Pre-allocated device memory arena for aclnn workspace.
///
/// Single `aclrtMalloc` at construction; ops bump-allocate from it and
/// rewind after each op (stream ordering guarantees safety). Falls back
/// to individual `aclrtMalloc` if the request exceeds remaining capacity.
pub struct WorkspacePool {
    base: *mut c_void,
    capacity: usize,
    offset: Cell<usize>,
    device_id: i32,
    /// Individually `aclrtMalloc`'d buffers that overflowed the pool. The
    /// bool is `pinned`:
    ///  - `false` (eager/prepare forwards): freed at the next `reset()`, i.e.
    ///    once the enclosing forward has been synchronized (the runner
    ///    synchronizes before entering the next session). This is what bounds
    ///    memory across repeated eager forwards (A1 fix).
    ///  - `true` (RI capture): the buffer's address is baked into the captured
    ///    graph and must survive replays, so it is kept until the pool drops
    ///    (the caller destroys the graph before the session/pool).
    overflow: RefCell<Vec<(bool, *mut c_void)>>,
}

impl WorkspacePool {
    /// Allocate a pool of `capacity` bytes on the given device.
    pub fn new(capacity: usize, device_id: i32) -> Result<Self, String> {
        if capacity == 0 {
            return Ok(Self {
                base: ptr::null_mut(),
                capacity: 0,
                offset: Cell::new(0),
                device_id,
                overflow: RefCell::new(Vec::new()),
            });
        }
        crate::context::restore_context();
        let mut dev_ptr: *mut c_void = ptr::null_mut();
        let ret = unsafe {
            acl::aclrtMalloc(
                &mut dev_ptr,
                capacity,
                acl::aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
            )
        };
        let ret = if ret == 107002 {
            crate::context::restore_context();
            unsafe {
                acl::aclrtMalloc(
                    &mut dev_ptr,
                    capacity,
                    acl::aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
                )
            }
        } else {
            ret
        };
        acl::check(ret, "WorkspacePool aclrtMalloc")?;
        Ok(Self {
            base: dev_ptr,
            capacity,
            offset: Cell::new(0),
            device_id,
            overflow: RefCell::new(Vec::new()),
        })
    }

    /// Bump-allocate `bytes` from the pool, 256-byte aligned.
    /// Returns `(ptr, owned)`: `owned=false` means pool-owned (no free needed),
    /// `owned=true` means individually malloc'd (caller must free).
    pub fn alloc(&self, bytes: u64) -> Result<(*mut c_void, bool), String> {
        if bytes == 0 {
            return Ok((ptr::null_mut(), false));
        }
        let bytes_usize = bytes as usize;
        let aligned_start = (self.offset.get() + WORKSPACE_ALIGNMENT - 1) & !(WORKSPACE_ALIGNMENT - 1);
        let end = aligned_start + bytes_usize;
        if end <= self.capacity {
            self.offset.set(end);
            let ptr = unsafe { (self.base as *mut u8).add(aligned_start) } as *mut c_void;
            Ok((ptr, false))
        } else {
            let ptr = alloc_workspace(bytes)?;
            Ok((ptr, true))
        }
    }

    /// Rewind the bump cursor by `bytes` (aligned), reclaiming workspace
    /// after an op has enqueued its execute. Safe due to stream ordering.
    pub fn rewind(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let aligned_bytes = (bytes as usize + WORKSPACE_ALIGNMENT - 1) & !(WORKSPACE_ALIGNMENT - 1);
        let current = self.offset.get();
        if current >= aligned_bytes {
            self.offset.set(current - aligned_bytes);
        }
    }

    /// Reset the bump cursor to 0. Called at session boundaries.
    ///
    /// Also releases unpinned overflow buffers from earlier forwards (safe:
    /// callers synchronize the stream before entering the next session),
    /// which keeps repeated-eager memory bounded.
    pub fn reset(&self) {
        self.offset.set(0);
        self.free_unpinned_overflow();
    }

    /// Current high-water mark in bytes.
    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Register an individually-allocated overflow buffer with the pool.
    ///
    /// `pinned=false` buffers are freed at the next `reset()` (once the
    /// enclosing forward has been synchronized); `pinned=true` buffers are
    /// kept for the pool's lifetime (RI capture, where the address is baked
    /// into the graph). Either way the allocation is never leaked.
    pub fn register_overflow(&self, ptr: *mut c_void, pinned: bool) {
        if !ptr.is_null() {
            self.overflow.borrow_mut().push((pinned, ptr));
        }
    }

    /// Free overflow buffers that are not pinned. Safe because callers
    /// synchronize the device stream before entering the next session.
    fn free_unpinned_overflow(&self) {
        crate::context::restore_context();
        let mut v = self.overflow.borrow_mut();
        let mut keep: Vec<(bool, *mut c_void)> = Vec::with_capacity(v.len());
        for (pinned, p) in v.drain(..) {
            if pinned {
                keep.push((pinned, p));
            } else if !p.is_null() {
                unsafe { acl::aclrtFree(p) };
            }
        }
        *v = keep;
    }
}

// SAFETY: The pool's device memory is accessed only through ACL stream
// operations which are serialized. The bump cursor is only modified by
// the thread holding the session.
unsafe impl Send for WorkspacePool {}

impl Drop for WorkspacePool {
    fn drop(&mut self) {
        // Free any remaining overflow buffers (pinned capture-time buffers in
        // particular — the RI graph referencing them is destroyed before the
        // pool drops, per the runner drop order).
        let overflow = std::mem::take(&mut *self.overflow.borrow_mut());
        for (_pinned, p) in overflow.iter() {
            if !p.is_null() {
                unsafe { acl::aclrtFree(*p) };
            }
        }
        // Free the pool arena itself.
        if !self.base.is_null() {
            unsafe { acl::aclrtFree(self.base) };
        }
    }
}

// ── Thread-local session state ───────────────────────────────────────

thread_local! {
    static ACTIVE_WORKSPACE_POOL: Cell<*const WorkspacePool> = const { Cell::new(ptr::null()) };
    static SKIP_SYNC: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

// ── Phase 5a diagnostics ─────────────────────────────────────────────
// Thread-local timing accumulators (in microseconds) and op counters.

use std::time::Instant;

thread_local! {
    static OP_COUNT: Cell<u64> = const { Cell::new(0) };
    static ASYNC_OP_COUNT: Cell<u64> = const { Cell::new(0) };
    static SYNC_OP_COUNT: Cell<u64> = const { Cell::new(0) };
    static GET_WS_TIME_US: Cell<u64> = const { Cell::new(0) };
    static EXECUTE_TIME_US: Cell<u64> = const { Cell::new(0) };
    static SYNC_TIME_US: Cell<u64> = const { Cell::new(0) };
    static OP_NAME_COUNTS: RefCell<HashMap<String, u64>> = RefCell::new(HashMap::new());
    static OP_NAME_TIME_US: RefCell<HashMap<String, (u64, u64)>> = RefCell::new(HashMap::new());
    static OUTPUT_OVERFLOW_COUNT: Cell<u64> = const { Cell::new(0) };
    static OUTPUT_OVERFLOW_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Reset all diagnostic counters and timers.
pub fn reset_diagnostics() {
    OP_COUNT.with(|c| c.set(0));
    ASYNC_OP_COUNT.with(|c| c.set(0));
    SYNC_OP_COUNT.with(|c| c.set(0));
    GET_WS_TIME_US.with(|c| c.set(0));
    EXECUTE_TIME_US.with(|c| c.set(0));
    SYNC_TIME_US.with(|c| c.set(0));
    OP_NAME_COUNTS.with(|c| c.borrow_mut().clear());
    OP_NAME_TIME_US.with(|c| c.borrow_mut().clear());
    OUTPUT_OVERFLOW_COUNT.with(|c| c.set(0));
    OUTPUT_OVERFLOW_BYTES.with(|c| c.set(0));
}

/// Print diagnostic summary to stderr.
pub fn print_diagnostics() {
    let op_count = OP_COUNT.with(|c| c.get());
    let async_count = ASYNC_OP_COUNT.with(|c| c.get());
    let sync_count = SYNC_OP_COUNT.with(|c| c.get());
    let get_ws_ms = GET_WS_TIME_US.with(|c| c.get()) as f64 / 1000.0;
    let exec_ms = EXECUTE_TIME_US.with(|c| c.get()) as f64 / 1000.0;
    let sync_ms = SYNC_TIME_US.with(|c| c.get()) as f64 / 1000.0;
    eprintln!(
        "[diag] ops={op_count} async={async_count} sync={sync_count} | \
         GetWorkspaceSize={get_ws_ms:.1}ms Execute={exec_ms:.1}ms Sync={sync_ms:.1}ms | \
         total={:.1}ms",
        get_ws_ms + exec_ms + sync_ms
    );

    if diag_mem_enabled() {
        let (ov_cnt, ov_bytes) = output_overflow_stats();
        let (free, total) = get_mem_info(crate::ffi::acl::aclrtMemAttr::ACL_HBM_MEM);
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        let fmt = |v: Option<usize>| -> String {
            v.map(|x| format!("{:.1}", mb(x))).unwrap_or_else(|| "n/a".into())
        };
        eprintln!(
            "[diag-mem] forward overflow.allocs={ov_cnt} bytes={:.1}MB | hbm.free={}MB hbm.total={}MB",
            mb(ov_bytes as usize),
            fmt(free),
            fmt(total),
        );
    }

    // Per-op-name breakdown (top 40 by count)
    let mut entries: Vec<(String, u64, u64, u64)> = OP_NAME_COUNTS.with(|c| {
        let counts = c.borrow();
        let times = OP_NAME_TIME_US.with(|t| t.borrow().clone());
        counts
            .iter()
            .map(|(k, &cnt)| {
                let (get_us, exec_us) = *times.get(k).unwrap_or(&(0, 0));
                (k.clone(), cnt, get_us, exec_us)
            })
            .collect()
    });
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("[diag] op-name breakdown (top 40 by count):");
    eprintln!(
        "  {:<50} {:>10} {:>10} {:>10}",
        "op_name", "count", "get_ms", "exec_ms"
    );
    for (name, cnt, get_us, exec_us) in entries.iter().take(40) {
        eprintln!(
            "  {:<50} {:>10} {:>10.2} {:>10.2}",
            name,
            cnt,
            *get_us as f64 / 1000.0,
            *exec_us as f64 / 1000.0
        );
    }
}

/// Whether detailed memory diagnostics are enabled (`APXINF_DIAG_MEM=1`).
pub fn diag_mem_enabled() -> bool {
    std::env::var("APXINF_DIAG_MEM").is_ok()
}

/// Record a session output buffer that overflowed the workspace pool.
///
/// Each such allocation is an individually `aclrtMalloc`'d output beyond the
/// pool's capacity. In the baseline these are never freed (the A1 leak).
pub fn note_output_overflow(bytes: usize) {
    OUTPUT_OVERFLOW_COUNT.with(|c| c.set(c.get() + 1));
    OUTPUT_OVERFLOW_BYTES.with(|c| c.set(c.get() + bytes as u64));
}

/// `(count, total_bytes)` of pool-overflow output allocations seen since the
/// last `reset_diagnostics`.
pub fn output_overflow_stats() -> (u64, u64) {
    (
        OUTPUT_OVERFLOW_COUNT.with(|c| c.get()),
        OUTPUT_OVERFLOW_BYTES.with(|c| c.get()),
    )
}

/// Query device memory usage via `aclrtGetMemInfo`.
///
/// Returns `(free, total)` in bytes; an entry is `None` when the driver does
/// not support the query (returns non-success).
pub fn get_mem_info(attr: crate::ffi::acl::aclrtMemAttr) -> (Option<usize>, Option<usize>) {
    let mut free_ptr: usize = 0;
    let mut total_ptr: usize = 0;
    let ret = unsafe { crate::ffi::acl::aclrtGetMemInfo(attr, &mut free_ptr, &mut total_ptr) };
    if ret == 0 {
        (Some(free_ptr), Some(total_ptr))
    } else {
        (None, None)
    }
}

/// RAII guard that restores previous thread-local state on drop.
pub struct SessionGuard {
    prev_pool: *const WorkspacePool,
    prev_skip_sync: bool,
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        ACTIVE_WORKSPACE_POOL.with(|p| p.set(self.prev_pool));
        SKIP_SYNC.with(|s| s.set(self.prev_skip_sync));
    }
}

/// Enter a session scope: set active workspace pool and skip-sync flag.
/// Returns a guard that restores state when dropped.
///
/// # Safety
/// The caller must ensure `pool` outlives the returned guard.
pub unsafe fn enter_session(pool: &WorkspacePool) -> SessionGuard {
    let prev_pool = ACTIVE_WORKSPACE_POOL.with(|p| p.replace(pool as *const WorkspacePool));
    let prev_skip = SKIP_SYNC.with(|s| s.replace(true));
    pool.reset();
    SessionGuard {
        prev_pool,
        prev_skip_sync: prev_skip,
    }
}

/// Whether a session is active (sync should be skipped).
pub fn is_in_session() -> bool {
    SKIP_SYNC.with(|s| s.get())
}

/// Get the current thread-local workspace pool pointer, or null.
pub fn current_workspace_pool() -> *const WorkspacePool {
    ACTIVE_WORKSPACE_POOL.with(|p| p.get())
}

/// Set/reset whether an RI graph capture is active on this thread.
///
/// While capture is active, `alloc_zeros_session_aware` marks pool-overflow
/// outputs as pinned, so the pool keeps them alive for replays (the captured
/// graph bakes in those addresses) and frees them when the pool drops, after
/// the graph is destroyed.
pub fn set_capture_active(active: bool) {
    CAPTURE_ACTIVE.with(|c| c.set(active));
}

/// Whether RI graph capture is currently active on this thread.
pub fn capture_active() -> bool {
    CAPTURE_ACTIVE.with(|c| c.get())
}

// ── Workspace allocation ──────────────────────────────────────────────

/// Allocate a workspace buffer on device if `size > 0`, returning the
/// pointer. Returns null if no workspace is needed.
pub fn alloc_workspace(size: u64) -> Result<*mut c_void, String> {
    if size == 0 {
        return Ok(ptr::null_mut());
    }
    // Ensure the current thread has an ACL context bound before malloc.
    crate::context::restore_context();
    let mut dev_ptr: *mut c_void = ptr::null_mut();
    let ret = unsafe {
        acl::aclrtMalloc(
            &mut dev_ptr,
            size as usize,
            acl::aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
        )
    };
    // Auto-recover from CONTEXT_NULL (107002).
    let ret = if ret == 107002 {
        crate::context::restore_context();
        unsafe {
            acl::aclrtMalloc(
                &mut dev_ptr,
                size as usize,
                acl::aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
            )
        }
    } else {
        ret
    };
    acl::check(ret, "workspace aclrtMalloc")?;
    Ok(dev_ptr)
}

/// Free a workspace buffer.
pub fn free_workspace(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { acl::aclrtFree(ptr) };
    }
}

/// Run the two-phase aclnn execution pattern.
///
/// 1. `get_workspace_size_fn` returns (workspace_size, executor)
/// 2. Allocate workspace (from pool if in session, else malloc)
/// 3. `execute_fn` enqueues the operator on the stream
/// 4. Free workspace (only if individually allocated; pool reclaims via rewind)
/// 5. Synchronize stream (ONLY in bare eager mode — skipped inside a session)
pub fn run_op<F, G>(
    get_workspace_size_fn: G,
    execute_fn: F,
    stream: aclrtStream,
    op_name: &str,
) -> Result<(), String>
where
    G: FnOnce(&mut u64, &mut *mut aclnn::aclOpExecutor) -> aclnn::aclnnStatus,
    F: FnOnce(*mut c_void, u64, *mut aclnn::aclOpExecutor, aclrtStream) -> aclnn::aclnnStatus,
{
    let mut workspace_size: u64 = 0;
    let mut executor_ptr: *mut aclnn::aclOpExecutor = ptr::null_mut();

    crate::context::restore_context();

    // Phase 5a diagnostics: time each phase
    OP_COUNT.with(|c| c.set(c.get() + 1));
    OP_NAME_COUNTS.with(|c| *c.borrow_mut().entry(op_name.to_string()).or_insert(0) += 1);
    let t0 = Instant::now();

    let ret = get_workspace_size_fn(&mut workspace_size, &mut executor_ptr);
    let get_ws_us = t0.elapsed().as_micros() as u64;
    GET_WS_TIME_US.with(|c| c.set(c.get() + get_ws_us));
    aclnn::check(ret, &format!("{op_name} GetWorkspaceSize"))?;

    let executor = AclExecutor::from_raw(executor_ptr);

    // Allocate workspace: from pool if in session, else individual malloc.
    let pool_ptr = ACTIVE_WORKSPACE_POOL.with(|p| p.get());
    let (workspace, workspace_owned) = if workspace_size > 0 {
        if !pool_ptr.is_null() {
            unsafe { (*pool_ptr).alloc(workspace_size) }?
        } else {
            let ptr = alloc_workspace(workspace_size)?;
            (ptr, true)
        }
    } else {
        (ptr::null_mut(), false)
    };

    let t1 = Instant::now();
    let ret = execute_fn(workspace, workspace_size, executor.raw(), stream);
    let exec_us = t1.elapsed().as_micros() as u64;
    EXECUTE_TIME_US.with(|c| c.set(c.get() + exec_us));
    OP_NAME_TIME_US.with(|c| {
        let mut m = c.borrow_mut();
        let e = m.entry(op_name.to_string()).or_insert((0, 0));
        e.0 += get_ws_us;
        e.1 += exec_us;
    });
    let exec_result = aclnn::check(ret, &format!("{op_name} Execute"));

    // Reclaim workspace.
    if workspace_owned {
        free_workspace(workspace);
    } else if !pool_ptr.is_null() && workspace_size > 0 {
        unsafe { (*pool_ptr).rewind(workspace_size) };
    }

    // Synchronize ONLY in bare eager mode (no active session).
    let skip_sync = SKIP_SYNC.with(|s| s.get());
    if !skip_sync {
        let t2 = Instant::now();
        let _ = unsafe { acl::aclrtSynchronizeStream(stream) };
        let sync_us = t2.elapsed().as_micros() as u64;
        SYNC_TIME_US.with(|c| c.set(c.get() + sync_us));
        SYNC_OP_COUNT.with(|c| c.set(c.get() + 1));
    } else {
        ASYNC_OP_COUNT.with(|c| c.set(c.get() + 1));
    }

    exec_result
}
