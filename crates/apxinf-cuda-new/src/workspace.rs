//! Persistent graph memory and session-owned prepared native executions.

use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

use apxinf_core::{Error, Result};

use crate::buffer::CudaBuffer;

const WORKSPACE_ALIGNMENT: usize = 256;

/// Persistent device arena used by a fixed-shape CUDA graph.
/// Native operator executions are deliberately not stored here.
pub struct GraphWorkspace {
    storage: CudaBuffer,
    offset: Cell<usize>,
}

impl GraphWorkspace {
    pub fn new(capacity_bytes: usize, device: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(Error::Other(
                "static inference workspace capacity must be non-zero".into(),
            ));
        }
        Ok(Self {
            storage: CudaBuffer::alloc(capacity_bytes, device).map_err(Error::Cuda)?,
            offset: Cell::new(0),
        })
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn used(&self) -> usize {
        self.offset.get()
    }

    fn reset(&self) {
        self.offset.set(0);
    }

    pub fn allocate(&self, bytes: usize, device: usize) -> Result<CudaBuffer> {
        if device != self.storage.device() {
            return Err(Error::Other(format!(
                "static inference workspace is on CUDA {}, but operation targets CUDA {device}",
                self.storage.device()
            )));
        }
        let start = self
            .offset
            .get()
            .checked_add(WORKSPACE_ALIGNMENT - 1)
            .ok_or_else(|| Error::Other("static inference workspace offset overflow".into()))?
            & !(WORKSPACE_ALIGNMENT - 1);
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::Other("static inference workspace size overflow".into()))?;
        if end > self.storage.len() {
            return Err(Error::Other(format!(
                "static inference workspace exhausted: need {end} bytes, capacity is {} bytes",
                self.storage.len()
            )));
        }
        self.offset.set(end);
        self.storage.view(start, bytes).map_err(Error::Cuda)
    }
}

struct ExecutionSessionInner {
    workspace: GraphWorkspace,
    caches: RefCell<HashMap<TypeId, Box<dyn Any>>>,
    prepared_sequence: RefCell<Vec<usize>>,
    sequence_cursor: Cell<usize>,
    /// Capture epoch the cursor was last reset for. A fresh capture restarts
    /// consumption from the beginning; successive run calls within one capture
    /// continue consuming the prepared sequence.
    consumed_epoch: Cell<u64>,
    /// Whether the previous traversal was a prepare. A prepare burst spans
    /// consecutive prepares and accumulates; a prepare after a run starts a
    /// fresh sequence.
    last_was_prepare: Cell<bool>,
}

impl ExecutionSessionInner {
    fn lookup<K, V>(&self, key: &K) -> Option<Rc<V>>
    where
        K: Eq + Hash + 'static,
        V: Any,
    {
        self.caches
            .borrow()
            .get(&TypeId::of::<(K, V)>())?
            .downcast_ref::<HashMap<K, Rc<V>>>()?
            .get(key)
            .cloned()
    }

    fn store<K, V>(&self, key: K, value: Rc<V>)
    where
        K: Eq + Hash + 'static,
        V: Any,
    {
        let mut caches = self.caches.borrow_mut();
        let cache = caches
            .entry(TypeId::of::<(K, V)>())
            .or_insert_with(|| Box::new(HashMap::<K, Rc<V>>::new()));
        cache
            .downcast_mut::<HashMap<K, Rc<V>>>()
            .expect("execution cache type identity collision")
            .insert(key, value);
    }
}

/// Owns reusable prepared resources for one eager/capture/replay session.
/// Only operators with real prepare state (currently GEMM and Attention) use
/// the typed cache and prepared sequence. Stateless operators launch directly.
#[derive(Clone)]
pub struct ExecutionSession {
    inner: Rc<ExecutionSessionInner>,
}

impl ExecutionSession {
    pub fn new(workspace: GraphWorkspace) -> Self {
        Self {
            inner: Rc::new(ExecutionSessionInner {
                workspace,
                caches: RefCell::new(HashMap::new()),
                prepared_sequence: RefCell::new(Vec::new()),
                sequence_cursor: Cell::new(0),
                consumed_epoch: Cell::new(0),
                last_was_prepare: Cell::new(false),
            }),
        }
    }

    pub fn with_capacity(capacity_bytes: usize, device: usize) -> Result<Self> {
        GraphWorkspace::new(capacity_bytes, device).map(Self::new)
    }

    pub fn workspace(&self) -> &GraphWorkspace {
        &self.inner.workspace
    }

    /// Arena capacity in bytes. Convenience passthrough for callers that
    /// treat the session as the legacy `GraphWorkspace`.
    pub fn capacity(&self) -> usize {
        self.inner.workspace.capacity()
    }

    /// Bytes handed out by the arena so far.
    pub fn used(&self) -> usize {
        self.inner.workspace.used()
    }

    /// Traverse the real execution once before graph capture.
    ///
    /// This allocates deterministic workspace slices, prepares native resources,
    /// and records the order of prepared operators. Stateless operators execute
    /// directly and do not enter the cache or sequence.
    pub fn prepare<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        prepare_with_session(self, operation)
    }

    /// Execute a traversal using resources installed by [`Self::prepare`].
    ///
    /// During CUDA graph capture this rejects missing or reordered operator
    /// executions instead of creating plans or tuning candidates in capture.
    pub fn run<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        with_session(self, operation)
    }

    /// Execute an eager (never captured) traversal over the session arena.
    ///
    /// Native resources stay installable exactly as they are without a
    /// session, and the prepared operator sequence is neither recorded nor
    /// consumed — mirroring the legacy `with_workspace_eager` contract for
    /// models that reuse one arena across plain eager inferences.
    pub fn eager<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        eager_with_session(self, operation)
    }
}

thread_local! {
    static ACTIVE_SESSION: Cell<*const ExecutionSessionInner> = const { Cell::new(std::ptr::null()) };
    static PREPARING: Cell<bool> = const { Cell::new(false) };
    /// Eager traversal: native resources stay installable and the prepared
    /// operator sequence is neither recorded nor consumed.
    static EAGER: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_EPOCH: Cell<u64> = const { Cell::new(0) };
    static CAPTURE_TARGET: Cell<Option<CaptureTarget>> = const { Cell::new(None) };
    /// Session that consumed the prepared sequence during the active capture,
    /// so its completion can be verified once at capture end rather than after
    /// every traversal (a multi-phase model trips a per-call check).
    static CAPTURE_SESSION: Cell<*const ExecutionSessionInner> = const { Cell::new(std::ptr::null()) };
    static CAPTURED_RESOURCES: RefCell<Vec<Rc<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct CaptureTarget {
    device: usize,
    stream: usize,
}

struct ActiveSessionGuard {
    session: *const ExecutionSessionInner,
    preparing: bool,
    eager: bool,
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        ACTIVE_SESSION.with(|active| active.set(self.session));
        PREPARING.with(|preparing| preparing.set(self.preparing));
        EAGER.with(|eager| eager.set(self.eager));
    }
}

fn with_session_phase<T>(
    session: &ExecutionSession,
    prepare: bool,
    eager: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    ACTIVE_SESSION.with(|active| {
        if !active.get().is_null() {
            return Err(Error::Other(
                "nested execution sessions are not supported".into(),
            ));
        }
        session.inner.workspace.reset();
        if eager {
            // Eager traversals only borrow the arena; they leave the prepared
            // sequence and its burst tracking untouched so a later
            // prepare/capture pair still sees a consistent sequence.
        } else if prepare {
            // A prepare burst may span several `prepare_with_workspace` calls
            // (a model that wraps each phase separately), so accumulate across
            // consecutive prepares. A prepare that follows a run starts a fresh
            // sequence instead.
            if !session.inner.last_was_prepare.replace(true) {
                session.inner.prepared_sequence.borrow_mut().clear();
                session.inner.sequence_cursor.set(0);
            }
        } else {
            session.inner.last_was_prepare.set(false);
            // The prepared sequence is built by a prepare burst that may span
            // several `prepare_with_workspace` calls (a model that wraps each
            // phase separately). Consumption mirrors that shape:
            //  - a new capture restarts from the beginning;
            //  - successive `with_session` calls inside one capture continue
            //    consuming the same sequence (multi-phase capture);
            //  - an eager traversal re-runs the whole prepared set from the
            //    beginning on every call.
            if !is_capturing() {
                session.inner.sequence_cursor.set(0);
            } else {
                let epoch = CAPTURE_EPOCH.with(Cell::get);
                if session.inner.consumed_epoch.get() != epoch {
                    session.inner.sequence_cursor.set(0);
                    session.inner.consumed_epoch.set(epoch);
                }
                // Remember the consuming session so capture end can assert the
                // whole prepared sequence was traversed.
                CAPTURE_SESSION.with(|slot| slot.set(Rc::as_ptr(&session.inner)));
            }
        }
        let previous = active.replace(Rc::as_ptr(&session.inner));
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let previous_eager = EAGER.with(|current| current.replace(eager));
        let _guard = ActiveSessionGuard {
            session: previous,
            preparing: previous_preparing,
            eager: previous_eager,
        };
        retain_resource(&session.inner);
        operation()
    })
}

pub(crate) fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_session_phase(session, true, false, operation)
}

pub(crate) fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_session_phase(session, false, false, operation)
}

/// Bind a session arena for an eager (never captured) traversal. Native
/// resources stay installable exactly as they are without a session; the
/// prepared operator sequence is untouched. Capture must keep using
/// [`with_session`]: an eager traversal cannot tell that it is recording, and
/// CUDA forbids allocating or re-planning inside a capture.
pub(crate) fn eager_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if is_capturing() {
        return Err(Error::Other(
            "eager session traversal is not allowed during CUDA graph capture".into(),
        ));
    }
    with_session_phase(session, false, true, operation)
}

/// Native resources may be created only outside capture. A non-preparation
/// traversal with an active session must hit the execution cache.
pub(crate) fn may_prepare_native_resources() -> bool {
    !is_capturing()
        && (PREPARING.with(Cell::get)
            || EAGER.with(Cell::get)
            || ACTIVE_SESSION.with(|active| active.get().is_null()))
}

pub(crate) fn has_active_session() -> bool {
    ACTIVE_SESSION.with(|active| !active.get().is_null())
}

pub(crate) fn is_preparing_session() -> bool {
    PREPARING.with(Cell::get)
}

pub(crate) fn begin_capture_retention(device: usize, stream: usize) {
    CAPTURED_RESOURCES.with(|resources| resources.borrow_mut().clear());
    CAPTURE_TARGET.with(|target| target.set(Some(CaptureTarget { device, stream })));
    CAPTURE_EPOCH.with(|epoch| epoch.set(epoch.get().wrapping_add(1)));
    CAPTURE_SESSION.with(|slot| slot.set(std::ptr::null()));
    CAPTURE_ACTIVE.with(|active| active.set(true));
}

/// End the active capture. When `capture_ok` is true, verify that the session
/// which consumed the prepared sequence traversed all of it: a captured graph
/// that omitted a prepared operator would silently replay wrong results, and
/// the per-operator mismatch check cannot see a sequence that ended early.
/// A failed capture legitimately under-runs, so it skips the check.
pub(crate) fn end_capture_retention(
    capture_ok: bool,
) -> Result<Vec<Rc<dyn Any>>> {
    CAPTURE_ACTIVE.with(|active| active.set(false));
    CAPTURE_TARGET.with(|target| target.set(None));
    let session = CAPTURE_SESSION.with(|slot| slot.replace(std::ptr::null()));
    let retained = CAPTURED_RESOURCES.with(|resources| std::mem::take(&mut *resources.borrow_mut()));
    if capture_ok && !session.is_null() {
        let inner = unsafe { &*session };
        let cursor = inner.sequence_cursor.get();
        let total = inner.prepared_sequence.borrow().len();
        if cursor != total {
            return Err(Error::Other(format!(
                "execution session traversal ended before the prepared operator sequence \
                 (consumed {cursor} of {total})"
            )));
        }
    }
    Ok(retained)
}

pub(crate) fn is_capturing() -> bool {
    CAPTURE_ACTIVE.with(Cell::get)
}

pub(crate) fn validate_capture_target(device: usize, stream: usize) -> Result<()> {
    CAPTURE_TARGET.with(|target| match target.get() {
        Some(active) if active.device != device || active.stream != stream => Err(Error::Other(
            format!(
                "prepared execution is bound to CUDA device {device} stream 0x{stream:x}, but the active capture targets CUDA device {} stream 0x{:x}; multi-stream capture is not supported",
                active.device, active.stream
            ),
        )),
        _ => Ok(()),
    })
}

pub(crate) fn lookup_execution<K, V>(key: &K) -> Option<Rc<V>>
where
    K: Eq + Hash + 'static,
    V: Any,
{
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            None
        } else {
            unsafe { &*inner }.lookup(key)
        }
    })
}

pub(crate) fn store_execution<K, V>(key: K, value: Rc<V>)
where
    K: Eq + Hash + 'static,
    V: Any,
{
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if !inner.is_null() {
            unsafe { &*inner }.store(key, value);
        }
    });
}

/// Record one operator occurrence and enforce that capture follows the exact
/// execution order established by `prepare_with_session`.
pub(crate) fn use_execution<T: Any>(resource: &Rc<T>) -> Result<()> {
    let identity = Rc::as_ptr(resource).cast::<()>() as usize;
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            return Ok(());
        }
        let inner = unsafe { &*inner };
        if EAGER.with(Cell::get) {
            // Eager traversals neither record nor consume the prepared
            // sequence; the cached execution is simply reused.
            return Ok(());
        }
        if PREPARING.with(Cell::get) {
            inner.prepared_sequence.borrow_mut().push(identity);
            return Ok(());
        }
        let cursor = inner.sequence_cursor.get();
        let sequence = inner.prepared_sequence.borrow();
        if sequence.get(cursor).copied() != Some(identity) {
            return Err(Error::Other(format!(
                "execution session operator sequence mismatch at index {cursor} \
                 (prepared total {}); prepare and capture must use identical operators and bindings \
                 [expected {:?}, got {identity:#x}]",
                sequence.len(),
                sequence.get(cursor).map(|value| format!("{value:#x}")),
            )));
        }
        inner.sequence_cursor.set(cursor + 1);
        Ok(())
    })?;
    retain_resource(resource);
    Ok(())
}

pub(crate) fn retain_resource<T: Any>(resource: &Rc<T>) {
    if !is_capturing() {
        return;
    }
    let resource: Rc<dyn Any> = resource.clone();
    CAPTURED_RESOURCES.with(|resources| {
        let mut resources = resources.borrow_mut();
        if !resources.iter().any(|stored| Rc::ptr_eq(stored, &resource)) {
            resources.push(resource);
        }
    });
}

/// Session-aware allocation for direct-launch operators. During an
/// execution session it sub-allocates deterministically from the session
/// arena; eager calls own a zeroed allocation directly.
pub(crate) fn output_buffer(ctx: &crate::CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            // The runtime stream is deliberately non-blocking, so a memset on
            // the legacy default stream is not ordered before the operator
            // that consumes this output. Keep initialization on the same
            // stream to prevent a late memset from erasing kernel results.
            CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream())
                .map_err(Error::Cuda)
        } else {
            unsafe { &*inner }
                .workspace
                .allocate(bytes, ctx.device_id())
        }
    })
}

/// Like [`output_buffer`], but cleared even when the session arena hands back
/// a reused view.
pub(crate) fn output_buffer_zeroed(ctx: &crate::CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream()).map_err(Error::Cuda)
        } else {
            let buffer = unsafe { &*inner }.workspace.allocate(bytes, ctx.device_id())?;
            buffer
                .memset_async(0, bytes, ctx.stream())
                .map_err(Error::Cuda)?;
            Ok(buffer)
        }
    })
}

/// As [`output_buffer_zeroed`], but only the trailing rows of each group:
/// `group_bytes - used_bytes` trailing bytes in each of `groups` groups.
pub(crate) fn output_buffer_tail_zeroed(
    ctx: &crate::CudaContext,
    groups: usize,
    group_bytes: usize,
    used_bytes: usize,
) -> Result<CudaBuffer> {
    let bytes = groups
        .checked_mul(group_bytes)
        .ok_or_else(|| Error::Other("scratch extent overflow".into()))?;
    let tail = group_bytes
        .checked_sub(used_bytes)
        .ok_or_else(|| Error::Other("scratch used bytes exceed group extent".into()))?;
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        let buffer = if inner.is_null() {
            // A fresh driver allocation still has to establish the tail, but
            // it is not a reused arena, so the payload rows need no clearing.
            CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?
        } else {
            unsafe { &*inner }.workspace.allocate(bytes, ctx.device_id())?
        };
        if tail > 0 && groups > 0 {
            buffer
                .memset_2d_async(0, used_bytes, group_bytes, tail, groups, ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(buffer)
    })
}
