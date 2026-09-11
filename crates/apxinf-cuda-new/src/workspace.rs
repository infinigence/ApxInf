//! Persistent graph memory and session-owned native executions.

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

/// Owns reusable native executions for one eager/capture/replay session.
/// Its typed cache is operator-independent.
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
            }),
        }
    }

    pub fn with_capacity(capacity_bytes: usize, device: usize) -> Result<Self> {
        GraphWorkspace::new(capacity_bytes, device).map(Self::new)
    }

    pub fn workspace(&self) -> &GraphWorkspace {
        &self.inner.workspace
    }
}

thread_local! {
    static ACTIVE_SESSION: Cell<*const ExecutionSessionInner> = const { Cell::new(std::ptr::null()) };
    static PREPARING: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_TARGET: Cell<Option<CaptureTarget>> = const { Cell::new(None) };
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
}

impl Drop for ActiveSessionGuard {
    fn drop(&mut self) {
        ACTIVE_SESSION.with(|active| active.set(self.session));
        PREPARING.with(|preparing| preparing.set(self.preparing));
    }
}

fn with_session_phase<T>(
    session: &ExecutionSession,
    prepare: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    ACTIVE_SESSION.with(|active| {
        if !active.get().is_null() {
            return Err(Error::Other(
                "nested execution sessions are not supported".into(),
            ));
        }
        session.inner.workspace.reset();
        if prepare {
            session.inner.prepared_sequence.borrow_mut().clear();
        }
        session.inner.sequence_cursor.set(0);
        let previous = active.replace(Rc::as_ptr(&session.inner));
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let _guard = ActiveSessionGuard {
            session: previous,
            preparing: previous_preparing,
        };
        retain_resource(&session.inner);
        let result = operation();
        if result.is_ok()
            && !prepare
            && session.inner.sequence_cursor.get() != session.inner.prepared_sequence.borrow().len()
        {
            return Err(Error::Other(
                "execution session traversal ended before the prepared operator sequence".into(),
            ));
        }
        result
    })
}

pub(crate) fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_session_phase(session, true, operation)
}

pub(crate) fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_session_phase(session, false, operation)
}

/// Native resources may be created only outside capture. A non-preparation
/// traversal with an active session must hit the execution cache.
pub(crate) fn may_prepare_native_resources() -> bool {
    !is_capturing()
        && (PREPARING.with(Cell::get) || ACTIVE_SESSION.with(|active| active.get().is_null()))
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
    CAPTURE_ACTIVE.with(|active| active.set(true));
}

pub(crate) fn end_capture_retention() -> Vec<Rc<dyn Any>> {
    CAPTURE_ACTIVE.with(|active| active.set(false));
    CAPTURE_TARGET.with(|target| target.set(None));
    CAPTURED_RESOURCES.with(|resources| std::mem::take(&mut *resources.borrow_mut()))
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
        if PREPARING.with(Cell::get) {
            inner.prepared_sequence.borrow_mut().push(identity);
            return Ok(());
        }
        let cursor = inner.sequence_cursor.get();
        let sequence = inner.prepared_sequence.borrow();
        if sequence.get(cursor).copied() != Some(identity) {
            return Err(Error::Other(format!(
                "execution session operator sequence mismatch at index {cursor}; prepare and capture must use identical operators and bindings"
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
