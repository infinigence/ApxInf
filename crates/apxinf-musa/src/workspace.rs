//! Persistent device arena and the prepare/run session a captured graph is
//! recorded against.
//!
//! The arena is a port of `apxinf-cuda/src/workspace.rs`: 256-byte alignment,
//! reused views that are **never cleared**, and FP8 emulation scratch for a
//! path this port does not compile.
//!
//! The session is the contract `apxinf-cuda-new/src/workspace.rs` defines,
//! because PI0.5's `model_runner` is written against it. `prepare` traverses
//! the model once outside capture, allocating arena slices and recording every
//! GEMM and attention shape whose first use binds vendor state; `run` replays
//! that traversal and refuses a shape the prepare pass never reached. That is
//! what makes a captured graph a record of a known traversal rather than of
//! whatever the second pass happened to do.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use apxinf_core::{DType, Error, Result};

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;
use crate::error::musa_error;

const WORKSPACE_ALIGNMENT: usize = 256;

/// Persistent device arena used by a fixed-shape graph.
pub struct GraphWorkspace {
    storage: MusaBuffer,
    offset: Cell<usize>,
    fp8_emulation: Option<Fp8EmulationWorkspace>,
}

struct Fp8EmulationWorkspace {
    activation: MusaBuffer,
    weight: MusaBuffer,
}

impl GraphWorkspace {
    pub fn new(capacity_bytes: usize, device: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(Error::Other(
                "static inference workspace capacity must be non-zero".into(),
            ));
        }
        Ok(Self {
            storage: MusaBuffer::alloc(capacity_bytes, device).map_err(musa_error)?,
            offset: Cell::new(0),
            fp8_emulation: None,
        })
    }

    pub fn new_fp8(
        capacity_bytes: usize,
        max_activation_elements: usize,
        max_weight_elements: usize,
        device: usize,
    ) -> Result<Self> {
        let mut workspace = Self::new(capacity_bytes, device)?;
        // MUSA native FP8 support on MP22 is **unverified**: the toolkit
        // exposes the FP8 types and `mublasLt` the `..._INPUT_8F_E4M3`
        // numerical flag, but whether M1000 executes them on tensor cores is
        // not established here. The CUDA side decides this from compute
        // capability; taking the conservative branch means an FP8 run on MUSA
        // gets the emulation scratch it would need if there is no native path.
        // FP8 is outside this port's scope, so nothing depends on the answer
        // yet — but it must be measured before an FP8 port starts.
        let native_fp8 = false;
        if !native_fp8 {
            if max_activation_elements == 0 || max_weight_elements == 0 {
                return Err(Error::Other(
                    "static inference FP8 emulation scratch capacities must be non-zero".into(),
                ));
            }
            let activation_bytes = max_activation_elements
                .checked_mul(DType::F16.size_in_bytes())
                .ok_or_else(|| {
                    Error::Other("static inference FP8 activation scratch overflow".into())
                })?;
            let weight_bytes = max_weight_elements
                .checked_mul(DType::F16.size_in_bytes())
                .ok_or_else(|| {
                    Error::Other("static inference FP8 weight scratch overflow".into())
                })?;
            workspace.fp8_emulation = Some(Fp8EmulationWorkspace {
                activation: MusaBuffer::alloc(activation_bytes, device).map_err(musa_error)?,
                weight: MusaBuffer::alloc(weight_bytes, device).map_err(musa_error)?,
            });
        }
        Ok(workspace)
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

    fn allocate(&self, bytes: usize, device: usize) -> Result<MusaBuffer> {
        if device != self.storage.device() {
            return Err(Error::Other(format!(
                "static inference workspace is on MUSA {}, but operation targets MUSA {device}",
                self.storage.device()
            )));
        }
        let start = self
            .offset
            .get()
            .checked_add(WORKSPACE_ALIGNMENT - 1)
            .ok_or_else(|| Error::Other("static inference workspace offset overflow".into()))?
            & !(WORKSPACE_ALIGNMENT - 1);
        // Diagnostic slack between arena allocations. The driver hands out
        // separate, non-adjacent blocks, so a kernel that writes slightly past
        // the end of its buffer lands in unused memory; the arena packs
        // allocations back to back, where the same overrun clobbers the next
        // one. If a slack makes a failure go away, that is the shape of it.
        let slack = workspace_slack();
        let end = start
            .checked_add(bytes)
            .and_then(|value| value.checked_add(slack))
            .ok_or_else(|| Error::Other("static inference workspace size overflow".into()))?;
        if end > self.storage.len() {
            return Err(Error::Other(format!(
                "static inference workspace exhausted: need {end} bytes, capacity is {} bytes",
                self.storage.len()
            )));
        }
        self.offset.set(end);
        self.storage.view(start, bytes).map_err(musa_error)
    }

    fn uses_fp8_emulation(&self) -> bool {
        self.fp8_emulation.is_some()
    }

    fn fp8_emulation_buffers(
        &self,
        activation_bytes: usize,
        weight_bytes: usize,
        device: usize,
    ) -> Result<(MusaBuffer, MusaBuffer)> {
        let scratch = self.fp8_emulation.as_ref().ok_or_else(|| {
            Error::Other(
                "static inference FP8 emulation requires GraphWorkspace::new_fp8 before graph capture".into(),
            )
        })?;
        if device != scratch.activation.device() {
            return Err(Error::Other(format!(
                "static inference FP8 emulation workspace is on MUSA {}, but operation targets MUSA {device}",
                scratch.activation.device()
            )));
        }
        if activation_bytes > scratch.activation.len() || weight_bytes > scratch.weight.len() {
            return Err(Error::Other(format!(
                "static inference FP8 emulation scratch exhausted: activation {activation_bytes}/{} bytes, weight {weight_bytes}/{} bytes",
                scratch.activation.len(),
                scratch.weight.len()
            )));
        }
        Ok((
            scratch
                .activation
                .view(0, activation_bytes)
                .map_err(musa_error)?,
            scratch.weight.view(0, weight_bytes).map_err(musa_error)?,
        ))
    }
}

struct ExecutionSessionInner {
    workspace: GraphWorkspace,
    /// GEMM and attention shapes the prepare traversal touched, in first-use
    /// order. A `run` traversal must stay inside this set; see [`note_shape`].
    ///
    /// A `Vec` rather than a `HashSet`: the order is part of the contract (it is
    /// what [`ExecutionSession::prepared_shapes`] hands back and what the tests
    /// assert), and a traversal touches tens of shapes, so the linear scan
    /// costs nothing next to the hash it replaces.
    prepared_shapes: RefCell<Vec<OperationShape>>,
}

/// Owns the arena and the prepared shape set for one eager/capture/replay
/// session.
///
/// Only operators whose first use binds vendor state appear in the set;
/// stateless operators launch directly and are not recorded.
#[derive(Clone)]
pub struct ExecutionSession {
    inner: Rc<ExecutionSessionInner>,
}

impl ExecutionSession {
    pub fn new(workspace: GraphWorkspace) -> Self {
        Self {
            inner: Rc::new(ExecutionSessionInner {
                workspace,
                prepared_shapes: RefCell::new(Vec::new()),
            }),
        }
    }

    pub fn with_capacity(capacity_bytes: usize, device: usize) -> Result<Self> {
        GraphWorkspace::new(capacity_bytes, device).map(Self::new)
    }

    pub fn workspace(&self) -> &GraphWorkspace {
        &self.inner.workspace
    }

    /// Traverse the real execution once before graph capture.
    ///
    /// This allocates deterministic arena slices, prepares native resources and
    /// records the shapes whose provider state now exists. Stateless operators
    /// execute directly and are not recorded.
    pub fn prepare<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        prepare_with_session(self, operation)
    }

    /// Execute a traversal using resources installed by [`Self::prepare`].
    ///
    /// During graph capture this rejects a shape outside the prepared set
    /// instead of creating plans or tuning candidates inside the capture.
    pub fn run<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        with_session(self, operation)
    }

    /// GEMM and attention shapes the prepare traversal touched, in first-use
    /// order.
    ///
    /// This is the list a captured graph is a record of. Its purpose is to be
    /// inspectable: covering it is what lets capture succeed, and a shape
    /// outside it fails [`Self::run`] with the shape named.
    pub fn prepared_shapes(&self) -> Vec<OperationShape> {
        self.inner.prepared_shapes.borrow().clone()
    }
}

/// A GEMM or attention shape whose first use binds vendor state.
///
/// muDNN builds a descriptor per `(m, n, k)` and per `(batch, heads, tokens,
/// dim)` and refuses to do so inside capture, because a captured graph bakes
/// the pointers it recorded. Every other path warms per stream instead of per
/// shape, so the set below is deliberately *logical* — the shape the model
/// means — rather than a mirror of one vendor's key: the prepare traversal that
/// covers it covers whichever provider a capture-time dispatch picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperationShape {
    /// `[m, k] @ [k, n]`: rows, output columns, contraction.
    Gemm { m: usize, n: usize, k: usize },
    /// `queries` rows of `heads` x `head_dim` over a `keys`-row K/V cache.
    Attention {
        queries: usize,
        keys: usize,
        heads: usize,
        head_dim: usize,
    },
}

impl std::fmt::Display for OperationShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gemm { m, n, k } => write!(formatter, "GEMM m={m} n={n} k={k}"),
            Self::Attention {
                queries,
                keys,
                heads,
                head_dim,
            } => write!(
                formatter,
                "attention {queries} queries over {keys} keys, {heads} heads of {head_dim}"
            ),
        }
    }
}

thread_local! {
    static ACTIVE_SESSION: Cell<*const ExecutionSessionInner> = const { Cell::new(std::ptr::null()) };
    static PREPARING: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CAPTURED_RESOURCES: RefCell<Vec<Rc<dyn Any>>> = const { RefCell::new(Vec::new()) };
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
        // The reset happens only once the call is accepted. Resetting before
        // the check would leave the *outer* traversal allocating from the
        // bottom of an arena it is already part-way through — two live outputs
        // aliasing one range, and nothing to say so.
        session.inner.workspace.reset();
        if prepare {
            session.inner.prepared_shapes.borrow_mut().clear();
        }
        let previous = active.replace(Rc::as_ptr(&session.inner));
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let _guard = ActiveSessionGuard {
            session: previous,
            preparing: previous_preparing,
        };
        retain_resource(&session.inner);
        operation()
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

pub(crate) fn begin_capture_retention() {
    CAPTURED_RESOURCES.with(|resources| resources.borrow_mut().clear());
    CAPTURE_ACTIVE.with(|active| active.set(true));
}

pub(crate) fn end_capture_retention() -> Vec<Rc<dyn Any>> {
    CAPTURE_ACTIVE.with(|active| active.set(false));
    CAPTURED_RESOURCES.with(|resources| std::mem::take(&mut *resources.borrow_mut()))
}

pub(crate) fn is_capturing() -> bool {
    CAPTURE_ACTIVE.with(Cell::get)
}

/// Record one GEMM or attention shape a traversal reached.
///
/// A prepare traversal records; a `run` traversal requires, and a shape the
/// prepare pass never saw is refused by name. Capture is where that matters:
/// muDNN answers a first-touch shape inside capture with its own status and a
/// paragraph, and this is the same refusal stated where the shape is known.
pub(crate) fn note_shape(shape: OperationShape) -> Result<()> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            return Ok(());
        }
        let inner = unsafe { &*inner };
        let mut shapes = inner.prepared_shapes.borrow_mut();
        if PREPARING.with(Cell::get) {
            if !shapes.contains(&shape) {
                shapes.push(shape);
            }
            return Ok(());
        }
        if shapes.contains(&shape) {
            return Ok(());
        }
        Err(Error::Other(format!(
            "MUSA execution session reached {shape}, which the prepare traversal never ran. \
             A captured graph can only bake shapes whose provider state already exists, so \
             every shape has to be warmed on the same stream before capture begins"
        )))
    })
}

/// Keep a prepared resource alive for as long as the captured graph lives.
fn retain_resource<T: Any>(resource: &Rc<T>) {
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

/// Diagnostic slack, read once: this is on the arena-allocation path.
fn workspace_slack() -> usize {
    static SLACK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SLACK.get_or_init(|| {
        std::env::var("APXINF_MUSA_WORKSPACE_SLACK")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// Which filler, if any, the workspace-less path applies to an operator output.
///
/// The workspace path already makes the case: [`GraphWorkspace::allocate`]
/// returns a reused view into its arena and never clears it, so every consumer
/// of `output_buffer` already tolerates dirty memory. The fallback cleared
/// anyway, and it is not cheap — the CUDA backend measured one Qwen-Drive VQA
/// inference on Orin issuing 33,545 memsets covering 41GB.
///
/// **Measured on M1000, and the clearing is not needed.** One PI0.5 inference
/// issues 3,523 memsets per pass. Filling those buffers with 0xFF instead —
/// NaN at every float width, so a read-before-write propagates into the output
/// — leaves the result **bit-identical**, end to end and at every stage: the
/// same cosine 0.9998294115066528 as the zeroed run. The clearing cost 213 ms
/// of a 1,081 ms pass, and the memsets cost more than their own device time:
/// without them the GEMM operators run 345 ms instead of 564 ms per pass
/// without a line of GEMM code changing, because a 21 us memset between two
/// GEMMs drains the queue and empties the caches in between.
///
/// So the default is `Dirty`. `=poison` is the check above: it turns "nothing
/// read uninitialized memory" from something a clean run merely fails to
/// disprove into something a run actively convicts.
#[derive(Clone, Copy, PartialEq)]
enum OutputFill {
    Dirty,
    Poison,
}

fn output_fill() -> OutputFill {
    static FILL: std::sync::OnceLock<OutputFill> = std::sync::OnceLock::new();
    *FILL.get_or_init(|| match std::env::var("APXINF_MUSA_OUTPUT_FILL").as_deref() {
        Ok("poison") => OutputFill::Poison,
        _ => OutputFill::Dirty,
    })
}

/// Session-aware allocation for direct-launch operators. During an execution
/// session it sub-allocates deterministically from the session arena; an eager
/// call owns a driver allocation on the context's stream.
pub(crate) fn output_buffer(ctx: &MusaContext, bytes: usize) -> Result<MusaBuffer> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            // On the context's stream, so these are the blocks the reuse cache
            // may hand back: an operator output is written by a kernel on that
            // stream and read by the next one on the same stream, so a recycled
            // block is ordered behind whatever last used it.
            match output_fill() {
                OutputFill::Dirty => MusaBuffer::alloc_on(ctx, bytes).map_err(musa_error),
                OutputFill::Poison => {
                    MusaBuffer::alloc_filled_on(ctx, bytes, 0xFF).map_err(musa_error)
                }
            }
        } else {
            unsafe { &*inner }.workspace.allocate(bytes, ctx.device_id())
        }
    })
}

/// Like [`output_buffer`], but cleared.
///
/// A fresh driver allocation is not zero either, but the workspace hands back a
/// reused arena view, so a caller that needs zeros has to say so. Used for the
/// padded tails that a model's prep kernels leave untouched.
pub(crate) fn output_buffer_zeroed(ctx: &MusaContext, bytes: usize) -> Result<MusaBuffer> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            MusaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream()).map_err(musa_error)
        } else {
            let buffer = unsafe { &*inner }
                .workspace
                .allocate(bytes, ctx.device_id())?;
            buffer
                .memset_async(0, bytes, ctx.stream())
                .map_err(musa_error)?;
            Ok(buffer)
        }
    })
}

/// Whether FP8 must be emulated rather than executed natively.
///
/// Every FP8 operator in this crate reports `not_ported`, so nothing asks yet;
/// this is what a ported one consults, next to the emulation buffers below.
#[allow(dead_code)]
pub(crate) fn fp8_emulation_required(ctx: &MusaContext) -> Result<bool> {
    Ok(ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            let _ = ctx;
            // No native FP8 path is established on M1000; see `new_fp8`.
            true
        } else {
            unsafe { &*inner }.workspace.uses_fp8_emulation()
        }
    }))
}

/// The FP8 emulation scratch bound to the active session.
///
/// Uncalled for the same reason as `fp8_emulation_required`; it is the pair
/// that a ported FP8 operator needs together.
#[allow(dead_code)]
pub(crate) fn fp8_emulation_buffers(
    _ctx: &MusaContext,
    activation_bytes: usize,
    weight_bytes: usize,
    device: usize,
) -> Result<(MusaBuffer, MusaBuffer)> {
    ACTIVE_SESSION.with(|active| {
        let inner = active.get();
        if inner.is_null() {
            return Err(Error::Other(
                "static inference FP8 emulation requires an active FP8 session".into(),
            ));
        }
        unsafe { &*inner }.workspace.fp8_emulation_buffers(
            activation_bytes,
            weight_bytes,
            device,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refused nested traversal must not touch the arena the outer one is
    /// still allocating from.
    ///
    /// The nested call is built by hand because nothing in the model layer is
    /// supposed to nest — that is what the guard is for, and this asserts that
    /// the refusal is *inert*. It used to zero the offset before it checked, so
    /// the outer traversal's next allocation restarted at the bottom of an
    /// arena it was half-way through: two live outputs, one range, nothing said.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn a_refused_nested_traversal_leaves_the_outer_arena_alone() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let device = ctx.device_id();
        let session = ExecutionSession::with_capacity(1 << 20, device).expect("session");

        let (used, message) = session
            .prepare(|| {
                session.workspace().allocate(1024, device).expect("outer allocation");
                assert_eq!(session.workspace().used(), 1024);

                let nested = session.run(|| Ok(()));
                let message = nested
                    .expect_err("a nested traversal must be refused")
                    .to_string();

                Ok((session.workspace().used(), message))
            })
            .expect("the outer traversal");

        assert!(
            message.contains("nested execution sessions"),
            "unhelpful refusal: {message}"
        );
        assert_eq!(
            used, 1024,
            "the refused call reset the arena the outer traversal was allocating from"
        );
    }

    /// A shape the prepare traversal never ran is refused by name, before it
    /// can reach a provider that would answer with a status of its own.
    ///
    /// The list `prepare` leaves behind is what capture is a record of, so it
    /// is also the acceptance surface: first-use order, deduplicated, and
    /// replaced — not extended — by the next prepare.
    #[test]
    fn a_shape_outside_the_prepare_traversal_is_refused() {
        let workspace = GraphWorkspace::new(1 << 20, 0).expect("host arena");
        let session = ExecutionSession::new(workspace);
        let gemm = OperationShape::Gemm {
            m: 8,
            n: 32,
            k: 16,
        };
        let narrower = OperationShape::Gemm {
            m: 4,
            n: 32,
            k: 16,
        };
        let attention = OperationShape::Attention {
            queries: 16,
            keys: 24,
            heads: 8,
            head_dim: 256,
        };

        session
            .prepare(|| {
                note_shape(gemm)?;
                note_shape(attention)?;
                note_shape(gemm)
            })
            .expect("prepare");
        assert_eq!(
            session.prepared_shapes(),
            vec![gemm, attention],
            "the list is first-use order with repeats folded"
        );

        session
            .run(|| {
                note_shape(gemm)?;
                note_shape(attention)
            })
            .expect("a traversal the prepare pass covered");

        let refused = session
            .run(|| note_shape(narrower))
            .expect_err("a shape prepare never ran must be refused");
        let message = refused.to_string();
        assert!(
            message.contains("GEMM m=4 n=32 k=16"),
            "the refusal must name the shape: {message}"
        );
        assert!(
            message.contains("prepare traversal"),
            "the refusal must name what was missing: {message}"
        );

        session
            .prepare(|| note_shape(narrower))
            .expect("a new prepare pass");
        assert_eq!(
            session.prepared_shapes(),
            vec![narrower],
            "a prepare pass replaces the list rather than extending it"
        );
    }

    /// Capture is what makes retention matter: a resource prepared for the
    /// recorded traversal has to outlive the graph, and nothing prepared
    /// outside capture is held.
    #[test]
    fn capture_retention_brackets_the_recorded_traversal() {
        assert!(!is_capturing());
        let outside: Rc<u32> = Rc::new(1);
        retain_resource(&outside);

        begin_capture_retention();
        assert!(is_capturing());
        let inside: Rc<u32> = Rc::new(2);
        retain_resource(&inside);
        retain_resource(&inside);
        let retained = end_capture_retention();

        assert!(!is_capturing());
        assert_eq!(
            retained.len(),
            1,
            "only what was prepared under capture is kept, and only once"
        );
        assert!(Rc::ptr_eq(
            &retained[0].clone().downcast::<u32>().expect("the retained u32"),
            &inside
        ));
    }

    /// An allocation outside a session is a driver allocation; it must not
    /// consume arena a later traversal would reuse.
    #[test]
    fn an_allocation_outside_a_session_leaves_the_arena_alone() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let shape = apxinf_core::Shape::new(vec![4, 8]);
        let session = ExecutionSession::with_capacity(1 << 20, ctx.device_id()).expect("session");

        session
            .prepare(|| ctx.allocate_output(shape.clone(), DType::BF16).map(|_| ()))
            .expect("a prepared traversal");
        let inside = session.workspace().used();
        assert!(inside > 0, "the session must have taken arena for the output");

        ctx.allocate_output(shape, DType::BF16)
            .expect("an eager output");
        ctx.synchronize().expect("synchronize");
        assert_eq!(
            session.workspace().used(),
            inside,
            "an allocation outside a session consumed session arena"
        );
    }
}
