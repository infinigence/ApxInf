//! Model/phase-level preparation, eager execution, and CUDA Graph capture.
//!
//! Fixed operators only enqueue work on the active stream. A phase owns the
//! prepared GEMM or Attention executions and the deterministic arena offsets
//! used by one fixed-shape traversal. The same phase can run kernel-by-kernel
//! or be captured and replayed as a CUDA Graph.

use apxinf_core::Result;

use crate::{capture, CapturedGraph, CudaContext, ExecutionSession};

/// One fixed-shape prepared model phase.
///
/// Preparation fixes the operator sequence, specs, bindings, stream, and arena
/// offsets. Later eager runs and graph capture must traverse that exact plan;
/// a cache miss is an error rather than an opportunity to grow the plan.
/// Runtime values may still change through the contents of fixed-address input
/// or control buffers.
pub struct PreparedPhase {
    /// Declared first so it drops first: tearing the graph down releases the
    /// captured nodes before the arena those nodes may address.
    ///
    /// `CapturedGraph::drop` destroys the exec, which CUDA defers until that
    /// exec's in-flight launches finish, and then drops the resources retained
    /// during capture. Each retained `Execution::drop` synchronizes the stream
    /// before releasing its buffers, so by the time `session` below is dropped
    /// no replay of this phase can still be running.
    graph: Option<CapturedGraph>,
    /// Held only to outlive `graph`.
    ///
    /// Captured nodes address this arena only for operators that took the
    /// `workspace::output_buffer` path; one that writes into a caller-owned
    /// tensor does not, and such a phase leaves the arena untouched (qwen38's
    /// decode phases are entirely of that kind — see `workspace_used`). The
    /// field is kept regardless, so the arena cannot outlive its graph in
    /// either case.
    session: ExecutionSession,
    /// Arena high-water mark of the preparation traversal.
    ///
    /// Snapshotted rather than read on demand: every session traversal resets
    /// the arena offset on entry, and the prepared GEMM/Attention executions
    /// hold the workspace views they were given during preparation, so they do
    /// not re-allocate when the phase is captured. Reading
    /// `session.workspace().used()` afterwards therefore reports nearly zero
    /// and says nothing about what the phase needs.
    prepared_workspace_used: usize,
}

impl PreparedPhase {
    /// Prepare one fixed-shape traversal without requiring CUDA Graph support.
    ///
    /// The operation executes once to select native implementations and bind
    /// their resources. Subsequent [`Self::run_eager`] calls reuse only those
    /// prepared executions, so their number stays fixed after this returns.
    pub fn prepare(
        ctx: &CudaContext,
        session: ExecutionSession,
        operation: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        crate::ops::prepare_with_session(&session, operation)?;
        // Read before another traversal resets the arena offset.
        let prepared_workspace_used = session.workspace().used();
        ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
        Ok(Self {
            graph: None,
            session,
            prepared_workspace_used,
        })
    }

    /// Run `operation` once to prepare native executions, then again to
    /// capture the same phase.
    pub fn prepare_and_capture(
        ctx: &CudaContext,
        session: ExecutionSession,
        mut operation: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let mut phase = Self::prepare(ctx, session, &mut operation)?;
        phase.capture(ctx, operation)?;
        Ok(phase)
    }

    /// Execute the prepared phase kernel-by-kernel.
    ///
    /// This is planned eager execution: it resets the deterministic arena and
    /// requires the same prepared operator sequence and bindings. It never
    /// prepares or caches a new native execution.
    pub fn run_eager(&self, operation: impl FnOnce() -> Result<()>) -> Result<()> {
        crate::ops::with_session(&self.session, operation)
    }

    /// Capture the prepared phase for later [`Self::replay`] calls.
    ///
    /// Capture is optional; callers may use [`Self::run_eager`] for the whole
    /// lifetime of the phase. Capturing twice is rejected so graph ownership
    /// and its fixed bindings stay unambiguous.
    pub fn capture(
        &mut self,
        ctx: &CudaContext,
        operation: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if self.graph.is_some() {
            return Err(apxinf_core::Error::Other(
                "prepared phase already has a captured CUDA Graph".into(),
            ));
        }
        let graph = crate::ops::with_session(&self.session, || capture(ctx, operation))?;
        self.graph = Some(graph);
        Ok(())
    }

    /// Replay the prepared phase on its original stream.
    pub fn replay(&self) -> Result<()> {
        self.graph
            .as_ref()
            .ok_or_else(|| {
                apxinf_core::Error::Other("prepared phase has not captured a CUDA Graph".into())
            })?
            .replay()
    }

    /// Whether this prepared phase also owns a captured CUDA Graph.
    pub fn is_captured(&self) -> bool {
        self.graph.is_some()
    }

    /// Execute the phase eagerly with the same prepared resources used by the
    /// captured graph. This is primarily useful for eager/captured parity
    /// checks without changing selected operator recipes.
    pub fn run(&self, operation: impl FnOnce() -> Result<()>) -> Result<()> {
        crate::ops::with_session(&self.session, operation)
    }

    /// Arena bytes this phase actually needs.
    ///
    /// Measured during preparation, which is the traversal that sub-allocates;
    /// see [`Self::prepared_workspace_used`]'s field documentation. Compare
    /// against [`Self::workspace_capacity`] to size the reservation.
    pub fn workspace_used(&self) -> usize {
        self.prepared_workspace_used
    }

    /// Arena bytes reserved by `cudaMalloc` when the phase was built.
    pub fn workspace_capacity(&self) -> usize {
        self.session.workspace().capacity()
    }
}
