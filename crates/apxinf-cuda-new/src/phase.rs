//! Model/phase-level preparation and CUDA Graph capture.
//!
//! Fixed operators only enqueue work on the active stream. A phase owns this
//! wrapper when it also contains persistent GEMM or Attention executions.

use apxinf_core::Result;

use crate::{capture, CapturedGraph, CudaContext, ExecutionSession};

/// A prepared model phase with one captured graph.
///
/// The phase, rather than an individual fixed operator, owns graph lifetime.
pub struct PreparedPhase {
    /// Declared first so it drops first: tearing the graph down releases the
    /// captured nodes before the arena those nodes may address.
    ///
    /// `CapturedGraph::drop` destroys the exec, which CUDA defers until that
    /// exec's in-flight launches finish, and then drops the resources retained
    /// during capture. Each retained `Execution::drop` synchronizes the stream
    /// before releasing its buffers, so by the time `session` below is dropped
    /// no replay of this phase can still be running.
    graph: CapturedGraph,
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
    /// Run `operation` once to prepare native executions, then again to
    /// capture the same phase.
    pub fn prepare_and_capture(
        ctx: &CudaContext,
        session: ExecutionSession,
        mut operation: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        crate::ops::prepare_with_session(&session, &mut operation)?;
        // Read before capture: `with_session` below resets the offset.
        let prepared_workspace_used = session.workspace().used();
        ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
        let graph = crate::ops::with_session(&session, || capture(ctx, operation))?;
        Ok(Self {
            graph,
            session,
            prepared_workspace_used,
        })
    }

    /// Replay the prepared phase on its original stream.
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
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
