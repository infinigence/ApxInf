//! Model/phase-level preparation and CUDA Graph capture.
//!
//! Fixed operators only enqueue work on the active stream. A phase owns this
//! wrapper when it also contains persistent GEMM or Attention executions.

use apxinf_core::Result;

use crate::{capture, CapturedGraph, CudaContext, ExecutionSession};

/// A prepared model phase with one captured graph.
///
/// The phase, rather than an individual fixed operator, owns graph lifetime.
/// The execution session is retained so persistent GEMM/Attention executions
/// and their workspace live at least as long as the graph.
pub struct PreparedPhase {
    session: ExecutionSession,
    graph: CapturedGraph,
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
        ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
        let graph = crate::ops::with_session(&session, || capture(ctx, operation))?;
        Ok(Self { session, graph })
    }

    /// Replay the prepared phase on its original stream.
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    /// Access the session for phase-owned preparation or diagnostics.
    pub fn session(&self) -> &ExecutionSession {
        &self.session
    }
}
