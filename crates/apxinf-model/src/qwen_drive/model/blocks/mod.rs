//! BF16 is the only maintained computation variant.
pub(crate) mod bf16;
use crate::qwen_drive::backend::RuntimeBackend;
use apxinf_core::{Result, Tensor};
use std::sync::Arc;
/// Facts needed to replay a stateful GDN block; no weight or runner dependency.
pub(crate) struct GdnRequest<'a> {
    pub backend: &'a Arc<RuntimeBackend>,
    pub layer: usize,
    pub layer_count: usize,
    pub parity: usize,
    pub decode: bool,
    pub decode_step: usize,
}
/// Execute the supplied computation eagerly or replay its captured kernels.
/// The bool reports replay without running the host body, so Blocks advances
/// its own double-buffer parity exactly once.
pub(crate) trait GdnExecution {
    fn run(
        &mut self,
        request: &GdnRequest<'_>,
        input: Tensor,
        eager: &mut dyn FnMut(Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, bool)>;
}
/// Execute one phase of a whole-model direct plan. The runner may reuse a
/// workspace between these sequential computations, so cross-phase values are
/// in persistent storage before the callback returns.
pub(crate) trait DirectExecution {
    fn run(&mut self, operation: &mut dyn FnMut() -> Result<Tensor>) -> Result<Tensor>;
}
