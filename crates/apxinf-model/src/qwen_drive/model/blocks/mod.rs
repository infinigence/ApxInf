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
/// `Some(parity)` asks the eager body to bind a graph to an explicit GDN state
/// pair without advancing host state. The bool reports a successful replay, so
/// Blocks advances its own double-buffer parity exactly once.
pub(crate) trait GdnExecution {
    fn run(
        &mut self,
        request: &GdnRequest<'_>,
        input: Tensor,
        eager: &mut dyn FnMut(Tensor, Option<usize>) -> Result<Tensor>,
    ) -> Result<(Tensor, bool)>;
}
