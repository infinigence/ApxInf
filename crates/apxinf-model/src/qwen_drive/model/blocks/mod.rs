//! BF16 is the only maintained computation variant.
pub(crate) mod bf16;
use apxinf_core::{Result, Tensor};

/// The runner may reuse a workspace between these sequential computations.
/// Cross-phase values are in persistent storage before the callback returns.
pub(crate) trait DirectExecution {
    fn run(&mut self, operation: &mut dyn FnMut() -> Result<Tensor>) -> Result<Tensor>;
}
