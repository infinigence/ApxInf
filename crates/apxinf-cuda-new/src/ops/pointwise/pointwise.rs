use apxinf_core::Result;

use super::contracts::{normalize, PointwiseArgs};
use super::launch;
use crate::CudaContext;

/// Runs one element-wise semantic. The semantic selects which of the optional
/// bindings participate; see [`PointwiseArgs`].
pub fn pointwise(ctx: &CudaContext, args: PointwiseArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}
