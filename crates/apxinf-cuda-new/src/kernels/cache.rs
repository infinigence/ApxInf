//! Legacy `kernels::cache` names over cuda-new cache operators.

use apxinf_core::{Result, Tensor};

use crate::{ops, CudaContext};

/// `reserve_prefix_bf16`: allocate a `[total_rows, cols]` cache seeded with
/// `prefix` and zero-padded to capacity.
pub fn reserve_prefix_bf16(
    ctx: &CudaContext,
    prefix: &Tensor,
    total_rows: usize,
) -> Result<Tensor> {
    ops::reserve_prefix(ctx, prefix, total_rows)
}
