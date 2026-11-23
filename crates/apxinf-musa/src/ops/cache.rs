//! KV-cache and row-manipulation contracts.

use apxinf_core::{Result, Tensor};

use crate::kernels::cache as cache_kernels;
use crate::kernels::elementwise;
use crate::MusaContext;

/// Allocate a persistent KV prefix and copy `prefix` into its first rows.
///
/// The tail deliberately keeps the arena's previous contents: `attention` reads
/// only the first `key_tokens` rows, which is exactly why the copy is allowed to
/// stop there.
pub fn reserve_prefix(ctx: &MusaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    cache_kernels::reserve_prefix_bf16(ctx, prefix, total_rows)
}

/// Concatenate two 2-D BF16 tensors along rows.
pub fn concat_rows(ctx: &MusaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    elementwise::concat_rows_bf16(ctx, first, second)
}
