use apxinf_core::Result;

use super::contracts::{execute, normalize, Normalized, RelativeAttentionArgs};
use crate::CudaContext;

/// Run relative-position self-attention into `args.out`.
pub fn relative_attention(ctx: &CudaContext, args: RelativeAttentionArgs<'_>) -> Result<()> {
    let normalized: Normalized = normalize(ctx, args)?;
    execute(ctx, normalized)
}
