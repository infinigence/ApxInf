use apxinf_core::Result;

use super::contracts::{execute, normalize, Conv1dArgs, Normalized};
use crate::CudaContext;

/// Run a channel-last Conv1d or ConvTranspose1d into `args.out`.
pub fn conv1d(ctx: &CudaContext, args: Conv1dArgs<'_>) -> Result<()> {
    let normalized: Normalized = normalize(ctx, args)?;
    execute(ctx, normalized)
}
