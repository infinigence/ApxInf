use apxinf_core::Result;

use super::contracts::{execute, normalize, Conv2dArgs, Normalized};
use crate::CudaContext;

/// Run a channel-last Conv2d (groups 1, dilation 1) into `args.out`.
pub fn conv2d(ctx: &CudaContext, args: Conv2dArgs<'_>) -> Result<()> {
    let normalized: Normalized = normalize(ctx, args)?;
    execute(ctx, normalized)
}
