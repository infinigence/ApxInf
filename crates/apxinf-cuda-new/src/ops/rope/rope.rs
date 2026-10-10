use apxinf_core::Result;

use super::contracts::{normalize, RopeArgs};
use super::launch;
use crate::CudaContext;

/// Splits a packed QKV projection, optionally applying rotary embedding.
pub fn rope(ctx: &CudaContext, args: RopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}
