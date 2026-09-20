use apxinf_core::Result;

use super::contracts::{normalize, AttentionArgs};
use super::execution;
use crate::CudaContext;

pub fn attention(ctx: &CudaContext, args: AttentionArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
