use apxinf_core::Result;

use super::contracts::{normalize, normalize_packed_qkv, AttentionArgs, PackedQkvAttentionArgs};
use super::execution;
use crate::CudaContext;

pub fn attention(ctx: &CudaContext, args: AttentionArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}

pub fn packed_qkv_attention(ctx: &CudaContext, args: PackedQkvAttentionArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize_packed_qkv(ctx, args)?)
}
