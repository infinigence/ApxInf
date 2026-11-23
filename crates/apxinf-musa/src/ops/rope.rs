//! Packed-QKV split and rotary position embedding contracts.

use apxinf_core::{Error, Result, Tensor};

use crate::kernels::attention as attention_kernels;
use crate::kernels::rope as rope_kernels;
use crate::MusaContext;

/// Which packed-QKV split the arguments describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeSemantic {
    /// Grouped-query split with rotary embedding on Q and K.
    SplitQkvRope,
    /// Vision MHA split with an optional bias and no rotation.
    SplitQkvBias,
}

/// Splits a packed QKV projection into separate Q, K and V buffers.
///
/// `k` and `v` may point either at fresh per-call buffers (`kv_output_offset`
/// zero) or into a KV cache, in which case `kv_output_offset` is the row at
/// which this call's tokens are appended and the kernel writes in place.
pub struct RopeArgs<'a> {
    pub semantic: RopeSemantic,
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub position_offset: usize,
    pub kv_output_offset: usize,
}

pub fn rope(ctx: &MusaContext, args: RopeArgs<'_>) -> Result<()> {
    match args.semantic {
        RopeSemantic::SplitQkvRope => match args.kv_output_offset {
            0 => rope_kernels::split_qkv_apply_bf16_into(
                ctx,
                args.qkv,
                args.bias,
                args.q,
                args.k,
                args.v,
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.theta,
                args.position_offset,
            ),
            // The cache case: the kernel writes K/V at the offset and returns
            // Q alone, so the borrowed k/v view is the cache itself.
            _ => {
                if args.kv_heads != 1 {
                    return Err(Error::Other(format!(
                        "MUSA rope cache write requires kv_heads == 1, got {}",
                        args.kv_heads
                    )));
                }
                rope_kernels::apply_q_write_kv_bf16_into(
                    ctx,
                    args.qkv,
                    args.bias,
                    args.q,
                    args.q_heads,
                    args.kv_heads,
                    args.head_dim,
                    args.theta,
                    args.position_offset,
                    args.k,
                    args.v,
                    args.kv_output_offset,
                )
            }
        },
        RopeSemantic::SplitQkvBias => {
            if args.kv_output_offset != 0 {
                return Err(Error::Other(
                    "MUSA qkv split with a bias has no KV-cache form".into(),
                ));
            }
            attention_kernels::split_qkv_bias_bf16_into(
                ctx, args.qkv, args.bias, args.q, args.k, args.v, args.q_heads, args.head_dim,
            )
        }
    }
}
