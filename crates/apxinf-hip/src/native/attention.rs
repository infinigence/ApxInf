//! Dense non-causal attention, the only kind pi05 uses.

use apxinf_core::{Error, Result, Tensor};

use super::{bf16_ptr, int, long, require_numel};
use crate::ffi::{self, check};
use crate::runtime::HipContext;

/// Extents of one attention call. Query head `h` reads KV head
/// `h / (q_heads / kv_heads)`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    pub batch: usize,
    pub q_tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Keys every query attends to, from row 0.
    pub kv_tokens: usize,
}

/// `out = softmax(q kᵀ / sqrt(head_dim)) v`, no mask.
///
/// `q` and `out` are `[batch, q_tokens, q_heads, head_dim]`. `k` and `v` are
/// `[batch, rows, kv_heads, head_dim]` with `rows >= kv_tokens`: a cache may
/// hold more rows than the call attends, and only the first `kv_tokens` of
/// each batch are read. Any leading-dimension grouping of these element
/// counts is accepted.
pub fn attention(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    out: &Tensor,
) -> Result<()> {
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
    } = *geometry;
    if batch == 0 || kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0 || kv_tokens == 0 {
        return Err(Error::Contract("apxinf-hip attention: invalid geometry"));
    }
    let q_elements = batch * q_tokens * q_heads * head_dim;
    for t in [q, out] {
        require_numel(t, q_elements, || {
            format!("[{batch}, {q_tokens}, {q_heads}, {head_dim}]")
        })?;
    }
    let kv_row = kv_heads * head_dim;
    let per_batch = k.numel() / batch;
    if k.shape() != v.shape()
        || !k.numel().is_multiple_of(batch)
        || !per_batch.is_multiple_of(kv_row)
    {
        return Err(Error::ShapeMismatch {
            expected: format!("K and V of [{batch}, rows, {kv_heads}, {head_dim}]"),
            got: format!("{} and {}", k.shape(), v.shape()),
        });
    }
    if per_batch / kv_row < kv_tokens {
        return Err(Error::Other(format!(
            "apxinf-hip attention: {kv_tokens} keys requested, the cache holds {}",
            per_batch / kv_row
        )));
    }
    let (pq, pk, pv, po) = (
        bf16_ptr(ctx, q)?,
        bf16_ptr(ctx, k)?,
        bf16_ptr(ctx, v)?,
        bf16_ptr(ctx, out)?,
    );
    check("attention", unsafe {
        ffi::apxinf_hip_native_attention(
            pq,
            pk,
            pv,
            po,
            int(batch)?,
            int(q_tokens)?,
            int(q_heads)?,
            int(kv_heads)?,
            int(head_dim)?,
            int(kv_tokens)?,
            long(per_batch)?,
            long(kv_row)?,
            1.0 / (head_dim as f32).sqrt(),
            ctx.stream(),
        )
    })
}
