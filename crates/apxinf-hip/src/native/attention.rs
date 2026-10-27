//! Dense non-causal attention, the only kind pi05 uses.
//!
//! Two implementations behind one contract:
//!
//! - [`AttentionKernel::Ck`] and [`AttentionKernel::CkSplitKv`]: Composable
//!   Kernel's flash-attention forward (vendored, `kernels/ck/VENDOR.md`).
//!   WMMA, tiled; the probabilities are rounded to BF16 before the value
//!   product, as FlashAttention-2 on CUDA does.
//! - [`AttentionKernel::Reference`]: apxinf's own single-pass kernel. F32
//!   scores and probabilities, slow; the test reference, and the fallback for
//!   shapes no CK instance covers.
//!
//! [`attention_kernel`] picks the default for a geometry; callers may force any
//! kernel, which is how tests compare them on the same inputs.

use apxinf_core::{DType, Error, Result, Tensor};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernel {
    /// apxinf's own kernel.
    Reference,
    /// CK flash-attention forward.
    Ck,
    /// CK split-KV forward: keys split across `splits` workgroups per query
    /// tile, then combined. Needs [`attention_scratch_elements`] F32 scratch.
    CkSplitKv { splits: usize },
}

/// Key splits for CK split-KV. 4 measured fastest for pi05's action expert
/// (10 queries x 8 heads over 532 keys) on gfx1151: 24 us, against 57 us
/// unsplit; 2 and 8 splits were within 15%.
const SPLITKV_SPLITS: usize = 4;

/// The default kernel for a geometry. CK covers head dims up to 256 in
/// multiples of 8 (head dim 72 runs on the padded 128 instances); the vendored
/// split-KV instances are head dim 256 only. Split-KV pays off when few query
/// rows face many keys: one 16-row query tile per head cannot fill the GPU.
pub fn attention_kernel(geometry: &AttentionGeometry) -> AttentionKernel {
    let g = geometry;
    if g.head_dim > 256 || !g.head_dim.is_multiple_of(8) {
        return AttentionKernel::Reference;
    }
    if g.head_dim == 256 && g.q_tokens <= 16 && g.kv_tokens >= 256 {
        return AttentionKernel::CkSplitKv {
            splits: SPLITKV_SPLITS,
        };
    }
    AttentionKernel::Ck
}

/// F32 scratch elements `kernel` needs for `geometry`: split-KV's partial
/// log-sum-exps and outputs; zero for the other kernels. The log-sum-exp block
/// is padded to 64 elements so the output block starts 256-byte aligned.
pub fn attention_scratch_elements(geometry: &AttentionGeometry, kernel: AttentionKernel) -> usize {
    match kernel {
        AttentionKernel::CkSplitKv { splits } => {
            let rows = geometry.batch * geometry.q_heads * splits * geometry.q_tokens;
            rows.next_multiple_of(64) + rows * geometry.head_dim
        }
        _ => 0,
    }
}

/// `out = softmax(q kᵀ / sqrt(head_dim)) v`, no mask, computed by `kernel`.
///
/// `q` and `out` are `[batch, q_tokens, q_heads, head_dim]`. `k` and `v` are
/// `[batch, rows, kv_heads, head_dim]` with `rows >= kv_tokens`: a cache may
/// hold more rows than the call attends, and only the first `kv_tokens` of
/// each batch are read. Any leading-dimension grouping of these element
/// counts is accepted. `scratch` is an F32 tensor of at least
/// [`attention_scratch_elements`] elements, or `None` when that is zero.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &HipContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    geometry: &AttentionGeometry,
    kernel: AttentionKernel,
    scratch: Option<&Tensor>,
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
    let scale = 1.0 / (head_dim as f32).sqrt();
    match kernel {
        AttentionKernel::Reference => check("attention", unsafe {
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
                scale,
                ctx.stream(),
            )
        }),
        AttentionKernel::Ck => {
            // With one KV head and no mask, the query heads are independent rows
            // over the same keys: [tokens, heads, d] is [tokens * heads, 1, d]
            // in memory. One head of more rows tiles better (prefix: 106 us
            // against 124 us on gfx1151).
            let (rows, heads) = if kv_heads == 1 {
                (q_tokens * q_heads, 1)
            } else {
                (q_tokens, q_heads)
            };
            check("ck attention", unsafe {
                ffi::apxinf_hip_ck_attention(
                    pq,
                    pk,
                    pv,
                    po,
                    int(batch)?,
                    int(rows)?,
                    int(heads)?,
                    int(kv_heads)?,
                    int(head_dim)?,
                    int(kv_tokens)?,
                    long(per_batch)?,
                    long(kv_row)?,
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    scale,
                    ctx.stream(),
                )
            })
        }
        AttentionKernel::CkSplitKv { splits } => {
            let needed = attention_scratch_elements(geometry, kernel);
            let scratch = scratch.ok_or(Error::Contract(
                "apxinf-hip attention: split-KV needs scratch",
            ))?;
            if scratch.dtype() != DType::F32 || scratch.numel() < needed {
                return Err(Error::ShapeMismatch {
                    expected: format!("at least {needed} f32 scratch elements"),
                    got: format!("{} {}", scratch.shape(), scratch.dtype()),
                });
            }
            let base = ctx.ptr(scratch)?;
            let lse_elements = (batch * q_heads * splits * q_tokens).next_multiple_of(64);
            // SAFETY: the scratch holds `needed` floats, which is the padded
            // log-sum-exp block plus the output block.
            let o_acc = unsafe { base.cast::<f32>().add(lse_elements) };
            check("ck split-KV attention", unsafe {
                ffi::apxinf_hip_ck_attention(
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
                    int(splits)?,
                    base,
                    o_acc.cast(),
                    scale,
                    ctx.stream(),
                )
            })
        }
    }
}
