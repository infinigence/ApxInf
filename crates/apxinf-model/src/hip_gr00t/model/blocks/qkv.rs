//! How a DiT block's self-attention q/k/v is projected.
//!
//! `fused` (the default) concatenates the three weight matrices along N at
//! load time so one GEMM produces `[tokens, 3 * width]`, and attention reads
//! the q/k/v column sections in place through `attention_qkv_packed`: one
//! GEMM, one attention kernel, no split pass. Measured on gfx1151 it is
//! 0.4-0.6 ms faster than three separate GEMMs per forward (0.7 ms under
//! graph capture) and bit-identical in output; the packed weights displace,
//! not duplicate, the separate ones.
//!
//! `serial` keeps the original three `gemm_bias_activation` calls on the
//! context stream. It stays selectable as the rollback path and as the
//! baseline for A/B measurement.
//!
//! The layout is chosen at upload time by `APXINF_GR00T_QKV`
//! (weights/bf16.rs); this forward dispatches on which layout the weights
//! actually carry, so the compute path can never disagree with what was
//! packed -- captured and eager runs see the same weights.
//!
//! Only self-attending DiT blocks use this. Cross-attending blocks take their
//! K and V from `CrossAttentionCache`, so they project Q alone and have no
//! independent work to fuse.

use apxinf_core::{DType, Result, Tensor};
use apxinf_hip::native::{self, Activation, HipArena};
use apxinf_hip::HipContext;

use crate::hip_gr00t::weights::{DitBlock, Linear};

/// How a self-attending block's projection came out, and what attention
/// consumes.
pub enum QkvProjection {
    /// Three `[tokens, heads, head_dim]` tensors: the serial path, and a
    /// cross-attending block's lone Q over cached K and V.
    Separate(Tensor, Tensor, Tensor),
    /// One `[tokens, 3 * width]` GEMM output, bias already applied, whose
    /// column sections are q, k and v. Attention reads them in place through
    /// `native::attention_qkv_packed`; the split kernel that would copy them
    /// out is what the fused mode no longer has.
    Packed(Tensor),
}

fn project_one(
    ctx: &HipContext,
    arena: &HipArena,
    linear: &Linear,
    normalized: &Tensor,
    tokens: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // The extra allocation mirrors the shipped path's arena layout. It is not
    // read, but keeping it means `serial` allocates exactly what it did before
    // this experiment, so the comparison is against the real baseline.
    let _raw = arena.tensor(vec![tokens, width], DType::BF16)?;
    let biased = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::gemm_bias_activation(
        ctx,
        normalized,
        &linear.weight,
        linear.bias.as_ref(),
        Activation::None,
        &biased,
    )?;
    biased.reshape(vec![tokens, heads, head_dim])
}

/// Project q, k and v for a self-attending DiT block. Attention consumes the
/// result through the matching [`QkvProjection`] arm.
pub fn project_qkv(
    ctx: &HipContext,
    arena: &HipArena,
    block: &DitBlock,
    normalized: &Tensor,
    tokens: usize,
    width: usize,
    heads: usize,
    head_dim: usize,
) -> Result<QkvProjection> {
    // The uploaded weights carry the layout `APXINF_GR00T_QKV` chose at load
    // time: packed when fused (the default), three separate projections when
    // serial. Dispatching here on what the weights hold keeps the forward and
    // the packing from ever disagreeing.
    match block.qkv_fused.as_ref() {
        None => Ok(QkvProjection::Separate(
            project_one(ctx, arena, &block.q, normalized, tokens, width, heads, head_dim)?,
            project_one(ctx, arena, &block.k, normalized, tokens, width, heads, head_dim)?,
            project_one(ctx, arena, &block.v, normalized, tokens, width, heads, head_dim)?,
        )),
        Some(packed) => {
            let fused = arena.tensor(vec![tokens, 3 * width], DType::BF16)?;
            native::gemm_bias_activation(
                ctx,
                normalized,
                &packed.weight,
                packed.bias.as_ref(),
                Activation::None,
                &fused,
            )?;
            // The bias is in the GEMM epilogue and attention takes the q/k/v
            // sections as strided reads, so there is nothing to split and no
            // q/k/v tensors to allocate: the GEMM output is the projection.
            Ok(QkvProjection::Packed(fused))
        }
    }
}
