//! The W8A8 (dynamic INT8) operator surface.
//!
//! PI0.5's INT8 route reaches the device through these names rather than
//! through the generic GEMM contract's `GemmQuantization::W8A8` arm, because
//! the operand contract differs in one place that costs time on every call:
//! upstream's typed GEMM carries f32 row scales of `[rows]`, while this
//! device's `s8mma` tiles read FP16 and index a vector with [`W8A8_SCALE_PAD`]
//! trailing zeros. Serving the generic arm would convert scales per GEMM —
//! over a thousand calls per traversal — so the quantizer here writes exactly
//! the padded FP16 vector the tile reads, and the GEMM consumes it as produced.
//!
//! Everything else is the CUDA route's shape: weights are output-major INT8
//! with one f32 dequantization scale per output channel, an FP16 rendering of
//! those scales is packed once at load, and the result is BF16. [`W8A8WeightView`]
//! borrows the values, both scale vectors and the dimensions together so a call
//! site cannot mismatch them.

use apxinf_core::{Result, Tensor};

use crate::kernels::elementwise;
use crate::kernels::gemm as gemm_kernels;
use crate::MusaContext;

pub use crate::kernels::gemm::{
    w8a8_decode_scope as decode_scope, w8a8_decode_supported as decode_supported, W8A8Activation,
    W8A8DecodeScope, W8A8Layout, W8A8ScaleMode, W8A8WeightView, W8A8_SCALE_PAD,
};

/// Row-quantize a BF16 `[rows, input_dim]` activation.
///
/// `inv_c` is the activation half of a per-channel equalization fold:
/// `input_dim` f32 factors the row is multiplied by before its amax is
/// measured, paired with a weight whose contraction rows were folded by `c` at
/// pack time. `None` quantizes the activation as-is.
pub fn quantize(
    ctx: &MusaContext,
    activation: &Tensor,
    inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    gemm_kernels::quantize_w8a8_activation(ctx, activation, inv_c)
}

/// INT8 GEMM over an already-quantized activation, into a caller-provided
/// BF16 `[rows, output_dim]` output.
pub fn gemm_quantized(
    ctx: &MusaContext,
    activation: &W8A8Activation,
    weight: W8A8WeightView<'_>,
    out: &mut Tensor,
) -> Result<()> {
    gemm_kernels::gemm_quantized_w8a8_into(ctx, activation, weight, out)
}

/// `out = activation @ weight`, quantizing the activation on the way.
///
/// The two-projection case that shares one activation (a transformer's gate and
/// up) is served better by [`quantize`] once and [`gemm_quantized`] twice.
pub fn gemm(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
    out: &mut Tensor,
) -> Result<()> {
    let quantized = quantize(ctx, activation, weight.equalization_inv_c)?;
    gemm_quantized(ctx, &quantized, weight, out)
}

/// `out = geglu(gate_up) @ weight`, with the activation quantized as it is
/// produced.
///
/// `gate_up` is the packed `[rows, 2 * input_dim]` projection output. The
/// fused producer applies the weight view's equalization factors while it
/// measures each row, so the down projection's per-channel fold costs no extra
/// pass; widths and alignments the fused kernel does not take fall back to a
/// materialized GeGLU and [`gemm`], which computes the same thing.
pub fn gemm_geglu(
    ctx: &MusaContext,
    gate_up: &Tensor,
    weight: W8A8WeightView<'_>,
    out: &mut Tensor,
) -> Result<()> {
    let quantized = gemm_kernels::quantize_w8a8_geglu_activation(
        ctx,
        gate_up,
        weight.equalization_inv_c,
    )?;
    gemm_quantized(ctx, &quantized, weight, out)
}

/// Whether a tile serves a `[rows, k] @ [k, n]` contraction.
///
/// `rows` is not consulted: the tiles mask a partial row tile, so only `n` and
/// `k` decide eligibility. The model asks this before committing a projection
/// to INT8 — π0.5's patch embedding, action input and action head reach shapes
/// no tile serves and must keep BF16 arithmetic.
pub fn tile_serves(n: usize, k: usize) -> bool {
    gemm_kernels::w8a8_tile_serves(n, k)
}

/// Multiply a BF16 matrix by a per-channel f32 scale vector.
///
/// The activation half of the equalization fold for a projection whose input
/// has no RMS norm in front of it; the fused quantizer applies the same factor
/// itself, so this is only for the BF16 fallback combination.
pub fn mul_channels(
    ctx: &MusaContext,
    input: &Tensor,
    channel_scale: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    elementwise::mul_channels_bf16_into(ctx, input, channel_scale, out)
}

#[cfg(test)]
mod tests {
    use super::tile_serves;

    #[test]
    fn tiles_serve_columns_and_contractions_only_where_one_exists() {
        // π0.5's ViT and language projections: tile widths are 64/128/256 and
        // the contraction must be a multiple of 64.
        assert!(tile_serves(1152, 1152));
        assert!(tile_serves(4352, 4352));
        assert!(tile_serves(2048, 64));
        // The ViT patch embedding (`k = 588`) and the action head (`k = 32`,
        // `n = 32`) reach shapes no tile serves; the model keeps them BF16.
        assert!(!tile_serves(1152, 588));
        assert!(!tile_serves(32, 32));
        // An unpadded ViT MLP is 4304 wide, which is not a tile multiple.
        assert!(!tile_serves(4304, 4304));
        assert!(!tile_serves(0, 64));
    }
}
