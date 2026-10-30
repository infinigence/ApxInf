//! Normalizations, plain and fused with the residual add that precedes them.
//!
//! Semantics are cuda-new's (see `kernels/norm.hip`). In the fused kernels,
//! `hidden` is the residual sum rounded to BF16 and `normalized` is computed
//! from that stored value.

use apxinf_core::{Result, Tensor};

use super::{bf16_ptr, int, long, optional_ptr, require_numel, rows_cols, same_shape};
use crate::ffi::{self, check, DTYPE_BF16};
use crate::runtime::HipContext;

/// `out = x * rsqrt(mean(x²) + eps) * weight`, per row of `[..., cols]`.
pub fn rms_norm(
    ctx: &HipContext,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(x)?;
    same_shape(x, out)?;
    require_numel(weight, cols, || format!("[{cols}]"))?;
    let (px, pw, py) = (
        bf16_ptr(ctx, x)?,
        bf16_ptr(ctx, weight)?,
        bf16_ptr(ctx, out)?,
    );
    check("rms_norm", unsafe {
        ffi::apxinf_hip_rms_norm(
            DTYPE_BF16,
            px,
            pw,
            py,
            long(rows)?,
            long(cols)?,
            eps,
            ctx.stream(),
        )
    })
}

/// `out = (x - mean) * rsqrt(var + eps) * weight + bias`, per row.
pub fn layer_norm(
    ctx: &HipContext,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(x)?;
    same_shape(x, out)?;
    require_numel(weight, cols, || format!("[{cols}]"))?;
    require_numel(bias, cols, || format!("[{cols}]"))?;
    let (px, pw, pb, py) = (
        bf16_ptr(ctx, x)?,
        bf16_ptr(ctx, weight)?,
        bf16_ptr(ctx, bias)?,
        bf16_ptr(ctx, out)?,
    );
    check("layer_norm", unsafe {
        ffi::apxinf_hip_native_layer_norm(
            px,
            pw,
            pb,
            py,
            long(rows)?,
            int(cols)?,
            eps,
            ctx.stream(),
        )
    })
}

/// Adaptive RMSNorm: `out = rms(x) * (1 + scale) + shift`, with `style`'s
/// first `2 * cols` elements holding scale then shift.
pub fn ada_rms_norm(
    ctx: &HipContext,
    x: &Tensor,
    style: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(x)?;
    same_shape(x, out)?;
    style_covers(style, 2 * cols)?;
    let (px, ps, py) = (
        bf16_ptr(ctx, x)?,
        bf16_ptr(ctx, style)?,
        bf16_ptr(ctx, out)?,
    );
    check("ada_rms_norm", unsafe {
        ffi::apxinf_hip_native_ada_rms_norm(px, ps, py, long(rows)?, int(cols)?, eps, ctx.stream())
    })
}

/// `hidden = projection + residual (+ bias)`, rounded once.
pub fn bias_residual(
    ctx: &HipContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    hidden: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(projection)?;
    same_shape(projection, residual)?;
    same_shape(projection, hidden)?;
    let pb = optional_ptr(ctx, bias, cols)?;
    let (pp, pr, ph) = (
        bf16_ptr(ctx, projection)?,
        bf16_ptr(ctx, residual)?,
        bf16_ptr(ctx, hidden)?,
    );
    check("bias_residual", unsafe {
        ffi::apxinf_hip_native_bias_residual(pp, pb, pr, ph, long(rows)?, int(cols)?, ctx.stream())
    })
}

/// [`bias_residual`] into `hidden`, then [`rms_norm`] of it into `normalized`.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_rms_norm(
    ctx: &HipContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    hidden: &Tensor,
    normalized: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(projection)?;
    for t in [residual, hidden, normalized] {
        same_shape(projection, t)?;
    }
    require_numel(weight, cols, || format!("[{cols}]"))?;
    let pb = optional_ptr(ctx, bias, cols)?;
    let (pp, pr, pw) = (
        bf16_ptr(ctx, projection)?,
        bf16_ptr(ctx, residual)?,
        bf16_ptr(ctx, weight)?,
    );
    let (ph, pn) = (bf16_ptr(ctx, hidden)?, bf16_ptr(ctx, normalized)?);
    check("bias_residual_rms_norm", unsafe {
        ffi::apxinf_hip_native_bias_residual_rms_norm(
            pp,
            pb,
            pr,
            pw,
            ph,
            pn,
            long(rows)?,
            int(cols)?,
            eps,
            ctx.stream(),
        )
    })
}

/// [`bias_residual`] into `hidden`, then [`layer_norm`] of it into `normalized`.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_norm(
    ctx: &HipContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    hidden: &Tensor,
    normalized: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(projection)?;
    for t in [residual, hidden, normalized] {
        same_shape(projection, t)?;
    }
    require_numel(norm_weight, cols, || format!("[{cols}]"))?;
    require_numel(norm_bias, cols, || format!("[{cols}]"))?;
    let pb = optional_ptr(ctx, bias, cols)?;
    let (pp, pr) = (bf16_ptr(ctx, projection)?, bf16_ptr(ctx, residual)?);
    let (pw, pnb) = (bf16_ptr(ctx, norm_weight)?, bf16_ptr(ctx, norm_bias)?);
    let (ph, pn) = (bf16_ptr(ctx, hidden)?, bf16_ptr(ctx, normalized)?);
    check("bias_residual_layer_norm", unsafe {
        ffi::apxinf_hip_native_bias_residual_layer_norm(
            pp,
            pb,
            pr,
            pw,
            pnb,
            ph,
            pn,
            long(rows)?,
            int(cols)?,
            eps,
            ctx.stream(),
        )
    })
}

/// The action expert's gated residual: `hidden = residual + projection * gate`,
/// the gate being elements `[2 cols, 3 cols)` of `gate_style`, then
/// [`ada_rms_norm`] of `hidden` under `norm_style` into `normalized`.
#[allow(clippy::too_many_arguments)]
pub fn ada_gate_residual_rms_norm(
    ctx: &HipContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    hidden: &Tensor,
    normalized: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(projection)?;
    for t in [residual, hidden, normalized] {
        same_shape(projection, t)?;
    }
    style_covers(gate_style, 3 * cols)?;
    style_covers(norm_style, 2 * cols)?;
    let (pp, pr) = (bf16_ptr(ctx, projection)?, bf16_ptr(ctx, residual)?);
    let (pg, ps) = (bf16_ptr(ctx, gate_style)?, bf16_ptr(ctx, norm_style)?);
    let (ph, pn) = (bf16_ptr(ctx, hidden)?, bf16_ptr(ctx, normalized)?);
    check("ada_gate_residual_rms_norm", unsafe {
        ffi::apxinf_hip_native_ada_gate_residual_rms_norm(
            pp,
            pr,
            pg,
            ps,
            ph,
            pn,
            long(rows)?,
            int(cols)?,
            eps,
            ctx.stream(),
        )
    })
}

/// Modulation vectors carry more segments than any one kernel reads; only the
/// prefix a kernel uses has to be present.
fn style_covers(style: &Tensor, elements: usize) -> Result<()> {
    if style.numel() >= elements {
        Ok(())
    } else {
        Err(apxinf_core::Error::ShapeMismatch {
            expected: format!("at least {elements} elements"),
            got: style.shape().to_string(),
        })
    }
}
