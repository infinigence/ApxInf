//! Row-wise normalization contracts.

use apxinf_core::{Result, Tensor};

use super::require_unit_output_scale;
use crate::kernels::fused as fused_kernels;
use crate::kernels::norm as norm_kernels;
use crate::MusaContext;

/// Inputs for RMS normalization over `input`'s last axis.
pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when the output is quantized. MUSA writes
    /// BF16 only, so anything but 1.0 is refused.
    pub output_scale: f32,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self {
            input,
            weight,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for ordinary layer normalization.
pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        bias: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for adaptive RMS normalization. `norm_style` is the scale-then-shift
/// pair, `[2 * cols]`; a longer vector is accepted because PI0.5's modulation
/// carries a third gate segment this operator does not consume.
pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        norm_style: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            norm_style,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for bias plus residual. Bias is optional by contract.
pub struct BiasResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> BiasResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            hidden,
        }
    }
}

pub struct BiasResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

pub struct BiasResidualLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        norm_bias: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            norm_bias,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for gated residual plus adaptive RMS normalization. `norm_style` is
/// `[2 * cols]`; `gate_style` is `[3 * cols]` and its gate is the third
/// segment.
pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual,
            norm_style,
            gate_style,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

pub fn rms_norm(ctx: &MusaContext, args: RmsNormArgs<'_>) -> Result<()> {
    require_unit_output_scale("rms_norm", args.output_scale)?;
    norm_kernels::rms_bf16_into(ctx, args.input, args.weight, args.normalized, args.eps)
}

pub fn layer_norm(ctx: &MusaContext, args: LayerNormArgs<'_>) -> Result<()> {
    require_unit_output_scale("layer_norm", args.output_scale)?;
    norm_kernels::layer_bf16_into(ctx, args.input, args.weight, args.bias, args.normalized, args.eps)
}

pub fn adaptive_rms_norm(ctx: &MusaContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    require_unit_output_scale("adaptive_rms_norm", args.output_scale)?;
    norm_kernels::adaptive_rms_bf16_into(ctx, args.input, args.norm_style, args.normalized, args.eps)
}

pub fn bias_residual(ctx: &MusaContext, args: BiasResidualArgs<'_>) -> Result<()> {
    fused_kernels::bias_residual_bf16_into(ctx, args.input, args.bias, args.residual, args.hidden)
}

pub fn bias_residual_rms_norm(ctx: &MusaContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    require_unit_output_scale("bias_residual_rms_norm", args.output_scale)?;
    fused_kernels::bias_residual_rms_bf16_into(
        ctx,
        args.input,
        args.bias,
        args.residual,
        args.weight,
        args.hidden,
        args.normalized,
        args.eps,
    )
}

pub fn bias_residual_layer_norm(
    ctx: &MusaContext,
    args: BiasResidualLayerNormArgs<'_>,
) -> Result<()> {
    require_unit_output_scale("bias_residual_layer_norm", args.output_scale)?;
    fused_kernels::bias_residual_layer_bf16_into(
        ctx,
        args.input,
        args.bias,
        args.residual,
        args.weight,
        args.norm_bias,
        args.hidden,
        args.normalized,
        args.eps,
    )
}

pub fn ada_gate_residual_rms_norm(
    ctx: &MusaContext,
    args: AdaGateResidualRmsNormArgs<'_>,
) -> Result<()> {
    require_unit_output_scale("ada_gate_residual_rms_norm", args.output_scale)?;
    fused_kernels::adaptive_gate_residual_rms_bf16_into(
        ctx,
        args.input,
        args.residual,
        args.gate_style,
        args.norm_style,
        args.hidden,
        args.normalized,
        args.eps,
    )
}
