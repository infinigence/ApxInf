use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, RawArgs, Semantic};
use super::launch;
use crate::CudaContext;

/// Inputs for ordinary RMS normalization.
pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, weight, normalized, eps, output_scale: 1.0 }
    }
}

/// Inputs for ordinary layer normalization.
pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, bias: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, weight, bias, normalized, eps, output_scale: 1.0 }
    }
}

/// Inputs for adaptive RMS normalization. `norm_style` is `[2 * cols]`.
pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, norm_style: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, norm_style, normalized, eps, output_scale: 1.0 }
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
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, bias, residual, hidden }
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
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, weight: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, bias, residual, weight, hidden, normalized, eps, output_scale: 1.0 }
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
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, weight: &'a Tensor, norm_bias: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, bias, residual, weight, norm_bias, hidden, normalized, eps, output_scale: 1.0 }
    }
}

/// Inputs for adaptive gate plus residual. `gate_style` is `[3 * cols]`.
pub struct AdaGateResidualArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> AdaGateResidualArgs<'a> {
    pub fn new(input: &'a Tensor, residual: &'a Tensor, gate_style: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, residual, gate_style, hidden }
    }
}

pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, residual: &'a Tensor, norm_style: &'a Tensor, gate_style: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, residual, norm_style, gate_style, hidden, normalized, eps, output_scale: 1.0 }
    }
}

/// BF16-only two-rounding bias-then-residual contract.
pub struct BiasThenResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> BiasThenResidualArgs<'a> {
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, bias, residual, hidden }
    }
}

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

pub fn rms_norm(ctx: &CudaContext, args: RmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::Rms, input: args.input, bias: None, residual: None, weight: Some(args.weight), norm_bias: None, norm_style: None, gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn layer_norm(ctx: &CudaContext, args: LayerNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::Layer, input: args.input, bias: None, residual: None, weight: Some(args.weight), norm_bias: Some(args.bias), norm_style: None, gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn adaptive_rms_norm(ctx: &CudaContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaptiveRms, input: args.input, bias: None, residual: None, weight: None, norm_bias: None, norm_style: Some(args.norm_style), gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn bias_residual(ctx: &CudaContext, args: BiasResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidual, input: args.input, bias: args.bias, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: None, eps: 1e-6, output_scale: 1.0 })
}

pub fn bias_residual_rms_norm(ctx: &CudaContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidualRms, input: args.input, bias: args.bias, residual: Some(args.residual), weight: Some(args.weight), norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn bias_residual_layer_norm(ctx: &CudaContext, args: BiasResidualLayerNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidualLayer, input: args.input, bias: args.bias, residual: Some(args.residual), weight: Some(args.weight), norm_bias: Some(args.norm_bias), norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn ada_gate_residual(ctx: &CudaContext, args: AdaGateResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaGateResidual, input: args.input, bias: None, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: Some(args.gate_style), hidden: Some(args.hidden), normalized: None, eps: 1e-6, output_scale: 1.0 })
}

pub fn ada_gate_residual_rms_norm(ctx: &CudaContext, args: AdaGateResidualRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaGateResidualRms, input: args.input, bias: None, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: Some(args.norm_style), gate_style: Some(args.gate_style), hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, output_scale: args.output_scale })
}

pub fn bias_then_residual(ctx: &CudaContext, args: BiasThenResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasThenResidual, input: args.input, bias: args.bias, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: None, eps: 1e-6, output_scale: 1.0 })
}
