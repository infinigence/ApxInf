use apxinf_core::{Result, Tensor};

use super::contracts::{
    normalize, AffineKind, BiasOrder, GateKind, NormNumerics, Operation, Quantization, RawArgs,
    Statistic,
};
use super::launch;
use crate::CudaContext;

/// Statistic computed by the normalization core.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormKind {
    Rms,
    Layer,
}

/// Affine transform applied after the normalization statistic.
#[derive(Clone, Copy)]
pub enum Affine<'a> {
    /// Learned weight and optional LayerNorm bias. `offset` means the stored
    /// checkpoint weight is zero-centered and the effective multiplier is
    /// `1 + weight`.
    Weight {
        weight: &'a Tensor,
        bias: Option<&'a Tensor>,
        offset: bool,
    },
    /// Per-call `[scale; shift]` conditioning vector.
    Adaptive { style: &'a Tensor },
    /// Learned RMS weight followed by independent adaptive scale and shift:
    /// `bf16(bf16(rms(x) * weight) * bf16(1 + scale) + shift)`.
    WeightedAdaptive {
        weight: &'a Tensor,
        scale: &'a Tensor,
        shift: &'a Tensor,
    },
}

/// Optional gate applied by the residual prologue.
#[derive(Clone, Copy)]
pub enum ResidualGate<'a> {
    /// Packed modulation `[scale; shift; gate]`; the third segment is the
    /// direct multiplier used by the existing adaptive Norm contract.
    PackedStyle(&'a Tensor),
    /// Per-column gate whose effective multiplier is BF16-rounded `1 + gate`.
    UnitOffset(&'a Tensor),
}

/// Quantization epilogue for a normalized output.
#[derive(Clone, Copy, Debug)]
pub enum QuantOut<'a> {
    /// Store a calibrated E4M3 copy of the normalized result. `NormArgs::output`
    /// or `FusedAddNormArgs::normalized` may independently retain the native
    /// normalized result; this binding always owns the quantized epilogue.
    StaticE4m3 { output: &'a Tensor, scale: f32 },
    /// Store one dynamically scaled E4M3 row per input row. `values` may pad
    /// the column dimension; padding values are zero. `scales` is F32 `[rows]`.
    RowwiseE4m3 {
        values: &'a Tensor,
        scales: &'a Tensor,
    },
    /// Store one dynamically scaled signed-INT8 row per input row. The native
    /// BF16 normalized output is retained independently by `output`.
    RowwiseI8 {
        values: &'a Tensor,
        scales: &'a Tensor,
    },
}

/// Plain normalization with caller-owned native and/or quantized outputs.
pub struct NormArgs<'a> {
    pub input: &'a Tensor,
    pub output: Option<&'a Tensor>,
    pub kind: NormKind,
    pub affine: Affine<'a>,
    pub quantized: Option<QuantOut<'a>>,
    pub numerics: NormNumerics,
    pub eps: f32,
}

impl<'a> NormArgs<'a> {
    pub fn rms(input: &'a Tensor, weight: &'a Tensor, output: &'a Tensor, eps: f32) -> Self {
        Self {
            input,
            output: Some(output),
            kind: NormKind::Rms,
            affine: Affine::Weight {
                weight,
                bias: None,
                offset: false,
            },
            quantized: None,
            numerics: NormNumerics::Flexible,
            eps,
        }
    }

    pub fn layer(
        input: &'a Tensor,
        weight: &'a Tensor,
        bias: &'a Tensor,
        output: &'a Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            output: Some(output),
            kind: NormKind::Layer,
            affine: Affine::Weight {
                weight,
                bias: Some(bias),
                offset: false,
            },
            quantized: None,
            numerics: NormNumerics::Flexible,
            eps,
        }
    }

    pub fn with_weight_offset(mut self) -> Self {
        match &mut self.affine {
            Affine::Weight { offset, .. } => *offset = true,
            Affine::Adaptive { .. } | Affine::WeightedAdaptive { .. } => {}
        }
        self
    }
}

/// Residual prologue followed by normalization. The pre-normalization hidden
/// stream and normalized result are both caller-owned.
pub struct FusedAddNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub gate: Option<ResidualGate<'a>>,
    pub hidden: &'a Tensor,
    pub normalized: Option<&'a Tensor>,
    pub kind: NormKind,
    pub affine: Affine<'a>,
    pub quantized: Option<QuantOut<'a>>,
    pub numerics: NormNumerics,
    pub bias_order: BiasOrder,
    pub eps: f32,
}

/// Residual prologue without a normalization core.
pub struct ResidualAddArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub gate: Option<ResidualGate<'a>>,
    pub hidden: &'a Tensor,
    pub bias_order: BiasOrder,
}

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

fn statistic(kind: NormKind) -> Statistic {
    match kind {
        NormKind::Rms => Statistic::Rms,
        NormKind::Layer => Statistic::Layer,
    }
}

fn quantization(value: Option<QuantOut<'_>>) -> Quantization {
    match value {
        None => Quantization::None,
        Some(QuantOut::StaticE4m3 { .. }) => Quantization::StaticE4m3,
        Some(QuantOut::RowwiseE4m3 { .. }) => Quantization::RowwiseE4m3,
        Some(QuantOut::RowwiseI8 { .. }) => Quantization::RowwiseI8,
    }
}

fn quantized_output(value: Option<QuantOut<'_>>) -> Option<&Tensor> {
    match value {
        None => None,
        Some(QuantOut::StaticE4m3 { output, .. }) => Some(output),
        Some(QuantOut::RowwiseE4m3 { values, .. }) | Some(QuantOut::RowwiseI8 { values, .. }) => {
            Some(values)
        }
    }
}

fn quantized_scales(value: Option<QuantOut<'_>>) -> Option<&Tensor> {
    match value {
        Some(QuantOut::RowwiseE4m3 { scales, .. }) | Some(QuantOut::RowwiseI8 { scales, .. }) => {
            Some(scales)
        }
        None | Some(QuantOut::StaticE4m3 { .. }) => None,
    }
}

fn output_scale(value: Option<QuantOut<'_>>) -> f32 {
    match value {
        Some(QuantOut::StaticE4m3 { scale, .. }) => scale,
        None | Some(QuantOut::RowwiseE4m3 { .. }) | Some(QuantOut::RowwiseI8 { .. }) => 1.0,
    }
}

/// Execute plain RMSNorm/LayerNorm through the normalization family.
pub fn norm(ctx: &CudaContext, args: NormArgs<'_>) -> Result<()> {
    let output_scale = output_scale(args.quantized);
    match args.affine {
        Affine::Weight {
            weight,
            bias,
            offset: true,
        } => {
            if args.kind != NormKind::Rms || bias.is_some() || args.quantized.is_some() {
                return Err(super::contracts::invalid(
                    "weight-offset affine currently supports unquantized RMSNorm only",
                ));
            }
            let output = args.output.ok_or_else(|| {
                super::contracts::invalid("weight-offset RMSNorm requires a native output")
            })?;
            if args.numerics == NormNumerics::StableDecode {
                return Err(super::contracts::invalid(
                    "weight-offset RMSNorm does not support StableDecode numerics",
                ));
            }
            execute(
                ctx,
                RawArgs {
                    operation: Operation::Norm,
                    statistic: Statistic::Rms,
                    affine: AffineKind::Weight,
                    gate: GateKind::None,
                    quantization: Quantization::None,
                    numerics: args.numerics,
                    bias_order: BiasOrder::SingleRound,
                    weight_offset: true,
                    input: args.input,
                    bias: None,
                    residual: None,
                    weight: Some(weight),
                    norm_bias: None,
                    norm_style: None,
                    gate_style: None,
                    hidden: None,
                    normalized: Some(output),
                    quantized_output: None,
                    quantized_scales: None,
                    eps: args.eps,
                    output_scale: 1.0,
                },
            )
        }
        Affine::Weight {
            weight,
            bias,
            offset: false,
        } => {
            match args.kind {
                NormKind::Rms if bias.is_none() => {}
                NormKind::Layer if bias.is_some() => {}
                NormKind::Rms => {
                    return Err(super::contracts::invalid(
                        "RMSNorm does not take an affine bias",
                    ))
                }
                NormKind::Layer => {
                    return Err(super::contracts::invalid(
                        "LayerNorm requires an affine bias",
                    ))
                }
            }
            execute(
                ctx,
                RawArgs {
                    operation: Operation::Norm,
                    statistic: statistic(args.kind),
                    affine: AffineKind::Weight,
                    gate: GateKind::None,
                    quantization: quantization(args.quantized),
                    numerics: args.numerics,
                    bias_order: BiasOrder::SingleRound,
                    weight_offset: false,
                    input: args.input,
                    bias: None,
                    residual: None,
                    weight: Some(weight),
                    norm_bias: bias,
                    norm_style: None,
                    gate_style: None,
                    hidden: None,
                    normalized: args.output,
                    quantized_output: quantized_output(args.quantized),
                    quantized_scales: quantized_scales(args.quantized),
                    eps: args.eps,
                    output_scale,
                },
            )
        }
        Affine::Adaptive { style } => execute(
            ctx,
            RawArgs {
                operation: Operation::Norm,
                statistic: statistic(args.kind),
                affine: AffineKind::Adaptive,
                gate: GateKind::None,
                quantization: quantization(args.quantized),
                numerics: args.numerics,
                bias_order: BiasOrder::SingleRound,
                weight_offset: false,
                input: args.input,
                bias: None,
                residual: None,
                weight: None,
                norm_bias: None,
                norm_style: Some(style),
                gate_style: None,
                hidden: None,
                normalized: args.output,
                quantized_output: quantized_output(args.quantized),
                quantized_scales: quantized_scales(args.quantized),
                eps: args.eps,
                output_scale,
            },
        ),
        Affine::WeightedAdaptive {
            weight,
            scale,
            shift,
        } => {
            if args.kind != NormKind::Rms
                || args.quantized.is_some()
                || args.numerics != NormNumerics::Flexible
            {
                return Err(super::contracts::invalid(
                    "weighted adaptive affine supports unquantized flexible RMSNorm only",
                ));
            }
            let output = args.output.ok_or_else(|| {
                super::contracts::invalid("weighted adaptive RMSNorm requires a native output")
            })?;
            super::candidates::weighted_adaptive(
                ctx, args.input, weight, scale, shift, output, args.eps,
            )
        }
    }
}

/// Execute a residual prologue followed by RMSNorm/LayerNorm.
pub fn fused_add_norm(ctx: &CudaContext, args: FusedAddNormArgs<'_>) -> Result<()> {
    if let (
        NormKind::Rms,
        Affine::Weight {
            weight,
            bias: None,
            offset: true,
        },
        None,
    ) = (args.kind, args.affine, args.gate)
    {
        if args.bias.is_some()
            || args.quantized.is_some()
            || args.numerics != NormNumerics::StablePlanning
            || args.bias_order != BiasOrder::SingleRound
        {
            return Err(super::contracts::invalid(
                "weight-offset fused RMSNorm requires no bias/quantization, SingleRound, and StablePlanning numerics",
            ));
        }
        let normalized = args.normalized.ok_or_else(|| {
            super::contracts::invalid("weight-offset fused RMSNorm requires a native output")
        })?;
        return execute(
            ctx,
            RawArgs {
                operation: Operation::FusedAddNorm,
                statistic: Statistic::Rms,
                affine: AffineKind::Weight,
                gate: GateKind::None,
                quantization: Quantization::None,
                numerics: NormNumerics::StablePlanning,
                bias_order: BiasOrder::SingleRound,
                weight_offset: true,
                input: args.input,
                bias: None,
                residual: Some(args.residual),
                weight: Some(weight),
                norm_bias: None,
                norm_style: None,
                gate_style: None,
                hidden: Some(args.hidden),
                normalized: Some(normalized),
                quantized_output: None,
                quantized_scales: None,
                eps: args.eps,
                output_scale: 1.0,
            },
        );
    }
    if let (
        NormKind::Rms,
        Affine::WeightedAdaptive {
            weight,
            scale,
            shift,
        },
        Some(ResidualGate::UnitOffset(gate)),
    ) = (args.kind, args.affine, args.gate)
    {
        if args.bias.is_some()
            || args.quantized.is_some()
            || args.numerics != NormNumerics::Flexible
            || args.bias_order != BiasOrder::SingleRound
        {
            return Err(super::contracts::invalid(
                "unit-offset weighted adaptive RMSNorm requires no bias/quantization and SingleRound flexible numerics",
            ));
        }
        let normalized = args.normalized.ok_or_else(|| {
            super::contracts::invalid(
                "unit-offset weighted adaptive RMSNorm requires a native output",
            )
        })?;
        return super::candidates::unit_offset_residual_norm(
            ctx,
            args.input,
            args.residual,
            gate,
            weight,
            scale,
            shift,
            args.hidden,
            normalized,
            args.eps,
        );
    }
    let output_scale = output_scale(args.quantized);
    let (affine, gate, weight, norm_bias, norm_style, gate_style) =
        match (args.kind, args.affine, args.gate) {
            (
                NormKind::Rms,
                Affine::Weight {
                    weight,
                    bias: None,
                    offset: false,
                },
                None,
            ) => (
                AffineKind::Weight,
                GateKind::None,
                Some(weight),
                None,
                None,
                None,
            ),
            (
                NormKind::Layer,
                Affine::Weight {
                    weight,
                    bias: Some(norm_bias),
                    offset: false,
                },
                None,
            ) => (
                AffineKind::Weight,
                GateKind::None,
                Some(weight),
                Some(norm_bias),
                None,
                None,
            ),
            (NormKind::Layer, Affine::Adaptive { style }, None) => (
                AffineKind::Adaptive,
                GateKind::None,
                None,
                None,
                Some(style),
                None,
            ),
            (NormKind::Rms, Affine::Adaptive { style }, Some(ResidualGate::PackedStyle(gate))) => (
                AffineKind::Adaptive,
                GateKind::Style,
                None,
                None,
                Some(style),
                Some(gate),
            ),
            _ => {
                return Err(super::contracts::invalid(
                    "unsupported fused-add NormKind/Affine/gate combination",
                ))
            }
        };
    execute(
        ctx,
        RawArgs {
            operation: Operation::FusedAddNorm,
            statistic: statistic(args.kind),
            affine,
            gate,
            quantization: quantization(args.quantized),
            numerics: args.numerics,
            bias_order: args.bias_order,
            weight_offset: false,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight,
            norm_bias,
            norm_style,
            gate_style,
            hidden: Some(args.hidden),
            normalized: args.normalized,
            quantized_output: quantized_output(args.quantized),
            quantized_scales: quantized_scales(args.quantized),
            eps: args.eps,
            output_scale,
        },
    )
}
/// Execute only the residual prologue.
pub fn residual_add(ctx: &CudaContext, args: ResidualAddArgs<'_>) -> Result<()> {
    if let Some(ResidualGate::UnitOffset(gate)) = args.gate {
        if args.bias.is_some() || args.bias_order != BiasOrder::SingleRound {
            return Err(super::contracts::invalid(
                "gated residual add does not take bias or BiasThenResidual",
            ));
        }
        return super::candidates::unit_offset_residual(
            ctx,
            args.input,
            args.residual,
            gate,
            args.hidden,
        );
    }
    let (gate, gate_style) = match args.gate {
        Some(ResidualGate::PackedStyle(gate)) => {
            if args.bias.is_some() || args.bias_order != BiasOrder::SingleRound {
                return Err(super::contracts::invalid(
                    "gated residual add does not take bias or BiasThenResidual",
                ));
            }
            (GateKind::Style, Some(gate))
        }
        None => (GateKind::None, None),
        Some(ResidualGate::UnitOffset(_)) => unreachable!("handled above"),
    };
    execute(
        ctx,
        RawArgs {
            operation: Operation::ResidualAdd,
            statistic: Statistic::None,
            affine: AffineKind::None,
            gate,
            quantization: Quantization::None,
            numerics: NormNumerics::Flexible,
            bias_order: args.bias_order,
            weight_offset: false,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: None,
            gate_style,
            hidden: Some(args.hidden),
            normalized: None,
            quantized_output: None,
            quantized_scales: None,
            eps: 1e-6,
            output_scale: 1.0,
        },
    )
}
