use apxinf_core::{Result, Tensor};

use super::contracts::{invalid, normalize, Fields, RawArgs};
use super::launch;
use crate::ffi::abi::norm as abi;
use crate::CudaContext;

/// Which reduction the row-wise normalization runs over a `[rows, cols]`
/// activation. Both reduce in f32 regardless of the F16/BF16 storage.
pub enum NormKind<'a> {
    /// `x[r,c] / sqrt(mean(x[r,:]^2) + eps)`
    Rms,
    /// `(x[r,c] - mean(x[r,:])) / sqrt(var(x[r,:]) + eps)`, then `+ bias[c]`
    /// after the affine. `bias` is the `[cols]` LayerNorm shift and is
    /// separate from the projection bias a fused add takes.
    Layer { bias: &'a Tensor },
}

/// How the normalized row is rescaled. There is no `None`: [`residual_add`],
/// the one form that applies no affine, takes no `Affine` at all.
pub enum Affine<'a> {
    /// `* weight[c]`, or `* (1 + weight[c])` when `offset`.
    ///
    /// The two settings are **not** one kernel with a flag: `offset` selects
    /// the Qwen3.8 RMSNorm kernel, which has its own block size and reduction
    /// tree, so the results differ by more than the `1 +`. That kernel covers
    /// exactly one pipeline -- [`norm`] with [`NormKind::Rms`], BF16 in and
    /// out, no `post`. Every other combination is refused by name rather than
    /// approximated: those describing a coherent pipeline with no kernel (any
    /// [`fused_add_norm`] form, F16) are `UNSUPPORTED`, while those the stage
    /// rules already exclude ([`NormKind::Layer`], a `post`) are
    /// `INVALID_ARGUMENT`.
    Weight { weight: &'a Tensor, offset: bool },
    /// `* (1 + style[c]) + style[cols + c]`, where `style` is `[2 * cols]`:
    /// scale in `[0, cols)` and shift in `[cols, 2 * cols)`.
    AdaScaleShift { style: &'a Tensor },
}

/// Fused output quantization, applied instead of writing the storage dtype.
/// Must agree with the output tensor's dtype: `Some` requires an E4M3 output
/// and an E4M3 output requires `Some`. The two state one decision, so a
/// disagreement is refused rather than resolved in either direction.
pub enum QuantOut {
    /// E4M3 saturating at +/-448 against a static dequantization `scale`.
    /// F16 input only.
    E4m3 { scale: f32 },
}

/// Rounding order of a bias plus residual combine.
pub enum BiasOrder {
    /// One rounding of `input + bias + residual`.
    SingleRound,
    /// `input + bias` rounds to the storage dtype before the residual is added
    /// and rounded again. BF16 only, and observably different from
    /// [`Self::SingleRound`] at rounding boundaries: it reproduces the legacy
    /// two-kernel contract in one launch.
    RoundAfterBias,
}

/// Pure row-wise normalization: one input, one output, no residual.
pub struct NormArgs<'a> {
    pub input: &'a Tensor,
    pub out: &'a mut Tensor,
    pub kind: NormKind<'a>,
    pub affine: Affine<'a>,
    pub eps: f32,
    pub post: Option<QuantOut>,
}

impl<'a> NormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        out: &'a mut Tensor,
        kind: NormKind<'a>,
        affine: Affine<'a>,
        eps: f32,
    ) -> Self {
        Self {
            input,
            out,
            kind,
            affine,
            eps,
            post: None,
        }
    }

    /// `out[r,c] = input[r,c] * rms(input[r,:])^-1 * (1 + weight[c])`, the
    /// Qwen3.8 RMSNorm. See [`Affine::Weight`] for why this is not the same
    /// kernel as the `offset: false` form.
    pub fn rms_weight_offset(
        input: &'a Tensor,
        weight: &'a Tensor,
        out: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self::new(
            input,
            out,
            NormKind::Rms,
            Affine::Weight {
                weight,
                offset: true,
            },
            eps,
        )
    }
}

/// Residual combine followed by normalization, writing both the combined row
/// (`hidden`) and the normalized row (`normalized`).
///
/// `bias` and `gate` are mutually exclusive: the gate multiplies the
/// projection where the bias would be added. `gate` is `[3 * cols]` and only
/// its third segment, `[2 * cols, 3 * cols)`, is read -- it is not
/// interchangeable with [`Affine::AdaScaleShift`]'s `[2 * cols]` style.
///
/// The gated form reads the gate from the *current* layer's conditioning and
/// normalizes with the *next* layer's: the pi05 action expert folds the next
/// layer's normalization into this layer's residual.
pub struct FusedAddNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub gate: Option<&'a Tensor>,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub kind: NormKind<'a>,
    pub affine: Affine<'a>,
    pub eps: f32,
    pub post: Option<QuantOut>,
}

impl<'a> FusedAddNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        kind: NormKind<'a>,
        affine: Affine<'a>,
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual,
            bias: None,
            gate: None,
            hidden,
            normalized,
            kind,
            affine,
            eps,
            post: None,
        }
    }
}

/// Residual combine with no normalization. `bias` and `gate` are mutually
/// exclusive; see [`FusedAddNormArgs`] for the gate layout.
pub struct ResidualAddArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub gate: Option<&'a Tensor>,
    pub order: BiasOrder,
    pub out: &'a mut Tensor,
}

impl<'a> ResidualAddArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            residual,
            bias,
            gate: None,
            order: BiasOrder::SingleRound,
            out,
        }
    }

    /// Gated combine: `out = residual + input * gate[2 * cols + c]`.
    pub fn gated(
        input: &'a Tensor,
        residual: &'a Tensor,
        gate: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            residual,
            bias: None,
            gate: Some(gate),
            order: BiasOrder::SingleRound,
            out,
        }
    }
}

/// The Spec stage fields and the tensors a `kind`/`affine` pair binds.
struct Stage<'a> {
    norm_kind: u32,
    affine_kind: u32,
    weight: Option<&'a Tensor>,
    norm_bias: Option<&'a Tensor>,
    norm_style: Option<&'a Tensor>,
}

fn stage<'a>(kind: &NormKind<'a>, affine: &Affine<'a>) -> Stage<'a> {
    let (norm_kind, norm_bias) = match kind {
        NormKind::Rms => (abi::NORM_KIND_RMS, None),
        NormKind::Layer { bias } => (abi::NORM_KIND_LAYER, Some(*bias)),
    };
    let (affine_kind, weight, norm_style) = match affine {
        Affine::Weight { weight, offset } => (
            if *offset {
                abi::AFFINE_WEIGHT_OFFSET
            } else {
                abi::AFFINE_WEIGHT
            },
            Some(*weight),
            None,
        ),
        Affine::AdaScaleShift { style } => (abi::AFFINE_ADA_SCALE_SHIFT, None, Some(*style)),
    };
    Stage {
        norm_kind,
        affine_kind,
        weight,
        norm_bias,
        norm_style,
    }
}

fn quant_scale(post: &Option<QuantOut>) -> Option<f32> {
    match post {
        Some(QuantOut::E4m3 { scale }) => Some(*scale),
        None => None,
    }
}

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

// Args -> RawArgs is kept as its own function per entry point so a test can
// assert the Spec fields a given Args produces without launching anything.
// This mapping is the whole content of the field model: a bug here silently
// routes an operator to an arithmetically different kernel, which is exactly
// what the nine fused semantics used to do by falling through.

fn norm_raw_args(args: NormArgs<'_>) -> RawArgs<'_> {
    let stage = stage(&args.kind, &args.affine);
    let post_scale = quant_scale(&args.post);
    RawArgs {
        fields: Fields {
            semantic: abi::SEMANTIC_NORM,
            norm_kind: stage.norm_kind,
            affine_kind: stage.affine_kind,
            has_gate: false,
            bias_order: abi::BIAS_ORDER_SINGLE_ROUND,
        },
        input: args.input,
        bias: None,
        residual: None,
        weight: stage.weight,
        norm_bias: stage.norm_bias,
        norm_style: stage.norm_style,
        gate_style: None,
        hidden: None,
        normalized: Some(args.out),
        eps: args.eps,
        post_scale,
    }
}

fn fused_add_norm_raw_args(args: FusedAddNormArgs<'_>) -> RawArgs<'_> {
    let stage = stage(&args.kind, &args.affine);
    let post_scale = quant_scale(&args.post);
    let has_gate = args.gate.is_some();
    RawArgs {
        fields: Fields {
            semantic: abi::SEMANTIC_FUSED_ADD_NORM,
            norm_kind: stage.norm_kind,
            affine_kind: stage.affine_kind,
            has_gate,
            bias_order: abi::BIAS_ORDER_SINGLE_ROUND,
        },
        input: args.input,
        bias: args.bias,
        residual: Some(args.residual),
        weight: stage.weight,
        norm_bias: stage.norm_bias,
        norm_style: stage.norm_style,
        gate_style: args.gate,
        hidden: Some(args.hidden),
        normalized: Some(args.normalized),
        eps: args.eps,
        post_scale,
    }
}

fn residual_add_raw_args(args: ResidualAddArgs<'_>) -> RawArgs<'_> {
    let bias_order = match args.order {
        BiasOrder::SingleRound => abi::BIAS_ORDER_SINGLE_ROUND,
        BiasOrder::RoundAfterBias => abi::BIAS_ORDER_ROUND_AFTER_BIAS,
    };
    let has_gate = args.gate.is_some();
    RawArgs {
        fields: Fields {
            semantic: abi::SEMANTIC_RESIDUAL_ADD,
            norm_kind: abi::NORM_KIND_NONE,
            affine_kind: abi::AFFINE_NONE,
            has_gate,
            bias_order,
        },
        input: args.input,
        bias: args.bias,
        residual: Some(args.residual),
        weight: None,
        norm_bias: None,
        norm_style: None,
        gate_style: args.gate,
        hidden: Some(args.out),
        normalized: None,
        // The combine forms take no epsilon; the ABI validates it as a
        // positive float whether or not a reduction consumes it.
        eps: 1e-6,
        post_scale: None,
    }
}

/// The `Spec` a `NormArgs` produces, without launching. Test-only hook for the
/// field-mapping table in `l3_behavior`.
#[cfg(test)]
pub(crate) fn norm_spec(ctx: &CudaContext, args: NormArgs<'_>) -> Result<abi::Spec> {
    Ok(normalize(ctx, norm_raw_args(args))?.spec)
}

/// The `Spec` a `FusedAddNormArgs` produces, without launching.
#[cfg(test)]
pub(crate) fn fused_add_norm_spec(
    ctx: &CudaContext,
    args: FusedAddNormArgs<'_>,
) -> Result<abi::Spec> {
    Ok(normalize(ctx, fused_add_norm_raw_args(args))?.spec)
}

/// The `Spec` a `ResidualAddArgs` produces, without launching.
#[cfg(test)]
pub(crate) fn residual_add_spec(
    ctx: &CudaContext,
    args: ResidualAddArgs<'_>,
) -> Result<abi::Spec> {
    Ok(normalize(ctx, residual_add_raw_args(args))?.spec)
}

/// Row-wise normalization of a `[rows, cols]` activation.
pub fn norm(ctx: &CudaContext, args: NormArgs<'_>) -> Result<()> {
    execute(ctx, norm_raw_args(args))
}

/// Residual combine plus normalization in one launch, writing the combined
/// row and the normalized row.
pub fn fused_add_norm(ctx: &CudaContext, args: FusedAddNormArgs<'_>) -> Result<()> {
    if args.bias.is_some() && args.gate.is_some() {
        return Err(invalid(
            "a fused add takes either a bias or a gate, not both",
        ));
    }
    execute(ctx, fused_add_norm_raw_args(args))
}

/// Bias or gate combine plus residual, with no normalization.
pub fn residual_add(ctx: &CudaContext, args: ResidualAddArgs<'_>) -> Result<()> {
    if args.bias.is_some() && args.gate.is_some() {
        return Err(invalid(
            "a residual add takes either a bias or a gate, not both",
        ));
    }
    execute(ctx, residual_add_raw_args(args))
}
