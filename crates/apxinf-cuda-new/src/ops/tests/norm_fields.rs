//! The norm family's field model, asserted as a table.
//!
//! PR #122 replaced nine fused semantic codes with six orthogonal Spec fields
//! and argued in its commit message that the mapping is a bijection onto the
//! old one. That argument lived only in prose: nothing in the crate failed if a
//! field combination was routed to an arithmetically different kernel, which is
//! precisely what the nine semantics used to do by falling through to a
//! neighbour.
//!
//! These tests assert the `Spec` each public `Args` produces, without launching
//! anything. They are contract tests, not numerics tests — a wrong field tuple
//! here means the native adapter picks a different kernel, and that is caught
//! by comparing tuples rather than outputs.
//!
//! The reference column is the table from `628fc34`'s commit message:
//!
//! ```text
//!   semantic            -> fields (sem/norm/affine/gate/order)   -> kernel
//!   Rms                 -> norm/rms/weight/0/single              -> rms_norm
//!   Layer               -> norm/layer/weight/0/single            -> layer_norm
//!   AdaptiveRms         -> norm/rms/ada/0/single                 -> ada_rms_norm
//!   BiasResidual        -> residual_add/none/none/0/single       -> bias_residual
//!   BiasThenResidual    -> residual_add/none/none/0/after_bias   -> bias_then_residual
//!   AdaGateResidual     -> residual_add/none/none/1/single       -> ada_gate_residual
//!   BiasResidualRms     -> fused_add_norm/rms/weight/0/single    -> bias_residual_rms_norm
//!   BiasResidualLayer   -> fused_add_norm/layer/weight/0/single  -> bias_residual_layer_norm
//!   AdaGateResidualRms  -> fused_add_norm/rms/ada/1/single       -> ada_gate_residual_rms_norm
//! ```

use apxinf_core::DType;

use super::framework::zeros_tensor;
use super::*;
use crate::ffi::abi::norm as abi;
use crate::CudaContext;

const ROWS: usize = 4;
const COLS: usize = 8;

/// The six stage fields plus the two dtype fields, as one comparable tuple.
#[derive(Debug, PartialEq, Eq)]
struct FieldTuple {
    semantic: u32,
    norm_kind: u32,
    affine_kind: u32,
    has_gate: u32,
    bias_order: u32,
    quant_kind: u32,
    has_bias: u32,
}

fn fields_of(spec: &abi::Spec) -> FieldTuple {
    FieldTuple {
        semantic: spec.semantic,
        norm_kind: spec.norm_kind,
        affine_kind: spec.affine_kind,
        has_gate: spec.has_gate,
        bias_order: spec.bias_order,
        quant_kind: spec.quant_kind,
        has_bias: spec.has_bias,
    }
}

fn expect(
    semantic: u32,
    norm_kind: u32,
    affine_kind: u32,
    has_gate: u32,
    bias_order: u32,
    quant_kind: u32,
    has_bias: u32,
) -> FieldTuple {
    FieldTuple {
        semantic,
        norm_kind,
        affine_kind,
        has_gate,
        bias_order,
        quant_kind,
        has_bias,
    }
}

fn bf16(shape: Vec<usize>) -> apxinf_core::Tensor {
    zeros_tensor(0, shape, DType::BF16)
}

fn f16(shape: Vec<usize>) -> apxinf_core::Tensor {
    zeros_tensor(0, shape, DType::F16)
}

/// The three pure-normalization forms: old `Rms`, `Layer`, `AdaptiveRms`.
#[test]
fn norm_maps_the_three_pure_normalization_semantics() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let weight = bf16(vec![COLS]);
    let norm_bias = bf16(vec![COLS]);
    let style = bf16(vec![2 * COLS]);

    // Rms -> norm/rms/weight/0/single
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_NORM,
            abi::NORM_KIND_RMS,
            abi::AFFINE_WEIGHT,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            0,
        ),
        "Rms"
    );
    assert_eq!(spec.version, abi::SPEC_VERSION);

    // Layer -> norm/layer/weight/0/single
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Layer { bias: &norm_bias },
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_NORM,
            abi::NORM_KIND_LAYER,
            abi::AFFINE_WEIGHT,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            0,
        ),
        "Layer"
    );

    // AdaptiveRms -> norm/rms/ada/0/single
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::AdaScaleShift { style: &style },
            1e-6,
        ),
    )
    .unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_NORM,
            abi::NORM_KIND_RMS,
            abi::AFFINE_ADA_SCALE_SHIFT,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            0,
        ),
        "AdaptiveRms"
    );
}

/// The three combine-only forms: old `BiasResidual`, `BiasThenResidual`,
/// `AdaGateResidual`. The first two differ only in `bias_order`, which is an
/// intermediate BF16 rounding and therefore part of the contract, not an
/// implementation detail.
#[test]
fn residual_add_maps_the_three_combine_only_semantics() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let residual = bf16(vec![ROWS, COLS]);
    let bias = bf16(vec![COLS]);
    let gate = bf16(vec![3 * COLS]);

    // BiasResidual -> residual_add/none/none/0/single, with a bias
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = residual_add_spec(
        &ctx,
        ResidualAddArgs::new(&input, Some(&bias), &residual, &mut out),
    )
    .unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_RESIDUAL_ADD,
            abi::NORM_KIND_NONE,
            abi::AFFINE_NONE,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            1,
        ),
        "BiasResidual"
    );

    // BiasThenResidual -> residual_add/none/none/0/after_bias
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = ResidualAddArgs::new(&input, Some(&bias), &residual, &mut out);
    args.order = BiasOrder::RoundAfterBias;
    let spec = residual_add_spec(&ctx, args).unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_RESIDUAL_ADD,
            abi::NORM_KIND_NONE,
            abi::AFFINE_NONE,
            0,
            abi::BIAS_ORDER_ROUND_AFTER_BIAS,
            abi::QUANT_NONE,
            1,
        ),
        "BiasThenResidual"
    );

    // AdaGateResidual -> residual_add/none/none/1/single
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = residual_add_spec(
        &ctx,
        ResidualAddArgs::gated(&input, &residual, &gate, &mut out),
    )
    .unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_RESIDUAL_ADD,
            abi::NORM_KIND_NONE,
            abi::AFFINE_NONE,
            1,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            0,
        ),
        "AdaGateResidual"
    );
}

/// The three fused forms: old `BiasResidualRms`, `BiasResidualLayer`,
/// `AdaGateResidualRms`.
#[test]
fn fused_add_norm_maps_the_three_fused_semantics() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let residual = bf16(vec![ROWS, COLS]);
    let bias = bf16(vec![COLS]);
    let weight = bf16(vec![COLS]);
    let norm_bias = bf16(vec![COLS]);
    let style = bf16(vec![2 * COLS]);
    let gate = bf16(vec![3 * COLS]);

    // BiasResidualRms -> fused_add_norm/rms/weight/0/single
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.bias = Some(&bias);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_FUSED_ADD_NORM,
            abi::NORM_KIND_RMS,
            abi::AFFINE_WEIGHT,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            1,
        ),
        "BiasResidualRms"
    );

    // BiasResidualLayer -> fused_add_norm/layer/weight/0/single
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Layer { bias: &norm_bias },
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.bias = Some(&bias);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_FUSED_ADD_NORM,
            abi::NORM_KIND_LAYER,
            abi::AFFINE_WEIGHT,
            0,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            1,
        ),
        "BiasResidualLayer"
    );

    // AdaGateResidualRms -> fused_add_norm/rms/ada/1/single
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Rms,
        Affine::AdaScaleShift { style: &style },
        1e-6,
    );
    args.gate = Some(&gate);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    assert_eq!(
        fields_of(&spec),
        expect(
            abi::SEMANTIC_FUSED_ADD_NORM,
            abi::NORM_KIND_RMS,
            abi::AFFINE_ADA_SCALE_SHIFT,
            1,
            abi::BIAS_ORDER_SINGLE_ROUND,
            abi::QUANT_NONE,
            0,
        ),
        "AdaGateResidualRms"
    );
}

/// The nine field tuples must be pairwise distinct. If two collided, the
/// mapping would not be a bijection onto the old nine semantics and one of them
/// would be unreachable.
///
/// The tuples are collected from the production mapping, not written down here.
/// An earlier version of this test asserted that nine hard-coded tuples differ
/// from each other, which is a property of the literals in the test file: it
/// passed under every mutation of `stage()` and so tested nothing.
#[test]
fn the_nine_field_tuples_are_pairwise_distinct() {
    use std::collections::BTreeMap;

    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let residual = bf16(vec![ROWS, COLS]);
    let bias = bf16(vec![COLS]);
    let weight = bf16(vec![COLS]);
    let norm_bias = bf16(vec![COLS]);
    let style = bf16(vec![2 * COLS]);
    let gate = bf16(vec![3 * COLS]);

    // Collect what the production mapping actually produces for each of the
    // nine shapes the old semantics covered.
    let mut seen: BTreeMap<(u32, u32, u32, u32, u32), &str> = BTreeMap::new();
    let mut record = |spec: &abi::Spec, label: &'static str| {
        let key = (
            spec.semantic,
            spec.norm_kind,
            spec.affine_kind,
            spec.has_gate,
            spec.bias_order,
        );
        if let Some(previous) = seen.insert(key, label) {
            panic!("{label} and {previous} produce the same field tuple {key:?}");
        }
    };

    // norm: Rms / Layer / AdaptiveRms
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    record(&spec, "Rms");

    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Layer { bias: &norm_bias },
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    record(&spec, "Layer");

    let mut out = bf16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::AdaScaleShift { style: &style },
            1e-6,
        ),
    )
    .unwrap();
    record(&spec, "AdaptiveRms");

    // residual_add: BiasResidual / BiasThenResidual / AdaGateResidual
    let mut out = bf16(vec![ROWS, COLS]);
    let spec = residual_add_spec(
        &ctx,
        ResidualAddArgs::new(&input, Some(&bias), &residual, &mut out),
    )
    .unwrap();
    record(&spec, "BiasResidual");

    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = ResidualAddArgs::new(&input, Some(&bias), &residual, &mut out);
    args.order = BiasOrder::RoundAfterBias;
    let spec = residual_add_spec(&ctx, args).unwrap();
    record(&spec, "BiasThenResidual");

    let mut out = bf16(vec![ROWS, COLS]);
    let spec = residual_add_spec(
        &ctx,
        ResidualAddArgs::gated(&input, &residual, &gate, &mut out),
    )
    .unwrap();
    record(&spec, "AdaGateResidual");

    // fused_add_norm: BiasResidualRms / BiasResidualLayer / AdaGateResidualRms
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.bias = Some(&bias);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    record(&spec, "BiasResidualRms");

    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Layer { bias: &norm_bias },
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.bias = Some(&bias);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    record(&spec, "BiasResidualLayer");

    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Rms,
        Affine::AdaScaleShift { style: &style },
        1e-6,
    );
    args.gate = Some(&gate);
    let spec = fused_add_norm_spec(&ctx, args).unwrap();
    record(&spec, "AdaGateResidualRms");

    assert_eq!(
        seen.len(),
        9,
        "the nine old semantics must map onto nine distinct field tuples, got {seen:?}"
    );
}

/// `Affine::Weight { offset: true }` is the Qwen3.8 `(1 + weight)` RMSNorm. It
/// has its own kernel with its own reduction order, so it must reach
/// `AFFINE_WEIGHT_OFFSET` and never be flattened into `AFFINE_WEIGHT`.
#[test]
fn weight_offset_is_a_distinct_affine_not_a_flag_on_weight() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let weight = bf16(vec![COLS]);

    let mut out = bf16(vec![ROWS, COLS]);
    let offset = norm_spec(
        &ctx,
        NormArgs::rms_weight_offset(&input, &weight, &mut out, 1e-6),
    )
    .unwrap();
    assert_eq!(offset.affine_kind, abi::AFFINE_WEIGHT_OFFSET);

    let mut out = bf16(vec![ROWS, COLS]);
    let plain = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    assert_eq!(plain.affine_kind, abi::AFFINE_WEIGHT);
    assert_ne!(
        offset.affine_kind, plain.affine_kind,
        "the two offset settings must not collapse to one affine: their kernels \
         reduce in different orders, so the results differ by more than the `1 +`"
    );
}

/// A fused E4M3 output sets `quant_kind`, and only then. The field is derived
/// from the output tensor's dtype, so a mismatch with `post` must be rejected
/// rather than silently quantizing against a unit scale.
#[test]
fn quant_kind_follows_the_output_dtype() {
    let ctx = CudaContext::new(0).unwrap();
    let input = f16(vec![ROWS, COLS]);
    let weight = f16(vec![COLS]);

    // F16 in, E4M3 out, with a scale -> QUANT_E4M3
    let mut out = zeros_tensor(0, vec![ROWS, COLS], DType::F8E4M3);
    let mut args = NormArgs::new(
        &input,
        &mut out,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.post = Some(QuantOut::E4m3 { scale: 0.5 });
    let spec = norm_spec(&ctx, args).unwrap();
    assert_eq!(spec.quant_kind, abi::QUANT_E4M3);
    assert_eq!(spec.output_scale_is_unit, 0);

    // The same pipeline without `post` keeps the storage dtype -> QUANT_NONE
    let mut out = f16(vec![ROWS, COLS]);
    let spec = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap();
    assert_eq!(spec.quant_kind, abi::QUANT_NONE);
    assert_eq!(spec.output_scale_is_unit, 1);

    // A scale with a non-quantized output is a caller error, not a silent unit
    // scale: the two would disagree about what the kernel should do.
    let mut out = f16(vec![ROWS, COLS]);
    let mut args = NormArgs::new(
        &input,
        &mut out,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.post = Some(QuantOut::E4m3 { scale: 0.5 });
    let error = norm_spec(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("must preserve dtype") || error.contains("unit output_scale"),
        "{error}"
    );

    // The same, at scale 1.0. This is the case that used to slip through: the
    // scale was the only thing carried forward from `post`, so a unit scale
    // made "quantize to E4M3" indistinguishable from "do not quantize", and
    // the request was dropped in favour of a plain F16 norm.
    let mut out = f16(vec![ROWS, COLS]);
    let mut args = NormArgs::new(
        &input,
        &mut out,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.post = Some(QuantOut::E4m3 { scale: 1.0 });
    let error = norm_spec(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("must preserve dtype") || error.contains("unit output_scale"),
        "{error}"
    );

    // And the converse: an E4M3 output with no `post` has no scale to quantize
    // against, so it is refused rather than defaulted to 1.0.
    let mut out = zeros_tensor(0, vec![ROWS, COLS], DType::F8E4M3);
    let error = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            1e-6,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("requires a post quantization scale"), "{error}");
}

/// Combinations the Rust contract must refuse before they reach the adapter.
/// Each would otherwise describe a pipeline with no kernel.
#[test]
fn the_contract_refuses_incoherent_stage_combinations() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let residual = bf16(vec![ROWS, COLS]);
    let bias = bf16(vec![COLS]);
    let weight = bf16(vec![COLS]);
    let gate = bf16(vec![3 * COLS]);

    // A gated fused add takes no bias: the gate replaces it in the combine.
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let mut args = FusedAddNormArgs::new(
        &input,
        &residual,
        &mut hidden,
        &mut normalized,
        NormKind::Rms,
        Affine::Weight {
            weight: &weight,
            offset: false,
        },
        1e-6,
    );
    args.bias = Some(&bias);
    args.gate = Some(&gate);
    let error = fused_add_norm(&ctx, args).unwrap_err().to_string();
    assert!(error.contains("either a bias or a gate"), "{error}");

    // Same rule for the combine-only form.
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = ResidualAddArgs::new(&input, Some(&bias), &residual, &mut out);
    args.gate = Some(&gate);
    let error = residual_add(&ctx, args).unwrap_err().to_string();
    assert!(error.contains("either a bias or a gate"), "{error}");

    // `RoundAfterBias` is BF16-only: its intermediate rounding is the contract.
    let input_f16 = f16(vec![ROWS, COLS]);
    let residual_f16 = f16(vec![ROWS, COLS]);
    let bias_f16 = f16(vec![COLS]);
    let mut out = f16(vec![ROWS, COLS]);
    let mut args =
        ResidualAddArgs::new(&input_f16, Some(&bias_f16), &residual_f16, &mut out);
    args.order = BiasOrder::RoundAfterBias;
    let error = residual_add_spec(&ctx, args).unwrap_err().to_string();
    assert!(error.contains("BF16-only"), "{error}");

    // eps must be finite and positive whether or not a reduction consumes it.
    let mut out = bf16(vec![ROWS, COLS]);
    let error = norm_spec(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: false,
            },
            0.0,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("eps must be finite and positive"), "{error}");
}

/// The adapter must refuse `weight_offset` outside its single supported
/// combination rather than falling back to the plain-weight kernel. This one
/// launches, because the refusal happens in C++.
#[test]
fn weight_offset_outside_its_one_combination_is_refused() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let residual = bf16(vec![ROWS, COLS]);
    let weight = bf16(vec![COLS]);

    // fused_add_norm + weight_offset: the kernel has no residual combine.
    let mut hidden = bf16(vec![ROWS, COLS]);
    let mut normalized = bf16(vec![ROWS, COLS]);
    let error = fused_add_norm(
        &ctx,
        FusedAddNormArgs::new(
            &input,
            &residual,
            &mut hidden,
            &mut normalized,
            NormKind::Rms,
            Affine::Weight {
                weight: &weight,
                offset: true,
            },
            1e-6,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.to_lowercase().contains("unsupported") || error.contains("weight_offset"),
        "fused_add_norm + weight_offset must be refused, got: {error}"
    );

    // LayerNorm + weight_offset: the Spec coherence rule requires a plain
    // weight under a LayerNorm, so this is INVALID_ARGUMENT, not UNSUPPORTED.
    let norm_bias = bf16(vec![COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let error = norm(
        &ctx,
        NormArgs::new(
            &input,
            &mut out,
            NormKind::Layer { bias: &norm_bias },
            Affine::Weight {
                weight: &weight,
                offset: true,
            },
            1e-6,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.to_lowercase().contains("invalid")
            || error.contains("plain weight")
            || error.contains("LayerNorm"),
        "LayerNorm + weight_offset must be refused, got: {error}"
    );
}
