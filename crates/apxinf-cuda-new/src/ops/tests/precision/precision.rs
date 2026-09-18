//! Torch-golden precision tests for every public L3 operator.
//!
//! Every new L3 operator must add `<op>_all_candidates_match_torch` here,
//! extend `generate_torch_l3_fixtures.py`, and regenerate
//! `torch_l3_fixtures.rs`. Tests compare final L3 outputs only; they must not
//! depend on or prescribe a candidate's internal implementation.

use super::framework::{bf16_bits_tensor, bytes_tensor, f32_tensor, scales, tensor, zeros_tensor};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};

#[path = "torch_l3_fixtures.rs"]
mod torch_fixture;

fn validate_all_candidates<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    bias: Option<&'a Tensor>,
    expected: &'a [f32],
) -> apxinf_core::Result<()> {
    let normalized = super::contracts::normalize(ctx, args, semantic, bias)?;
    super::execution::validate_candidates(ctx, &normalized, expected)
}

fn validate_all_candidates_with_residual<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    bias: &'a Tensor,
    residual: &'a Tensor,
    expected: &'a [f32],
) -> apxinf_core::Result<()> {
    let normalized =
        super::contracts::normalize_with_residual(ctx, args, semantic, Some(bias), residual)?;
    super::execution::validate_candidates(ctx, &normalized, expected)
}

fn configure_torch_case(args: &mut GemmArgs<'_>, alpha: f32, output_scale: f32) {
    args.alpha = alpha;
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
}

fn bf16_values(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|value| f32::from_bits(u32::from(*value) << 16))
        .collect()
}

#[test]
fn torch_validation_requires_the_l3_output_shape() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let args = GemmArgs::new(&a, &b, &mut out);
    let error = validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        &[1.0; 7],
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("Torch validation output does not match"));
}

#[test]
fn gemm_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::BF16_GEMM,
    )
    .unwrap();

    // GR00T static-FP8 uses the product of its activation and weight scales as
    // alpha and stores the projection directly as BF16.
    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_BF16_ALPHA, 1.0);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        &bf16_values(f::FP8_BF16_GEMM_BITS),
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_UNIT_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_SCALED_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::W8A8_GEMM,
    )
    .unwrap();
}

#[test]
fn gemm_bias_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::BF16_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::W8A8_GEMM_BIAS,
    )
    .unwrap();
}

#[test]
fn gemm_bias_gelu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::BF16_GEMM_BIAS_GELU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS_GELU,
    )
    .unwrap();
}

#[test]
fn gemm_geglu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, 2 * f::N], f::BF16_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::BF16_GEMM_GEGLU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, 2 * f::N], DType::F8E4M3, f::FP8_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::FP8_UNIT_GEMM_GEGLU,
    )
    .unwrap();
}

#[test]
fn gr00t_bias_activation_candidates_match_torch() {
    use torch_fixture as f;

    for (semantic, expected) in [
        (
            super::contracts::Semantic::GemmBiasRelu,
            f::BF16_GEMM_BIAS_RELU,
        ),
        (
            super::contracts::Semantic::GemmBiasSilu,
            f::BF16_GEMM_BIAS_SILU,
        ),
    ] {
        let ctx = CudaContext::new(0).unwrap();
        let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
        let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
        let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
        let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
        validate_all_candidates(&ctx, args, semantic, Some(&bias), expected).unwrap();
    }

    for (semantic, expected) in [
        (
            super::contracts::Semantic::GemmBiasRelu,
            f::FP8_SCALED_GEMM_BIAS_RELU,
        ),
        (
            super::contracts::Semantic::GemmBiasSilu,
            f::FP8_SCALED_GEMM_BIAS_SILU,
        ),
    ] {
        let ctx = CudaContext::new(0).unwrap();
        let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
        let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
        let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
        let row_scales = scales(0, f::ROW_SCALES);
        let channel_scales = scales(0, f::CHANNEL_SCALES);
        let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
        let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
        configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
        validate_all_candidates(&ctx, args, semantic, Some(&bias), expected).unwrap();
    }
}

#[test]
fn gr00t_bias_residual_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let residual = bf16_bits_tensor(0, vec![f::M, f::N], f::BF16_RESIDUAL);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates_with_residual(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasResidual,
        &bias,
        &residual,
        f::BF16_GEMM_BIAS_RESIDUAL,
    )
    .unwrap();
}

#[test]
fn gr00t_swiglu_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, 2 * f::N], f::BF16_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmSwiglu,
        None,
        f::BF16_GEMM_SWIGLU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, 2 * f::N], DType::I8, f::W8A8_SWIGLU_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::SWIGLU_CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmSwiglu,
        None,
        f::W8A8_GEMM_SWIGLU,
    )
    .unwrap();
}

#[test]
fn gr00t_w8a8_bias_family_candidates_match_torch() {
    use torch_fixture as f;

    for (semantic, expected) in [
        (
            super::contracts::Semantic::GemmBiasGelu,
            f::W8A8_GEMM_BIAS_GELU,
        ),
        (
            super::contracts::Semantic::GemmBiasRelu,
            f::W8A8_GEMM_BIAS_RELU,
        ),
        (
            super::contracts::Semantic::GemmBiasSilu,
            f::W8A8_GEMM_BIAS_SILU,
        ),
    ] {
        let ctx = CudaContext::new(0).unwrap();
        let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
        let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
        let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
        let row_scales = scales(0, f::ROW_SCALES);
        let channel_scales = scales(0, f::CHANNEL_SCALES);
        let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
        let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
        configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
        validate_all_candidates(&ctx, args, semantic, Some(&bias), expected).unwrap();
    }

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let residual = bf16_bits_tensor(0, vec![f::M, f::N], f::BF16_RESIDUAL);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates_with_residual(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasResidual,
        &bias,
        &residual,
        f::W8A8_GEMM_BIAS_RESIDUAL,
    )
    .unwrap();
}
