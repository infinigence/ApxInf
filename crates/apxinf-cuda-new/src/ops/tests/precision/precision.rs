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

fn configure_torch_case(args: &mut GemmArgs<'_>, alpha: f32, output_scale: f32) {
    args.alpha = alpha;
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
}

fn nvfp4_scales(rows: usize, k: usize, mut scale: impl FnMut(usize, usize) -> u8) -> Vec<u8> {
    let mut output = vec![0; rows * (k / 16)];
    for row in 0..rows {
        for block in 0..k / 16 {
            output[row * (k / 16) + block] = scale(row, block);
        }
    }
    output
}

fn packed_e2m1(count: usize, mut value: impl FnMut(usize) -> u8) -> Vec<u8> {
    let mut output = vec![0; count.div_ceil(2)];
    for logical in 0..count {
        output[logical / 2] |= (value(logical) & 0xf) << ((logical % 2) * 4);
    }
    output
}

fn f16_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&value| half::f16::from_f32(value).to_bits().to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::F16, &bytes)
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
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
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
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

    // Exact Torch oracle case. E2M1 values are 1 while block scales alternate
    // between exactly representable UE4M3 values 1 and 2. This exercises
    // packed nibble order, non-uniform block-scale addressing, alpha, all
    // NVFP4 configurations, and the final F16 L3 output without folding first
    // quantization error into the candidate tolerance.
    let (m, n, k) = (16, 256, 128);
    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    // Both operands use the canonical row-major public L3 layout.
    let b_bytes = packed_e2m1(k * n, |logical| {
        let column = logical % n;
        if column % 2 == 0 {
            2
        } else {
            4
        }
    });
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &b_bytes);
    let a_scale_bytes = nvfp4_scales(
        m,
        k,
        |row, block| {
            if (row + block) % 2 == 0 {
                0x38
            } else {
                0x40
            }
        },
    );
    let b_scale_bytes = nvfp4_scales(n, k, |row, block| {
        if (row + 2 * block) % 3 == 0 {
            0x40
        } else {
            0x38
        }
    });
    let a_scales = bytes_tensor(0, vec![m, k / 16], DType::F8UE4M3, &a_scale_bytes);
    let b_scales = bytes_tensor(0, vec![n, k / 16], DType::F8UE4M3, &b_scale_bytes);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
    configure_torch_case(&mut args, 1.25, 2.5);
    let expected: Vec<f32> = (0..m)
        .flat_map(|row| {
            (0..n).map(move |column| {
                0.5 * (0..k / 16)
                    .map(|block| {
                        let a_scale = if (row + block) % 2 == 0 { 1.0 } else { 2.0 };
                        let b_scale = if (column + 2 * block) % 3 == 0 {
                            2.0
                        } else {
                            1.0
                        };
                        let b_value = if column % 2 == 0 { 1.0 } else { 2.0 };
                        16.0 * a_scale * b_scale * b_value
                    })
                    .sum::<f32>()
            })
        })
        .collect();
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        &expected,
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

/// Expensive Pi0.5 shape coverage. Run explicitly with
/// `cargo test nvfp4_pi05_shapes_match_reference -- --ignored --nocapture`.
#[test]
#[ignore = "large Pi0.5 GPU acceptance matrix"]
fn nvfp4_pi05_shapes_match_reference() {
    for (name, m, n, k) in [
        ("vision-qkv", 512, 3456, 1152),
        ("vision-o", 512, 1152, 1152),
        ("vision-fc1", 512, 4304, 1152),
        ("multimodal-projector", 512, 2048, 1152),
        ("language-qkv", 968, 2560, 2048),
        ("language-o", 968, 2048, 2048),
        ("language-gate", 968, 16384, 2048),
        ("language-up", 968, 16384, 2048),
        ("language-down", 968, 2048, 16384),
        ("action-qkv", 10, 2560, 1024),
        ("action-o", 10, 1024, 2048),
        ("action-gate", 10, 4096, 1024),
        ("action-up", 10, 4096, 1024),
        ("action-down", 10, 1024, 4096),
        ("action-output", 10, 32, 1024),
    ] {
        let ctx = CudaContext::new(0).unwrap();
        let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
        let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; k * n / 2]);
        let a_scales = bytes_tensor(
            0,
            vec![m, k / 16],
            DType::F8UE4M3,
            &vec![0x38; m * (k / 16)],
        );
        let b_scales = bytes_tensor(
            0,
            vec![n, k / 16],
            DType::F8UE4M3,
            &vec![0x38; n * (k / 16)],
        );
        let mut out = zeros_tensor(0, vec![m, n], DType::F16);
        let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
        configure_torch_case(&mut args, 1.0, 1.0);
        let expected = vec![k as f32; m * n];
        validate_all_candidates(
            &ctx,
            args,
            super::contracts::Semantic::Gemm,
            None,
            &expected,
        )
        .unwrap();
        println!("NVFP4 {name}: pass");
    }
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
    let (m, n, k) = (16, 256, 128);
    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b_bytes = packed_e2m1(k * n, |logical| if (logical % n) % 2 == 0 { 2 } else { 4 });
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &b_bytes);
    let a_scales = bytes_tensor(
        0,
        vec![m, k / 16],
        DType::F8UE4M3,
        &vec![0x20; m * (k / 16)],
    );
    let b_scales = bytes_tensor(
        0,
        vec![n, k / 16],
        DType::F8UE4M3,
        &vec![0x20; n * (k / 16)],
    );
    let bias_values: Vec<f32> = (0..n)
        .map(|column| if column % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let bias = f16_tensor(0, vec![n], &bias_values);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
    configure_torch_case(&mut args, 1.25, 2.0);
    let expected: Vec<f32> = (0..m)
        .flat_map(|_| {
            bias_values.iter().enumerate().map(|(column, &bias)| {
                let b_value = if column % 2 == 0 { 1.0 } else { 2.0 };
                gelu(1.25 * (2.0 * b_value) + bias) / 2.0
            })
        })
        .collect();
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        &expected,
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
    let (m, width, k) = (16, 256, 128);
    let projection_width = 2 * width;
    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b_bytes = packed_e2m1(k * projection_width, |logical| {
        if logical % projection_width < width {
            2
        } else {
            4
        }
    });
    let b = bytes_tensor(0, vec![k, projection_width], DType::F4E2M1, &b_bytes);
    let a_scales = bytes_tensor(
        0,
        vec![m, k / 16],
        DType::F8UE4M3,
        &vec![0x20; m * (k / 16)],
    );
    let b_scales = bytes_tensor(
        0,
        vec![projection_width, k / 16],
        DType::F8UE4M3,
        &vec![0x20; projection_width * (k / 16)],
    );
    let mut out = zeros_tensor(0, vec![m, width], DType::F16);
    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
    configure_torch_case(&mut args, 1.25, 2.0);
    let expected = vec![gelu(1.25 * 2.0) * (1.25 * 4.0) / 2.0; m * width];
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        &expected,
    )
    .unwrap();
}
