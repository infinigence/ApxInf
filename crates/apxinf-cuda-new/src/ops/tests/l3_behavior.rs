//! Per-operator L3 semantic and conditional Graph behavior tests.
//!
//! Every new L3 operator must add a public semantic-contract test here. Add a
//! Graph replay test only when the operator has an independent execution path,
//! resource lifetime, binding rule, or capture behavior not already covered by
//! the shared framework tests.

use super::framework::{bytes_tensor, f16_values, tensor, values, zeros_tensor};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};
use half::bf16;

fn nvfp4_scales(rows: usize, k: usize, mut scale: impl FnMut(usize, usize) -> u8) -> Vec<u8> {
    let mut output = vec![0; rows * (k / 16)];
    for row in 0..rows {
        for block in 0..k / 16 {
            output[row * (k / 16) + block] = scale(row, block);
        }
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

#[test]
fn nvfp4_gemm_rejects_missing_or_wrong_block_scale_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (16, 16, 128);
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; n * k / 2]);
    let wrong_a_scales = zeros_tensor(0, vec![m, k / 16], DType::F32);
    let b_scale_bytes = n * (k / 16);
    let b_scales = bytes_tensor(
        0,
        vec![n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; b_scale_bytes],
    );
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let error = gemm(
        &ctx,
        GemmArgs::nvfp4(&a, &wrong_a_scales, &b, &b_scales, &mut out),
    )
    .unwrap_err();
    assert!(error.to_string().contains("device/dtype/shape mismatch"));
}

#[test]
fn nvfp4_gemm_prepares_captures_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (16, 256, 128);
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; n * k / 2]);
    let scale_bytes = |rows: usize| rows * (k / 16);
    let a_scales = bytes_tensor(
        0,
        vec![m, k / 16],
        DType::F8UE4M3,
        &vec![0x38; scale_bytes(m)],
    );
    let b_scales = bytes_tensor(
        0,
        vec![n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; scale_bytes(n)],
    );
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let output_buffer = crate::CudaBuffer::from_tensor(&out).unwrap();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    let run = |out: &mut Tensor| {
        let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, out);
        args.policy.online_tune = false;
        gemm(&ctx, args)
    };
    prepare_with_session(&session, || run(&mut out)).unwrap();
    let graph = crate::capture(&ctx, || with_session(&session, || run(&mut out))).unwrap();

    output_buffer
        .copy_from_host(&vec![0xff; m * n * 2])
        .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&out).iter().all(|&value| value == k as f32));
}

#[test]
fn nvfp4_online_tuning_uses_the_dequantized_block_scale_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (17, 256, 128);
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; n * k / 2]);
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
    let b_scale_bytes = nvfp4_scales(n, k, |column, block| {
        if (column + block) % 2 == 0 {
            0x40
        } else {
            0x38
        }
    });
    let a_scales = bytes_tensor(0, vec![m, k / 16], DType::F8UE4M3, &a_scale_bytes);
    let b_scales = bytes_tensor(0, vec![n, k / 16], DType::F8UE4M3, &b_scale_bytes);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    gemm(&ctx, args).unwrap();

    let actual = f16_values(&out);
    for row in 0..m {
        for column in 0..n {
            let expected: f32 = (0..k / 16)
                .map(|block| {
                    let a_scale = if (row + block) % 2 == 0 { 1.0 } else { 2.0 };
                    let b_scale = if (column + block) % 2 == 0 { 2.0 } else { 1.0 };
                    16.0 * a_scale * b_scale
                })
                .sum();
            assert_eq!(actual[row * n + column], expected);
        }
    }
}

#[test]
fn nvfp4_fused_semantics_prepare_capture_and_replay() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (16, 256, 128);
    let a = bytes_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = bytes_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; k * n / 2]);
    let b_geglu = bytes_tensor(0, vec![k, 2 * n], DType::F4E2M1, &vec![0x22; k * 2 * n / 2]);
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
    let b_geglu_scales = bytes_tensor(
        0,
        vec![2 * n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; 2 * n * (k / 16)],
    );
    let bias = f16_tensor(0, vec![n], &vec![0.0; n]);
    let mut gelu_out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut geglu_out = zeros_tensor(0, vec![m, n], DType::F16);
    let gelu_buffer = crate::CudaBuffer::from_tensor(&gelu_out).unwrap();
    let geglu_buffer = crate::CudaBuffer::from_tensor(&geglu_out).unwrap();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    let mut run = || -> apxinf_core::Result<()> {
        let mut gelu_args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut gelu_out);
        gelu_args.policy.online_tune = false;
        gemm_bias_gelu(
            &ctx,
            GemmBiasGeluArgs {
                gemm: gelu_args,
                bias: &bias,
            },
        )?;
        let mut geglu_args =
            GemmArgs::nvfp4(&a, &a_scales, &b_geglu, &b_geglu_scales, &mut geglu_out);
        geglu_args.policy.online_tune = false;
        gemm_geglu(&ctx, GemmGegluArgs { gemm: geglu_args })
    };

    prepare_with_session(&session, &mut run).unwrap();
    let graph = crate::capture(&ctx, || with_session(&session, &mut run)).unwrap();
    drop(run);
    gelu_buffer.copy_from_host(&vec![0xff; m * n * 2]).unwrap();
    geglu_buffer.copy_from_host(&vec![0xff; m * n * 2]).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&gelu_out).iter().all(|&value| value == 128.0));
    assert!(f16_values(&geglu_out)
        .iter()
        .all(|&value| value == 16_384.0));
}

#[test]
fn gemm_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (3, 5, 1024);
    let a_values = vec![0.25; m * k];
    let b_values: Vec<_> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n / 2], &vec![0.0; m * n / 2]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();
    let actual = values(&out);
    assert_eq!(actual.len(), m * n / 2);
    for row in 0..m {
        for column in 0..n / 2 {
            let dot = |target| {
                bf16::from_f32(
                    (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + target])
                        .sum(),
                )
                .to_f32()
            };
            let gate = dot(column);
            let expected = 0.5
                * gate
                * (1.0 + (0.79788456 * (gate + 0.044715 * gate.powi(3))).tanh())
                * dot(column + n / 2);
            assert!((actual[row * n / 2 + column] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5; 4]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let output = values(&out);
    let x: f32 = 3.5;
    let expected = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
    assert!(output.iter().all(|value| (*value - expected).abs() < 0.02));
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let expected = [3.5, 2.5, 4.0, 2.0, 3.5, 2.5, 4.0, 2.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.01);
    }
}
