//! Independent F64 Welford reference for the fixed LayerNorm candidate.
use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};

fn upload(shape: Vec<usize>, values: &[f32]) -> Tensor {
    crate::transfers::to_cuda(&Tensor::from_f32(shape, values).unwrap(), 0).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    crate::transfers::to_cpu(tensor)
        .unwrap()
        .to_f32_vec()
        .unwrap()
}

fn reference(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    input
        .chunks_exact(weight.len())
        .flat_map(|row| {
            // Online F64 moments are independent of the GPU's two-pass F32 tree.
            let (mut mean, mut m2) = (0.0f64, 0.0f64);
            for (index, &value) in row.iter().enumerate() {
                let delta = value as f64 - mean;
                mean += delta / (index + 1) as f64;
                m2 += delta * (value as f64 - mean);
            }
            let inv = (m2 / row.len() as f64 + eps as f64).sqrt().recip();
            row.iter()
                .zip(weight)
                .zip(bias)
                .map(move |((&x, &w), &b)| (((x as f64 - mean) * inv) * w as f64 + b as f64) as f32)
        })
        .collect()
}

fn check(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 3e-5 + 3e-5 * b.abs(),
            "LayerNorm index={index}: actual={a}, reference={b}"
        );
    }
}

#[test]
fn layer_norm_f32_real_shapes_match_independent_f64_eager_and_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for (rows, cols) in [(32, 1024), (98, 3072)] {
        for affine in [false, true] {
            let host = (0..rows * cols)
                .map(|i| {
                    match i / cols % 8 {
                        0 => 1.25, // Zero variance and nonzero mean.
                        1 => ((i * 17 % 251) as f32 - 125.0) * 0.0000013,
                        2 => 32.0 + ((i * 13 % 251) as f32 - 125.0) / 31.0,
                        4 => 1000.1, // Non-exact local sums previously shifted the mean.
                        5 => 10000.1,
                        6 => 10000.1 + (i % 17) as f32 * 0.0013,
                        7 => -100000.1 + (i % 19) as f32 * 0.0079,
                        _ => ((i * 13 % 251) as f32 - 125.0) / 31.0 + (i / cols) as f32 * 0.0013,
                    }
                })
                .collect::<Vec<_>>();
            assert!(host.iter().any(|&x| half::bf16::from_f32(x).to_f32() != x));
            let weight = (0..cols)
                .map(|i| {
                    if affine {
                        (i % 19) as f32 / 11.0 - 0.8137
                    } else {
                        1.0
                    }
                })
                .collect::<Vec<_>>();
            let bias = (0..cols)
                .map(|i| {
                    if affine {
                        (i % 13) as f32 / 23.0 - 0.2703
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let input = upload(vec![rows, cols], &host);
            let w = upload(vec![cols], &weight);
            let b = upload(vec![cols], &bias);
            let mut output = ctx
                .allocate_output(input.shape().clone(), DType::F32)
                .unwrap();
            crate::ops::norm::norm(&ctx, NormArgs::layer(&input, &w, &b, &mut output, 1e-6)).unwrap();
            ctx.synchronize().unwrap();
            let eager = values(&output);
            check(&eager, &reference(&host, &weight, &bias, 1e-6));
            let graph = crate::capture(&ctx, || {
                crate::ops::norm::norm(&ctx, NormArgs::layer(&input, &w, &b, &mut output, 1e-6))
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(values(&output), eager);
            // A captured graph observes new values at the same safe binding.
            let changed = host
                .iter()
                .enumerate()
                .map(|(i, &x)| x * 0.713 + (i % 7) as f32 * 0.0017)
                .collect::<Vec<_>>();
            crate::transfers::copy_cpu_to_cuda(
                &Tensor::from_f32(vec![rows, cols], &changed).unwrap(),
                &input,
            )
            .unwrap();
            // Baseline norm capture borrows operand storage; retain it through replay.
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            check(&values(&output), &reference(&changed, &weight, &bias, 1e-6));
        }
    }
}

#[test]
fn layer_norm_f32_rejects_mixed_dtypes_aliases_and_other_norm_semantics() {
    let ctx = CudaContext::new(0).unwrap();
    let input = upload(vec![2, 1024], &vec![0.25; 2048]);
    let weight = upload(vec![1024], &vec![1.0; 1024]);
    let bias = upload(vec![1024], &vec![0.0; 1024]);
    let mut output = ctx
        .allocate_output(input.shape().clone(), DType::F32)
        .unwrap();
    let bad = ctx
        .allocate_output(weight.shape().clone(), DType::BF16)
        .unwrap();
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &bad, &bias, &mut output, 1e-6)
    )
    .is_err());
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &weight, &bad, &mut output, 1e-6)
    )
    .is_err());
    let mut bad_output = ctx
        .allocate_output(input.shape().clone(), DType::BF16)
        .unwrap();
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &weight, &bias, &mut bad_output, 1e-6)
    )
    .is_err());
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(crate::ops::norm::norm(
            &ctx,
            NormArgs::layer(&input, &weight, &bias, &mut output, eps)
        )
        .is_err());
    }
    let mut aliased = input.clone();
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &weight, &bias, &mut aliased, 1e-6)
    )
    .is_err());
    let allocation = ctx.allocate_output(vec![2049].into(), DType::F32).unwrap();
    let buffer = CudaBuffer::from_tensor(&allocation).unwrap();
    let read = buffer
        .view(0, 8192)
        .unwrap()
        .as_tensor(vec![2, 1024].into(), DType::F32)
        .unwrap();
    let mut write = buffer
        .view(4, 8192)
        .unwrap()
        .as_tensor(vec![2, 1024].into(), DType::F32)
        .unwrap();
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&read, &weight, &bias, &mut write, 1e-6)
    )
    .is_err());
    let alias_weight = CudaBuffer::from_tensor(&output)
        .unwrap()
        .view(0, 4096)
        .unwrap()
        .as_tensor(vec![1024].into(), DType::F32)
        .unwrap();
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &alias_weight, &bias, &mut output, 1e-6)
    )
    .is_err());
    assert!(crate::ops::norm::norm(
        &ctx,
        NormArgs::layer(&input, &weight, &alias_weight, &mut output, 1e-6)
    )
    .is_err());
    assert!(crate::ops::norm::norm(&ctx, NormArgs::rms(&input, &weight, &mut output, 1e-6)).is_err());
    assert!(residual_add(
        &ctx,
        ResidualAddArgs { input: &input, bias: None, residual: &input, gate: None, hidden: &output, bias_order: BiasOrder::SingleRound }
    )
    .is_err());
}
