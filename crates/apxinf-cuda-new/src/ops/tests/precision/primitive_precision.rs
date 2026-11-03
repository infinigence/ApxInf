use super::framework::{tensor, zeros_tensor};
use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};

#[path = "torch_primitive_fixtures.rs"]
mod golden;

fn values(t: &Tensor) -> Vec<f32> {
    let b = CudaBuffer::from_tensor(t).unwrap();
    let mut bytes = vec![0; b.len()];
    b.copy_to_host(&mut bytes).unwrap();
    match t.dtype() {
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32())
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|v| half::f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32())
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect(),
        _ => panic!("unexpected dtype"),
    }
}

#[test]
fn pointwise_f32_binary_real_shape_and_graph() {
    use super::framework::f32_tensor;
    let ctx = CudaContext::new(0).unwrap();
    let rows = 128;
    let cols = 4096;
    let host: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 7 % 101) as f32 - 50.0) / 23.0)
        .collect();
    let masks: Vec<f32> = (0..rows).map(|i| if i < 93 { 1.0 } else { 0.0 }).collect();
    let input = f32_tensor(0, vec![rows, cols], &host);
    let mask = f32_tensor(0, vec![rows], &masks);
    let mut out = zeros_tensor(0, vec![rows, cols], DType::F32);
    let run = |out: &mut Tensor| {
        let mut args = PointwiseArgs::new(PointwiseSemantic::RowMultiply, &input, out);
        args.secondary = Some(&mask);
        pointwise(&ctx, args)
    };
    run(&mut out).unwrap();
    ctx.synchronize().unwrap();
    let expected: Vec<f32> = host
        .iter()
        .enumerate()
        .map(|(i, &x)| x * masks[i / cols])
        .collect();
    assert_eq!(values(&out), expected);
    let graph = crate::capture(&ctx, || run(&mut out)).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn pointwise_f32_extended_candidate_matches_torch() {
    use super::framework::f32_tensor;
    let ctx = CudaContext::new(0).unwrap();
    let x = f32_tensor(0, vec![2, 8], golden::F32_X);
    let y = f32_tensor(0, vec![2, 8], golden::F32_Y);
    let mut out = zeros_tensor(0, vec![2, 8], DType::F32);
    let check = |out: &Tensor, expected: &[f32]| {
        ctx.synchronize().unwrap();
        for (a, &e) in values(out).iter().zip(expected) {
            assert!((a - e).abs() < 2e-6, "{a} != {e}");
        }
    };
    for (activation, expected) in [
        (PointwiseActivation::Gelu, golden::F32_GELU_FUSED),
        (PointwiseActivation::GeluExact, golden::F32_GELU_EXACT),
        (PointwiseActivation::Silu, golden::F32_SILU),
    ] {
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &x, &mut out);
        args.activation = activation;
        pointwise(&ctx, args).unwrap();
        check(&out, expected);
    }
    let mut packed = Vec::new();
    for row in 0..2 {
        packed.extend_from_slice(&golden::F32_X[row * 8..(row + 1) * 8]);
        packed.extend_from_slice(&golden::F32_Y[row * 8..(row + 1) * 8]);
    }
    let input = f32_tensor(0, vec![2, 16], &packed);
    pointwise(
        &ctx,
        PointwiseArgs::new(PointwiseSemantic::Geglu, &input, &mut out),
    )
    .unwrap();
    check(&out, golden::F32_GEGLU);
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &x, &mut out);
    args.secondary = Some(&y);
    args.dt = -0.1777;
    pointwise(&ctx, args).unwrap();
    check(&out, golden::F32_EULER_FUSED);
}

#[test]
fn rounded_pairwise_rope_real_shapes_and_changed_input_replay() {
    let ctx = CudaContext::new(0).unwrap();
    for (tokens, heads, dim) in [(32, 16, 64), (98, 24, 128)] {
        let input_values: Vec<f32> = (0..tokens * heads * dim)
            .map(|i| half::bf16::from_f32(((i % 101) as f32 - 50.0) / 17.0).to_f32())
            .collect();
        let coefficients: Vec<(f64, f64)> = (0..tokens * dim / 2)
            .map(|i| {
                let angle = (i / (dim / 2)) as f64 * 0.17 + (i % (dim / 2)) as f64 * 0.031;
                (angle.cos(), angle.sin())
            })
            .collect();
        let expected: Vec<f32> = input_values
            .chunks_exact(2)
            .enumerate()
            .flat_map(|(pair, xy)| {
                let token = pair / (heads * dim / 2);
                let (c, s) = coefficients[token * dim / 2 + pair % (dim / 2)];
                let (x, y) = (xy[0] as f64, xy[1] as f64);
                [(x * c - y * s) as f32, (x * s + y * c) as f32]
            })
            .collect();
        let table = PairwiseRopeTable::new(&ctx, tokens, dim, &coefficients).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            let input = tensor(0, vec![tokens, heads * dim], &input_values);
            let mut output = zeros_tensor(0, vec![tokens, heads * dim], dtype);
            let graph = crate::capture(&ctx, || {
                pairwise_rope(
                    &ctx,
                    PairwiseRopeArgs {
                        input: &input,
                        table: &table,
                        output: &mut output,
                        heads,
                    },
                )
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            let rounded: Vec<f32> = expected
                .iter()
                .map(|&x| {
                    if dtype == DType::BF16 {
                        half::bf16::from_f32(x).to_f32()
                    } else {
                        x
                    }
                })
                .collect();
            assert_eq!(
                values(&output),
                rounded,
                "tokens={tokens} heads={heads} dim={dim} dtype={dtype:?}"
            );
            let changed: Vec<u8> = input_values
                .iter()
                .flat_map(|&x| half::bf16::from_f32(-x).to_bits().to_ne_bytes())
                .collect();
            CudaBuffer::from_tensor(&input)
                .unwrap()
                .copy_from_host(&changed)
                .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                values(&output),
                rounded.iter().map(|&x| -x).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn rms_intermediate_dtype_preserves_default_and_explicit_boundary() {
    let ctx = CudaContext::new(0).unwrap();
    let (rows, cols) = (128, 4096);
    let host: Vec<f32> = (0..rows * cols)
        .map(|i| half::bf16::from_f32(((i * 17 % 251) as f32 - 125.0) / 37.0).to_f32())
        .collect();
    let weights: Vec<f32> = (0..cols)
        .map(|i| half::bf16::from_f32((i % 19) as f32 / 13.0 + 0.25).to_f32())
        .collect();
    let input = tensor(0, vec![rows, cols], &host);
    let weight = tensor(0, vec![cols], &weights);
    let mut out = zeros_tensor(0, vec![rows, cols], DType::BF16);
    let mut baseline = Vec::new();
    for intermediate_dtype in [DType::F32, DType::BF16] {
        let run = |out: &mut Tensor| {
            let mut args = NormArgs::rms(&input, &weight, out, 1e-6);
            assert_eq!(args.intermediate_dtype, DType::F32);
            args.intermediate_dtype = intermediate_dtype;
            crate::ops::norm::norm(&ctx, args)
        };
        run(&mut out).unwrap();
        ctx.synchronize().unwrap();
        let actual = values(&out);
        let mut error = 0f64;
        let mut norm = 0f64;
        for (row, a) in host.chunks_exact(cols).zip(actual.chunks_exact(cols)) {
            let inv = (row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / cols as f64 + 1e-6)
                .sqrt()
                .recip();
            for ((&x, &w), &a) in row.iter().zip(&weights).zip(a) {
                let mut y = (x as f64 * inv) as f32;
                if intermediate_dtype == DType::BF16 {
                    y = half::bf16::from_f32(y).to_f32();
                }
                let y = half::bf16::from_f32(y * w).to_f32();
                error += (a as f64 - y as f64).powi(2);
                norm += (y as f64).powi(2);
            }
        }
        assert!(
            (error / norm).sqrt() < 1e-4,
            "RMS {:?} relative_l2={}",
            intermediate_dtype,
            (error / norm).sqrt()
        );
        if intermediate_dtype == DType::F32 {
            baseline = actual.clone();
        } else {
            assert_ne!(baseline, actual);
        }
        let graph = crate::capture(&ctx, || run(&mut out)).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out), actual);
        let started = std::time::Instant::now();
        for _ in 0..100 {
            graph.replay().unwrap();
        }
        ctx.synchronize().unwrap();
        eprintln!(
            "RMS {:?} graph_replay_mean_us={}",
            intermediate_dtype,
            started.elapsed().as_secs_f64() * 1e4
        );
    }
    let mut invalid = NormArgs::rms(&input, &weight, &mut out, 1e-6);
    invalid.intermediate_dtype = DType::F16;
    assert!(crate::ops::norm::norm(&ctx, invalid).is_err());
}
