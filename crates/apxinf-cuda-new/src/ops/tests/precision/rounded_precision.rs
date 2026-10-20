use super::framework::{tensor, zeros_tensor};
use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};

#[path = "torch_rounded_fixtures.rs"]
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
fn rounded_primitives_all_candidates_match_torch() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(0, vec![2, 8], golden::X);
    let y = tensor(0, vec![2, 8], golden::Y);
    let w = tensor(0, vec![8], golden::WEIGHT);
    let rows = tensor(0, vec![2], &[0.75, -0.25]);
    let mut out = zeros_tensor(0, vec![2, 8], DType::BF16);
    for (semantic, secondary, expected) in [
        (PointwiseSemantic::Multiply, &y, golden::MULTIPLY),
        (PointwiseSemantic::Add, &y, golden::ADD),
        (
            PointwiseSemantic::BroadcastMultiply,
            &w,
            golden::BROADCAST_MULTIPLY,
        ),
        (PointwiseSemantic::BroadcastAdd, &w, golden::BROADCAST_ADD),
        (PointwiseSemantic::RowMultiply, &rows, golden::ROW_MULTIPLY),
        (PointwiseSemantic::EulerUpdateRounded, &y, golden::EULER),
    ] {
        let mut args = PointwiseArgs::new(semantic, &x, &mut out);
        args.secondary = Some(secondary);
        args.dt = -0.1777;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out), expected, "{semantic:?}");
    }
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &x, &mut out);
    args.activation = PointwiseActivation::GeluRounded;
    pointwise(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), golden::GELU);
    rms_norm_rounded(&ctx, RmsNormArgs::new(&x, &w, &mut out, 1e-6)).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), golden::RMS);
    let table = PairwiseRopeTable::new(&ctx, 2, 8, golden::COEFFICIENTS).unwrap();
    for (dtype, expected) in [
        (DType::BF16, golden::ROPE_BF16),
        (DType::F32, golden::ROPE_F32),
    ] {
        let mut out = zeros_tensor(0, vec![2, 8], dtype);
        pairwise_rope(
            &ctx,
            PairwiseRopeArgs {
                input: &x,
                table: &table,
                output: &mut out,
                heads: 1,
            },
        )
        .unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out), expected, "pairwise dtype={dtype:?}");
        let graph = crate::capture(&ctx, || {
            pairwise_rope(
                &ctx,
                PairwiseRopeArgs {
                    input: &x,
                    table: &table,
                    output: &mut out,
                    heads: 1,
                },
            )
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out), expected);
    }
}

#[test]
fn rounded_primitives_real_shapes_match_independent_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let round = |v: f32| half::bf16::from_f32(v).to_f32();
    for (rows, cols) in [(32, 1024), (98, 3072), (128, 4096), (128, 10240)] {
        let host: Vec<f32> = (0..rows * cols)
            .map(|i| round(((i * 13 % 251) as f32 - 125.0) / 31.0))
            .collect();
        let weight: Vec<f32> = (0..cols)
            .map(|i| round(0.5 + (i % 17) as f32 / 19.0))
            .collect();
        let input = tensor(0, vec![rows, cols], &host);
        let weight_tensor = tensor(0, vec![cols], &weight);
        let mut out = zeros_tensor(0, vec![rows, cols], DType::BF16);
        rms_norm_rounded(
            &ctx,
            RmsNormArgs::new(&input, &weight_tensor, &mut out, 1e-6),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let actual = values(&out);
        for r in 0..rows {
            let square_sum: f64 = host[r * cols..(r + 1) * cols]
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum();
            let inv = (square_sum / cols as f64 + 1e-6).sqrt().recip() as f32;
            for c in 0..cols {
                let expected = round(round(host[r * cols + c] * inv) * weight[c]);
                assert!(
                    (actual[r * cols + c] - expected).abs() <= 0.03125,
                    "r={r} c={c} shape={rows}x{cols}"
                );
            }
        }
        let graph = crate::capture(&ctx, || {
            rms_norm_rounded(
                &ctx,
                RmsNormArgs::new(&input, &weight_tensor, &mut out, 1e-6),
            )
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out), actual);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            graph.replay().unwrap();
        }
        ctx.synchronize().unwrap();
        eprintln!(
            "rounded_rms shape={rows}x{cols} graph_mean_us={:.3}",
            start.elapsed().as_secs_f64() * 1e4
        );
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
fn rounded_primitives_f16_f32_all_candidates_match_torch() {
    use super::framework::{f16_tensor, f32_tensor};
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::F32] {
        let make = if dtype == DType::F16 {
            f16_tensor
        } else {
            f32_tensor
        };
        let (xv, yv, wv, gelu, rms, euler, mul, add, bmul, badd, rmul) = if dtype == DType::F16 {
            (
                golden::F16_X,
                golden::F16_Y,
                golden::F16_WEIGHT,
                golden::F16_GELU,
                golden::F16_RMS,
                golden::F16_EULER,
                golden::F16_MULTIPLY,
                golden::F16_ADD,
                golden::F16_BROADCAST_MULTIPLY,
                golden::F16_BROADCAST_ADD,
                golden::F16_ROW_MULTIPLY,
            )
        } else {
            (
                golden::F32_X,
                golden::F32_Y,
                golden::F32_WEIGHT,
                golden::F32_GELU,
                golden::F32_RMS,
                golden::F32_EULER,
                golden::F32_MULTIPLY,
                golden::F32_ADD,
                golden::F32_BROADCAST_MULTIPLY,
                golden::F32_BROADCAST_ADD,
                golden::F32_ROW_MULTIPLY,
            )
        };
        let x = make(0, vec![2, 8], xv);
        let y = make(0, vec![2, 8], yv);
        let w = make(0, vec![8], wv);
        let row = make(0, vec![2], &[0.75, -0.25]);
        let mut out = zeros_tensor(0, vec![2, 8], dtype);
        let assert_close = |actual: Vec<f32>, expected: &[f32]| {
            for (a, &e) in actual.iter().zip(expected) {
                assert!((a - e).abs() <= 1e-6, "dtype={dtype:?}: {a} != {e}");
            }
        };
        for (semantic, secondary, expected) in [
            (PointwiseSemantic::Multiply, &y, mul),
            (PointwiseSemantic::Add, &y, add),
            (PointwiseSemantic::BroadcastMultiply, &w, bmul),
            (PointwiseSemantic::BroadcastAdd, &w, badd),
            (PointwiseSemantic::RowMultiply, &row, rmul),
            (PointwiseSemantic::EulerUpdateRounded, &y, euler),
        ] {
            let mut args = PointwiseArgs::new(semantic, &x, &mut out);
            args.secondary = Some(secondary);
            args.dt = -0.1777;
            pointwise(&ctx, args).unwrap();
            ctx.synchronize().unwrap();
            assert_close(values(&out), expected);
        }
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &x, &mut out);
        args.activation = PointwiseActivation::GeluRounded;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        assert_close(values(&out), gelu);
        if dtype == DType::F16 {
            rms_norm_rounded(&ctx, RmsNormArgs::new(&x, &w, &mut out, 1e-6)).unwrap();
            ctx.synchronize().unwrap();
            assert_close(values(&out), rms);
        } else {
            assert!(rms_norm_rounded(&ctx, RmsNormArgs::new(&x, &w, &mut out, 1e-6)).is_err());
        }
    }
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
fn rounded_capture_retains_operand_and_coefficient_owners() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(0, vec![2, 8], golden::X);
    let table = PairwiseRopeTable::new(&ctx, 2, 8, golden::COEFFICIENTS).unwrap();
    let mut output = zeros_tensor(0, vec![2, 8], DType::F32);
    let graph = crate::capture(&ctx, || {
        pairwise_rope(
            &ctx,
            PairwiseRopeArgs {
                input: &x,
                table: &table,
                output: &mut output,
                heads: 1,
            },
        )
    })
    .unwrap();
    drop(x);
    drop(table);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&output), golden::ROPE_F32);
    let x = tensor(0, vec![2, 8], golden::X);
    let weight = tensor(0, vec![8], golden::WEIGHT);
    let mut norm = zeros_tensor(0, vec![2, 8], DType::BF16);
    let mut product = zeros_tensor(0, vec![2, 8], DType::BF16);
    let graph = crate::capture(&ctx, || {
        rms_norm_rounded(&ctx, RmsNormArgs::new(&x, &weight, &mut norm, 1e-6))?;
        let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, &norm, &mut product);
        args.secondary = Some(&norm);
        pointwise(&ctx, args)
    })
    .unwrap();
    drop(x);
    drop(weight);
    drop(norm);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let expected = golden::RMS
        .iter()
        .map(|x| half::bf16::from_f32(x * x).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(values(&product), expected);
}

#[test]
fn rounded_primitives_reject_partial_and_broadcast_aliases() {
    use apxinf_core::Shape;
    let ctx = CudaContext::new(0).unwrap();
    let storage = CudaBuffer::alloc_zeros(64, 0).unwrap();
    let input = storage
        .view(0, 32)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 8]), DType::BF16)
        .unwrap();
    let mut overlapping = storage
        .view(2, 32)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 8]), DType::BF16)
        .unwrap();
    let secondary = tensor(0, vec![2, 8], golden::Y);
    let mut args = PointwiseArgs::new(PointwiseSemantic::Add, &input, &mut overlapping);
    args.secondary = Some(&secondary);
    assert!(pointwise(&ctx, args)
        .unwrap_err()
        .to_string()
        .contains("partially overlap"));
    let weight = tensor(0, vec![8], golden::WEIGHT);
    assert!(rms_norm_rounded(
        &ctx,
        RmsNormArgs::new(&input, &weight, &mut overlapping, 1e-6)
    )
    .unwrap_err()
    .to_string()
    .contains("partially overlap"));
    let table = PairwiseRopeTable::new(&ctx, 2, 8, golden::COEFFICIENTS).unwrap();
    assert!(pairwise_rope(
        &ctx,
        PairwiseRopeArgs {
            input: &input,
            table: &table,
            output: &mut overlapping,
            heads: 1
        }
    )
    .is_err());
    let mut widened = storage
        .as_tensor(Shape::new(vec![2, 8]), DType::F32)
        .unwrap();
    assert!(pairwise_rope(
        &ctx,
        PairwiseRopeArgs {
            input: &input,
            table: &table,
            output: &mut widened,
            heads: 1
        }
    )
    .is_err());
    let columns = storage
        .view(0, 16)
        .unwrap()
        .as_tensor(Shape::new(vec![8]), DType::BF16)
        .unwrap();
    let rows = storage
        .view(0, 4)
        .unwrap()
        .as_tensor(Shape::new(vec![2]), DType::BF16)
        .unwrap();
    let mut same = input.clone();
    for (semantic, secondary) in [
        (PointwiseSemantic::BroadcastMultiply, &columns),
        (PointwiseSemantic::BroadcastAdd, &columns),
        (PointwiseSemantic::RowMultiply, &rows),
    ] {
        let mut args = PointwiseArgs::new(semantic, &input, &mut same);
        args.secondary = Some(secondary);
        assert!(pointwise(&ctx, args)
            .unwrap_err()
            .to_string()
            .contains("secondary/output overlap"));
    }
    assert!(
        rms_norm_rounded(&ctx, RmsNormArgs::new(&input, &columns, &mut same, 1e-6))
            .unwrap_err()
            .to_string()
            .contains("weight/output overlap")
    );
}

#[test]
fn rounded_primitives_allow_exact_elementwise_and_norm_aliases() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 8], golden::X);
    let secondary = tensor(0, vec![2, 8], golden::Y);
    let mut output = input.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, &input, &mut output);
    args.secondary = Some(&secondary);
    pointwise(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&output), golden::MULTIPLY);
    let input = tensor(0, vec![2, 8], golden::X);
    let weight = tensor(0, vec![8], golden::WEIGHT);
    let mut output = input.clone();
    rms_norm_rounded(&ctx, RmsNormArgs::new(&input, &weight, &mut output, 1e-6)).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&output), golden::RMS);
    let input = tensor(0, vec![2, 8], golden::X);
    let mut output = input.clone();
    let table = PairwiseRopeTable::new(&ctx, 2, 8, golden::COEFFICIENTS).unwrap();
    pairwise_rope(
        &ctx,
        PairwiseRopeArgs {
            input: &input,
            table: &table,
            output: &mut output,
            heads: 1,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&output), golden::ROPE_BF16);
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
fn rounded_gelu_preserves_eager_cuda_pow3_rounding() {
    use super::framework::f16_tensor;
    let ctx = CudaContext::new(0).unwrap();
    for (dtype, input, expected, once_rounded_cube) in [
        (
            DType::BF16,
            golden::BF16_POW3_SENSITIVE_X,
            golden::BF16_POW3_SENSITIVE_GELU,
            golden::BF16_ONCE_ROUNDED_CUBE_GELU,
        ),
        (
            DType::F16,
            golden::F16_POW3_SENSITIVE_X,
            golden::F16_POW3_SENSITIVE_GELU,
            golden::F16_ONCE_ROUNDED_CUBE_GELU,
        ),
    ] {
        assert_ne!(
            expected, once_rounded_cube,
            "fixture must discriminate the cube roundpoint"
        );
        let make = if dtype == DType::BF16 {
            tensor
        } else {
            f16_tensor
        };
        let input = make(0, vec![1, input.len()], input);
        let mut output = zeros_tensor(0, input.shape().dims().to_vec(), dtype);
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
        args.activation = PointwiseActivation::GeluRounded;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&output), expected, "dtype={dtype:?}");
    }
}
