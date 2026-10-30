//! Independent arithmetic oracle and lifetime checks for direct pointwise ops.
use super::*;
use crate::{capture, transfers, CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};
use half::{bf16, f16};

fn upload(shape: Vec<usize>, values: &[f32], dtype: DType) -> Tensor {
    let host = match dtype {
        DType::F32 => Tensor::from_f32(shape, values).unwrap(),
        DType::F16 => Tensor::from_f16(
            shape,
            &values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        _ => unreachable!(),
    };
    transfers::to_cuda(&host, 0).unwrap()
}

fn rounded(value: f64, dtype: DType) -> f32 {
    match dtype {
        DType::F32 => value as f32,
        DType::F16 => f16::from_f64(value).to_f32(),
        DType::BF16 => bf16::from_f64(value).to_f32(),
        _ => unreachable!(),
    }
}

#[test]
fn pointwise_binary_scalar_matches_independent_f64_oracle() {
    use PointwiseSemantic::*;
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for (rows, cols) in [(3, 7), (2, 5120), (24, 32)] {
            let input: Vec<_> = (0..rows * cols)
                .map(|i| rounded(((i % 101) as f64 - 50.) / 16., dtype))
                .collect();
            let x = upload(vec![rows, cols], &input, dtype);
            for semantic in [
                Add,
                Subtract,
                Multiply,
                BroadcastAdd,
                BroadcastMultiply,
                RowMultiply,
                Scale,
                ScalarAdd,
                ScalarDivide,
            ] {
                let shape = match semantic {
                    BroadcastAdd | BroadcastMultiply => vec![1, cols],
                    RowMultiply => vec![rows, 1],
                    _ => vec![rows, cols],
                };
                let secondary: Vec<_> = (0..shape.iter().product())
                    .map(|i| rounded(((i % 23) as f64 - 11.) / 8., dtype))
                    .collect();
                let y = upload(shape, &secondary, dtype);
                let mut output = upload(vec![rows, cols], &vec![0.; rows * cols], dtype);
                let scalar = -0.375f32;
                let mut args = PointwiseArgs::new(semantic, &x, &mut output);
                if !matches!(semantic, Scale | ScalarAdd | ScalarDivide) {
                    args.secondary = Some(&y);
                }
                args.dt = scalar;
                pointwise(&ctx, args).unwrap();
                ctx.synchronize().unwrap();
                let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
                let expected: Vec<_> = input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let j = match semantic {
                            BroadcastAdd | BroadcastMultiply => i % cols,
                            RowMultiply => i / cols,
                            _ => i,
                        };
                        let y = if matches!(semantic, Scale | ScalarAdd | ScalarDivide) {
                            scalar
                        } else {
                            secondary[j]
                        };
                        let (x, y) = (f64::from(x), f64::from(y));
                        rounded(
                            match semantic {
                                Add | BroadcastAdd | ScalarAdd => x + y,
                                Subtract => x - y,
                                ScalarDivide => x / y,
                                _ => x * y,
                            },
                            dtype,
                        )
                    })
                    .collect();
                for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual, expected,
                        "{dtype:?} {semantic:?} {rows}x{cols} element {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn pointwise_broadcast_rejects_wrong_shape_alias_and_nonfinite_scalar() {
    let ctx = CudaContext::new(0).unwrap();
    let x = upload(vec![2, 3], &[1.; 6], DType::BF16);
    let bad_shape = upload(vec![3, 1], &[1.; 3], DType::BF16);
    let y = upload(vec![1, 3], &[1.; 3], DType::BF16);
    let mut out = upload(vec![2, 3], &[0.; 6], DType::BF16);
    let mut args = PointwiseArgs::new(PointwiseSemantic::BroadcastAdd, &x, &mut out);
    args.secondary = Some(&bad_shape);
    assert!(pointwise(&ctx, args)
        .unwrap_err()
        .to_string()
        .contains("shape"));
    let mut alias = x.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::BroadcastAdd, &x, &mut alias);
    args.secondary = Some(&y);
    assert!(pointwise(&ctx, args)
        .unwrap_err()
        .to_string()
        .contains("overlap"));
    for scalar in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args = PointwiseArgs::new(PointwiseSemantic::Scale, &x, &mut out);
        args.dt = scalar;
        assert!(pointwise(&ctx, args)
            .unwrap_err()
            .to_string()
            .contains("finite"));
    }
}

#[test]
fn pointwise_graph_rebinds_values_and_retains_dropped_inputs() {
    let ctx = CudaContext::new(0).unwrap();
    for cols in [3, 5120] {
        let input: Vec<_> = (0..2 * cols).map(|i| (i % 6 + 1) as f32).collect();
        let bias: Vec<_> = (0..cols).map(|i| [2., -1., 0.5][i % 3]).collect();
        let x = upload(vec![2, cols], &input, DType::BF16);
        let y = upload(vec![1, cols], &bias, DType::BF16);
        let mut output = upload(vec![2, cols], &vec![0.; 2 * cols], DType::BF16);
        let invoke = |out: &mut Tensor| {
            let mut args = PointwiseArgs::new(PointwiseSemantic::BroadcastMultiply, &x, out);
            args.secondary = Some(&y);
            pointwise(&ctx, args)
        };
        invoke(&mut output).unwrap();
        let graph = capture(&ctx, || invoke(&mut output)).unwrap();
        let x_buffer = CudaBuffer::from_tensor(&x).unwrap();
        let data = vec![bf16::from_f32(8.); 2 * cols];
        let bytes: Vec<_> = data
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes())
            .collect();
        ctx.synchronize().unwrap();
        x_buffer.copy_from_host(&bytes).unwrap();
        drop(x_buffer);
        drop(x);
        drop(y);
        let expected: Vec<_> = (0..2 * cols).map(|i| 8. * bias[i % cols]).collect();
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                expected
            );
        }
    }
}

#[test]
fn pointwise_relu_preserves_mainline_fmax_behavior() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let x = upload(
            vec![1, 5],
            &[-1., 2., f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
            dtype,
        );
        let mut out = upload(vec![1, 5], &[0.; 5], dtype);
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &x, &mut out);
        args.activation = PointwiseActivation::Relu;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        let result = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
        assert_eq!(
            [result[0], result[1], result[3], result[4]],
            [0., 2., 0., f32::INFINITY]
        );
        // The existing L3 ReLU uses fmax(x, 0), including its NaN -> 0 rule.
        assert_eq!(result[2], 0.);
    }
}

#[test]
fn pointwise_rounded_gelu_matches_torch_golden() {
    // Curated from precision/generate_pointwise_fixtures.py on NVIDIA Thor,
    // PyTorch 2.11.0+cu130. These expose cancellation/rounding differences from
    // fused GELU (notably the BF16 result at -3.14).
    let source = [
        -3.14, -2.5, -2.0, -1.5, -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0, 1.5, 2.0, 2.5, 3.14,
        4.0, -0.90625, -0.828125, 0.90625, 0.828125,
    ];
    let f16_expected = [
        -0.002300262451171875,
        -0.0152587890625,
        -0.04541015625,
        -0.1007080078125,
        -0.158935546875,
        -0.154296875,
        -0.0562744140625,
        0.0,
        0.0687255859375,
        0.345703125,
        0.8408203125,
        1.3984375,
        1.955078125,
        2.484375,
        3.13671875,
        4.0,
        -0.16552734375,
        -0.1688232421875,
        0.74072265625,
        0.6591796875,
    ];
    let bf16_expected = [
        -0.0,
        -0.0146484375,
        -0.046875,
        -0.1025390625,
        -0.158203125,
        -0.154296875,
        -0.056396484375,
        0.0,
        0.06884765625,
        0.345703125,
        0.84375,
        1.390625,
        1.953125,
        2.484375,
        3.140625,
        4.0,
        -0.166015625,
        -0.16796875,
        0.73828125,
        0.66015625,
    ];
    let f32_expected = [
        -0.0022966230753809214,
        -0.015084296464920044,
        -0.04540228843688965,
        -0.10042843222618103,
        -0.15880799293518066,
        -0.1542859971523285,
        -0.05628279969096184,
        0.0,
        0.06871719658374786,
        0.3457140028476715,
        0.8411920070648193,
        1.3995715379714966,
        1.9545977115631104,
        2.4849157333374023,
        3.1377034187316895,
        3.999929904937744,
        -0.16542117297649384,
        -0.16886474192142487,
        0.740828812122345,
        0.6592602729797363,
    ];
    let ctx = CudaContext::new(0).unwrap();
    for (dtype, expected) in [
        (DType::F16, f16_expected),
        (DType::BF16, bf16_expected),
        (DType::F32, f32_expected),
    ] {
        let input = upload(vec![1, source.len()], &source, dtype);
        let mut output = upload(vec![1, source.len()], &vec![0.; source.len()], dtype);
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
        args.activation = PointwiseActivation::GeluRounded;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
        for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
            if dtype == DType::F32 {
                assert!((a - b).abs() < 3e-7, "F32 element {i}: {a} != {b}");
            } else {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{dtype:?} element {i}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn rounded_norm_matches_independent_reference_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        for width in [3, 128, 4096, 5120, 8193] {
            let source: Vec<_> = (0..2 * width)
                .map(|i| rounded(((i % 37) as f64 - 18.) / 8., dtype))
                .collect();
            let weights: Vec<_> = (0..width)
                .map(|i| rounded(0.5 + (i % 11) as f64 / 16., dtype))
                .collect();
            let x = upload(vec![2, width], &source, dtype);
            let w = upload(vec![width], &weights, dtype);
            let mut out = upload(vec![2, width], &vec![0.; 2 * width], dtype);
            for l2 in [false, true] {
                let eps = if l2 { 1e-4f32 } else { 1e-6f32 };
                let invoke = |out: &mut Tensor| {
                    if l2 {
                        staged_l2(&ctx, &x, out, eps)
                    } else {
                        staged_rms(&ctx, NormArgs::rms(&x, &w, out, eps))
                    }
                };
                invoke(&mut out).unwrap();
                ctx.synchronize().unwrap();
                let eager = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
                for row in 0..2 {
                    let square_sum: f64 = source[row * width..(row + 1) * width]
                        .iter()
                        .map(|&v| f64::from(v).powi(2))
                        .sum();
                    for col in 0..width {
                        let i = row * width + col;
                        let expected = if l2 {
                            rounded(
                                f64::from(source[i])
                                    / f64::from(
                                        rounded(square_sum.sqrt(), dtype)
                                            .max(rounded(f64::from(eps), dtype)),
                                    ),
                                dtype,
                            )
                        } else {
                            let normalized = rounded(
                                f64::from(source[i])
                                    / (square_sum / width as f64 + f64::from(eps)).sqrt(),
                                dtype,
                            );
                            rounded(f64::from(normalized) * f64::from(weights[col]), dtype)
                        };
                        // Independent double-precision reductions can fall on
                        // the other side of one low-precision rounding tie.
                        let unit = if dtype == DType::BF16 {
                            1. / 128.
                        } else {
                            1. / 1024.
                        };
                        assert!(
                            (eager[i] - expected).abs() <= expected.abs() * unit + 1e-7,
                            "{dtype:?}, width {width}, l2 {l2}, element {i}: {} != {expected}",
                            eager[i]
                        );
                    }
                }
                let graph = capture(&ctx, || invoke(&mut out)).unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
                    eager
                );
            }
        }
    }
    let x = upload(vec![1, 3], &[1., 2., 3.], DType::BF16);
    let w = upload(vec![3], &[0.75, 1.25, 1.5], DType::BF16);
    let mut out = upload(vec![1, 3], &[0.; 3], DType::BF16);
    let graph = capture(&ctx, || {
        staged_rms(&ctx, NormArgs::rms(&x, &w, &mut out, 1e-6))
    })
    .unwrap();
    let mut alias = x.clone();
    assert!(staged_l2(&ctx, &x, &mut alias, 1e-4)
        .unwrap_err()
        .to_string()
        .contains("overlap"));
    assert!(staged_l2(&ctx, &x, &mut out, f32::NAN).is_err());
    drop(alias);
    drop(x);
    drop(w);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(transfers::to_cpu(&out)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .all(|v| v.is_finite() && *v > 0.));
}

#[test]
fn pointwise_swish_rounded_matches_torch_tensor_expression() {
    let ctx = CudaContext::new(0).unwrap();
    let source = [
        -3.14, -2.5, -2.0, -1.5, -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0, 1.5, 2.0, 2.5, 3.14, 4.0,
    ];
    for (dtype, expected) in [
        (
            DType::F32,
            [
                -0.13026954233646393,
                -0.1896454393863678,
                -0.23840583860874176,
                -0.2736383080482483,
                -0.2689414322376251,
                -0.1887703388929367,
                -0.058598827570676804,
                0.0,
                0.0664011761546135,
                0.3112296760082245,
                0.7310585379600525,
                1.226361632347107,
                1.7615940570831299,
                2.310354471206665,
                3.009730577468872,
                3.9280550479888916,
            ],
        ),
        (
            DType::F16,
            [
                -0.1302490234375,
                -0.189697265625,
                -0.2384033203125,
                -0.2734375,
                -0.26904296875,
                -0.188720703125,
                -0.05859375,
                0.0,
                0.06640625,
                0.311279296875,
                0.73095703125,
                1.2265625,
                1.76171875,
                2.310546875,
                3.009765625,
                3.927734375,
            ],
        ),
        (
            DType::BF16,
            [
                -0.1298828125,
                -0.189453125,
                -0.23828125,
                -0.2734375,
                -0.26953125,
                -0.1884765625,
                -0.05859375,
                0.0,
                0.06640625,
                0.310546875,
                0.73046875,
                1.2265625,
                1.7578125,
                2.3125,
                3.0,
                3.921875,
            ],
        ),
    ] {
        let input = upload(vec![1, 16], &source, dtype);
        let mut output = upload(vec![1, 16], &[0.; 16], dtype);
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
        args.activation = PointwiseActivation::SwishRounded;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*a - b).abs() <= if dtype == DType::F32 { 1e-6 } else { 0. },
                "{dtype:?} {i}: {a} != {b}"
            );
        }
        let mut args = PointwiseArgs::new(PointwiseSemantic::ScalarDivide, &input, &mut output);
        args.dt = 0.;
        assert!(pointwise(&ctx, args).is_err());
    }
}

#[test]
fn rounded_rms_vector_reduction_matches_torch_materialization() {
    // Every BF16 output is covered by the independent CUDA golden digest from
    // precision/generate_rounded_norm_fixtures.py (Torch 2.11, Thor SM110).
    let ctx = CudaContext::new(0).unwrap();
    let source: Vec<f32> = (0..16 * 4096)
        .map(|i| ((i * 37) % 4093) as f32 / 128. - 2046. / 128.)
        .collect();
    let weights: Vec<f32> = (0..4096)
        .map(|i| ((i * 13) % 127 + 1) as f32 / 128.)
        .collect();
    let x = upload(vec![16, 4096], &source, DType::BF16);
    let w = upload(vec![4096], &weights, DType::BF16);
    let mut output = upload(vec![16, 4096], &vec![0.; source.len()], DType::BF16);
    staged_rms(&ctx, NormArgs::rms(&x, &w, &mut output, 1e-6)).unwrap();
    ctx.synchronize().unwrap();
    let actual = transfers::to_cpu(&output).unwrap();
    let mut digest = 14695981039346656037u64;
    for value in actual.as_bf16().unwrap() {
        for byte in value.to_bits().to_le_bytes() {
            digest = (digest ^ u64::from(byte)).wrapping_mul(1099511628211);
        }
    }
    assert_eq!(digest, 16015136281615547435u64);
    let expected = actual.to_f32_vec().unwrap();
    let graph = crate::capture(&ctx, || {
        staged_rms(&ctx, NormArgs::rms(&x, &w, &mut output, 1e-6))
    })
    .unwrap();
    drop(x);
    drop(w);
    for _ in 0..3 {
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            expected,
            transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap()
        );
    }
}

#[test]
fn l2_f32_preserves_promoted_precision_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    let values: Vec<f32> = (0..3 * 384)
        .map(|i| {
            if i < 384 {
                0.0
            } else {
                ((i * 17 % 29) as f32 - 14.0) / 8.0
            }
        })
        .collect();
    let input = upload(vec![3, 384], &values, DType::F32);
    let mut out = upload(vec![3, 384], &vec![0.0; values.len()], DType::F32);
    // Independent f64 row oracle includes a zero row and clamp behavior.
    let expected: Vec<f32> = values
        .chunks(384)
        .flat_map(|row| {
            let norm = row
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            row.iter().map(move |&x| (f64::from(x) / norm) as f32)
        })
        .collect();
    staged_l2(&ctx, &input, &mut out, 1e-12).unwrap();
    ctx.synchronize().unwrap();
    let eager = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
    for (&actual, &expected) in eager.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-7);
    }
    let graph = capture(&ctx, || staged_l2(&ctx, &input, &mut out, 1e-12)).unwrap();
    drop(input);
    for _ in 0..3 {
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
            eager
        );
    }
}

#[test]
fn pointwise_transcendentals_match_f64_and_replay_after_owner_drop() {
    let ctx = CudaContext::new(0).unwrap();
    let pattern = [-8., -3.125, -0.5, 0., 0.125, 1.25, 8.];
    for count in [7, 10240] {
        let values: Vec<_> = (0..count).map(|i| pattern[i % pattern.len()]).collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for semantic in [
                PointwiseSemantic::Exp,
                PointwiseSemantic::Sin,
                PointwiseSemantic::Cos,
            ] {
                let input = upload(vec![1, values.len()], &values, dtype);
                let mut output = upload(vec![1, values.len()], &vec![0.; values.len()], dtype);
                let check = |output: &Tensor, source: &[f32]| {
                    ctx.synchronize().unwrap();
                    let actual = transfers::to_cpu(output).unwrap().to_f32_vec().unwrap();
                    for (&actual, &x) in actual.iter().zip(source) {
                        let x = f64::from(x);
                        let expected = rounded(
                            match semantic {
                                PointwiseSemantic::Exp => x.exp(),
                                PointwiseSemantic::Sin => x.sin(),
                                PointwiseSemantic::Cos => x.cos(),
                                _ => unreachable!(),
                            },
                            dtype,
                        );
                        let tolerance = if dtype == DType::F32 {
                            2e-6 * expected.abs().max(1.)
                        } else {
                            0.
                        };
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "{semantic:?} {dtype:?}: {actual} != {expected}"
                        );
                    }
                };
                pointwise(&ctx, PointwiseArgs::new(semantic, &input, &mut output)).unwrap();
                check(&output, &values);
                let mut invalid = PointwiseArgs::new(semantic, &input, &mut output);
                invalid.secondary = Some(&input);
                assert!(pointwise(&ctx, invalid).is_err());
                let graph = capture(&ctx, || {
                    pointwise(&ctx, PointwiseArgs::new(semantic, &input, &mut output))
                })
                .unwrap();
                let changed: Vec<_> = values.iter().map(|x| -*x).collect();
                let new_input = upload(vec![1, values.len()], &changed, dtype);
                transfers::copy_cuda_to_cuda(&ctx, &new_input, &input).unwrap();
                drop(new_input);
                drop(input);
                for _ in 0..2 {
                    graph.replay().unwrap();
                    check(&output, &changed);
                }
            }
        }
    }
}

#[test]
fn l2_channel_rows_match_torch_and_changed_input_capture() {
    // Independent NCHW torch.nn.functional.normalize goldens, rearranged into
    // channel rows (Torch 2.11, Thor SM110). The generator is maintained in
    // precision/generate_l2_channel_fixtures.py. Wide exponents expose reduction
    // ordering; the first row exercises the epsilon clamp on an exact zero.
    let ctx = CudaContext::new(0).unwrap();
    for (dtype, width, expected_hash) in [
        (DType::F32, 96, 3494330373509610637u64),
        (DType::F32, 192, 10142548330631730086u64),
        (DType::F32, 384, 18176394244215574920u64),
        (DType::F16, 96, 11912173056842318875u64),
        (DType::F16, 192, 14196645568024147634u64),
        (DType::F16, 384, 3462228948769288594u64),
        (DType::BF16, 96, 10377534575474924866u64),
        (DType::BF16, 192, 1259611236160645793u64),
        (DType::BF16, 384, 2302197685747167972u64),
    ] {
        let source: Vec<f32> = (0..128 * width)
            .map(|i| {
                if i < width {
                    0.
                } else {
                    let sign = if i % 3 == 0 { -1. } else { 1. };
                    sign * (1. + (i * 37 % 127) as f32 / 128.)
                        * 2f32.powi((i * 11 % 23) as i32 - 11)
                }
            })
            .collect();
        let input = upload(vec![128, width], &source, dtype);
        let retained_input = input.clone();
        let mut output = upload(vec![128, width], &vec![0.; source.len()], dtype);
        let eps = if dtype == DType::F16 { 1e-4 } else { 1e-12 };
        staged_l2(&ctx, &input, &mut output, eps).unwrap();
        ctx.synchronize().unwrap();
        let host = transfers::to_cpu(&output).unwrap();
        let mut hash = 14695981039346656037u64;
        for &byte in host.storage().as_cpu().unwrap() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211);
        }
        assert_eq!(hash, expected_hash, "{dtype:?} width={width}");
        let expected = host.to_f32_vec().unwrap();
        let graph = capture(&ctx, || staged_l2(&ctx, &input, &mut output, eps)).unwrap();
        drop(input);
        for sign in [-1., 1.] {
            let changed = upload(
                vec![128, width],
                &source.iter().map(|v| v * sign).collect::<Vec<_>>(),
                dtype,
            );
            transfers::copy_cuda_to_cuda(&ctx, &changed, &retained_input).unwrap();
            ctx.synchronize().unwrap();
            drop(changed);
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
            for (&a, &e) in actual.iter().zip(&expected) {
                assert_eq!(a.to_bits(), (e * sign).to_bits());
            }
        }
        drop(retained_input);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
            expected
        );
    }
}

fn staged_affine_bits(ctx: &CudaContext, x: &Tensor) -> Vec<u32> {
    ctx.synchronize().unwrap();
    let h = transfers::to_cpu(x).unwrap();
    match h.dtype() {
        DType::BF16 => h
            .as_bf16()
            .unwrap()
            .iter()
            .map(|x| u32::from(x.to_bits()))
            .collect(),
        DType::F16 => h
            .as_f16()
            .unwrap()
            .iter()
            .map(|x| u32::from(x.to_bits()))
            .collect(),
        DType::F32 => h
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        _ => unreachable!(),
    }
}

#[test]
fn staged_affine_matches_composition_oracle_aliases_and_retained_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for (rows, cols, raw_bits) in [
            (3, 7, false),
            (1, 128, false),
            (17, 128, false),
            (25, 5120, false),
            (128, 512, true),
        ] {
            if raw_bits && dtype == DType::F32 {
                continue;
            }
            let values: Vec<_> = (0..rows * cols)
                .map(|i| (i % 257) as f32 / 32. - 4.)
                .collect();
            let x = if raw_bits {
                let bytes: Vec<_> = (0..rows * cols)
                    .flat_map(|i| (i as u16).to_ne_bytes())
                    .collect();
                transfers::to_cuda(
                    &Tensor::from_raw(
                        apxinf_core::Shape::new(vec![rows, cols]),
                        dtype,
                        apxinf_core::Device::Cpu,
                        bytes,
                    )
                    .unwrap(),
                    0,
                )
                .unwrap()
            } else {
                upload(vec![rows, cols], &values, dtype)
            };
            let original = transfers::to_cpu(&x).unwrap();
            let scale_values: Vec<_> = (0..cols).map(|i| (i % 73) as f32 / 32. - 1.).collect();
            let column_values: Vec<_> = (0..cols).map(|i| (i % 53) as f32 / 32. - 0.5).collect();
            for gate in [false, true] {
                let secondary = if gate {
                    upload(
                        vec![rows, cols],
                        &values.iter().rev().copied().collect::<Vec<_>>(),
                        dtype,
                    )
                } else {
                    upload(vec![1, cols], &scale_values, dtype)
                };
                let column = upload(vec![cols], &column_values, dtype);
                let secondary_original = transfers::to_cpu(&secondary).unwrap();
                let mut expected = ctx
                    .allocate_output(apxinf_core::Shape::new(vec![rows, cols]), dtype)
                    .unwrap();
                let mut actual = ctx
                    .allocate_output(apxinf_core::Shape::new(vec![rows, cols]), dtype)
                    .unwrap();
                let mut scratch = ctx
                    .allocate_output(apxinf_core::Shape::new(vec![rows, cols]), dtype)
                    .unwrap();
                let mut unit_scale = ctx
                    .allocate_output(apxinf_core::Shape::new(vec![1, cols]), dtype)
                    .unwrap();
                let mut compose = |out: &mut Tensor| {
                    if gate {
                        let mut a = PointwiseArgs::new(
                            PointwiseSemantic::BroadcastMultiply,
                            &x,
                            &mut scratch,
                        );
                        let c = column.reshape(vec![1, cols]).unwrap();
                        a.secondary = Some(&c);
                        pointwise(&ctx, a).unwrap();
                        let mut a = PointwiseArgs::new(PointwiseSemantic::Add, &secondary, out);
                        a.secondary = Some(&scratch);
                        pointwise(&ctx, a).unwrap();
                    } else {
                        let mut a = PointwiseArgs::new(
                            PointwiseSemantic::ScalarAdd,
                            &secondary,
                            &mut unit_scale,
                        );
                        a.dt = 1.;
                        pointwise(&ctx, a).unwrap();
                        let mut a = PointwiseArgs::new(
                            PointwiseSemantic::BroadcastMultiply,
                            &x,
                            &mut scratch,
                        );
                        a.secondary = Some(&unit_scale);
                        pointwise(&ctx, a).unwrap();
                        let mut a =
                            PointwiseArgs::new(PointwiseSemantic::BroadcastAdd, &scratch, out);
                        let c = column.reshape(vec![1, cols]).unwrap();
                        a.secondary = Some(&c);
                        pointwise(&ctx, a).unwrap();
                    }
                };
                let semantic = if gate {
                    PointwiseSemantic::StagedGatedResidual
                } else {
                    PointwiseSemantic::StagedModulation
                };
                let fused = |out: &mut Tensor| {
                    let mut a = PointwiseArgs::new(semantic, &x, out);
                    a.secondary = Some(&secondary);
                    a.bias = Some(&column);
                    pointwise(&ctx, a)
                };
                compose(&mut expected);
                fused(&mut actual).unwrap();
                assert_eq!(
                    staged_affine_bits(&ctx, &actual),
                    staged_affine_bits(&ctx, &expected)
                );
                // An independent F64 oracle explicitly rounds each F32 elementary
                // operation, then the dtype materialization, for finite inputs.
                if !raw_bits {
                    let xx = transfers::to_cpu(&x).unwrap().to_f32_vec().unwrap();
                    let yy = secondary_original.to_f32_vec().unwrap();
                    let cc = transfers::to_cpu(&column).unwrap().to_f32_vec().unwrap();
                    let reference: Vec<_> = xx
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let c = i % cols;
                            let factor = if gate {
                                cc[c]
                            } else {
                                rounded(f64::from((f64::from(yy[c]) + 1.) as f32), dtype)
                            };
                            let product = rounded(
                                f64::from((f64::from(v) * f64::from(factor)) as f32),
                                dtype,
                            );
                            rounded(
                                f64::from(
                                    (if gate {
                                        f64::from(yy[i]) + f64::from(product)
                                    } else {
                                        f64::from(product) + f64::from(cc[c])
                                    }) as f32,
                                ),
                                dtype,
                            )
                        })
                        .collect();
                    let oracle = upload(vec![rows, cols], &reference, dtype);
                    assert_eq!(
                        staged_affine_bits(&ctx, &actual),
                        staged_affine_bits(&ctx, &oracle)
                    );
                }
                let mut alias = x.clone();
                fused(&mut alias).unwrap();
                assert_eq!(
                    staged_affine_bits(&ctx, &alias),
                    staged_affine_bits(&ctx, &expected)
                );
                transfers::copy_cpu_to_cuda(&original, &x).unwrap();
                if gate {
                    let mut alias = secondary.clone();
                    fused(&mut alias).unwrap();
                    assert_eq!(
                        staged_affine_bits(&ctx, &alias),
                        staged_affine_bits(&ctx, &expected)
                    );
                    transfers::copy_cpu_to_cuda(&secondary_original, &secondary).unwrap();
                }
                let graph = capture(&ctx, || fused(&mut actual)).unwrap();
                let changed = upload(
                    vec![rows, cols],
                    &values.iter().map(|x| -x).collect::<Vec<_>>(),
                    dtype,
                );
                let changed_column = upload(
                    vec![cols],
                    &column_values.iter().map(|x| x * 0.5).collect::<Vec<_>>(),
                    dtype,
                );
                transfers::copy_cuda_to_cuda(&ctx, &changed, &x).unwrap();
                transfers::copy_cuda_to_cuda(&ctx, &changed_column, &column).unwrap();
                let changed_secondary = upload(
                    secondary.shape().dims().to_vec(),
                    &secondary_original
                        .to_f32_vec()
                        .unwrap()
                        .iter()
                        .map(|x| -x * 0.5)
                        .collect::<Vec<_>>(),
                    dtype,
                );
                transfers::copy_cuda_to_cuda(&ctx, &changed_secondary, &secondary).unwrap();
                compose(&mut expected);
                let bits = staged_affine_bits(&ctx, &expected);
                drop(secondary);
                drop(column);
                drop(changed);
                drop(changed_column);
                drop(changed_secondary);
                for _ in 0..3 {
                    graph.replay().unwrap();
                    assert_eq!(staged_affine_bits(&ctx, &actual), bits);
                }
                transfers::copy_cpu_to_cuda(&original, &x).unwrap();
            }
        }
    }
}

#[test]
fn staged_affine_rejects_missing_broadcast_and_partial_alias_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    let x = upload(vec![3, 7], &[1.; 21], DType::BF16);
    let scale = upload(vec![1, 7], &[1.; 7], DType::BF16);
    let residual = upload(vec![3, 7], &[1.; 21], DType::BF16);
    let col = upload(vec![7], &[1.; 7], DType::BF16);
    let mut out = upload(vec![3, 7], &[0.; 21], DType::BF16);
    for semantic in [
        PointwiseSemantic::StagedModulation,
        PointwiseSemantic::StagedGatedResidual,
    ] {
        let secondary = if semantic == PointwiseSemantic::StagedModulation {
            &scale
        } else {
            &residual
        };
        let mut a = PointwiseArgs::new(semantic, &x, &mut out);
        a.secondary = Some(secondary);
        assert!(pointwise(&ctx, a).is_err());
        let mut a = PointwiseArgs::new(semantic, &x, &mut out);
        a.bias = Some(&col);
        assert!(pointwise(&ctx, a).is_err());
        let mut a = PointwiseArgs::new(semantic, &x, &mut out);
        a.secondary = Some(if semantic == PointwiseSemantic::StagedModulation {
            &residual
        } else {
            &scale
        });
        a.bias = Some(&col);
        assert!(pointwise(&ctx, a).is_err());
        let wrong = col.reshape(vec![1, 7]).unwrap();
        let mut a = PointwiseArgs::new(semantic, &x, &mut out);
        a.secondary = Some(secondary);
        a.bias = Some(&wrong);
        assert!(pointwise(&ctx, a).is_err());
        let backing = CudaBuffer::alloc(44, 0).unwrap();
        let input = backing
            .view(0, 42)
            .unwrap()
            .as_tensor(apxinf_core::Shape::new(vec![3, 7]), DType::BF16)
            .unwrap();
        let mut overlap = backing
            .view(2, 42)
            .unwrap()
            .as_tensor(apxinf_core::Shape::new(vec![3, 7]), DType::BF16)
            .unwrap();
        let mut a = PointwiseArgs::new(semantic, &input, &mut overlap);
        a.secondary = Some(secondary);
        a.bias = Some(&col);
        assert!(pointwise(&ctx, a).is_err());
        let one = x.reshape(vec![1, 21]).unwrap();
        let broadcast = one.reshape(vec![21]).unwrap();
        let mut alias = one.clone();
        let mut a = PointwiseArgs::new(semantic, &one, &mut alias);
        a.secondary = Some(&one);
        a.bias = Some(&broadcast);
        assert!(pointwise(&ctx, a).is_err());
    }
}
