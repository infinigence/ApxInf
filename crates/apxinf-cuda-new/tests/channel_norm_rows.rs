use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{capture, channel_l2_norm, ChannelL2NormArgs, CudaBuffer, CudaContext};
use half::bf16;
fn tensor(shape: Vec<usize>, v: &[f32]) -> Tensor {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
        .collect();
    let b = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    b.copy_from_host(&bytes).unwrap();
    b.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}
fn bits(t: &Tensor) -> Vec<u16> {
    let mut b = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut b)
        .unwrap();
    b.chunks_exact(2)
        .map(|p| u16::from_ne_bytes(p.try_into().unwrap()))
        .collect()
}
#[test]
fn channel_norm_row_threshold_preserves_staged_rounding_and_in_place_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let round = |x| bf16::from_f32(x).to_f32();
    for cols in [96, 192, 384] {
        for rows in [1023, 1024, 1025, 1568] {
            for eps in [1e-12, 0.5] {
                let values: Vec<f32> = (0..rows * cols)
                    .map(|i| {
                        if i < cols {
                            0.
                        } else if i < 2 * cols {
                            1. / 1048576.
                        } else {
                            ((i * 17 % 49) as f32 - 24.) / 8.
                        }
                    })
                    .collect();
                let weights: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 / 8. - 0.5).collect();
                let x = tensor(vec![rows, cols], &values);
                let weight = tensor(vec![cols], &weights);
                let mut out = tensor(vec![rows, cols], &vec![0.; values.len()]);
                let mut alias = tensor(vec![rows, cols], &values);
                let alias_input = alias.clone();
                channel_l2_norm(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &x,
                        weight: &weight,
                        out: &mut out,
                        eps,
                    },
                )
                .unwrap();
                let graph = capture(&ctx, || {
                    channel_l2_norm(
                        &ctx,
                        ChannelL2NormArgs {
                            input: &alias_input,
                            weight: &weight,
                            out: &mut alias,
                            eps,
                        },
                    )
                })
                .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let got = bits(&out);
                assert_eq!(got, bits(&alias));
                for (r, row) in values.chunks_exact(cols).enumerate() {
                    let norm = round(row.iter().map(|v| v * v).sum::<f32>().sqrt());
                    let denom = round(norm.max(eps));
                    let scale = (cols as f32).sqrt();
                    for c in 0..cols {
                        let expected =
                            bf16::from_f32(round(round(row[c] / denom) * scale) * weights[c])
                                .to_bits();
                        assert_eq!(
                            got[r * cols + c],
                            expected,
                            "rows={rows},cols={cols},r={r},c={c}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn channel_l2_silu_preserves_bf16_boundary_and_in_place_graph() {
    use apxinf_cuda_new::{
        channel_l2_norm_silu, pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
    };
    let ctx = CudaContext::new(0).unwrap();
    for cols in [3, 96, 192, 384] {
        for rows in [1, 1023, 1024, 1025] {
            for eps in [1e-12, 0.5] {
                let values: Vec<_> = (0..rows * cols)
                    .map(|i| {
                        if i < cols {
                            0.
                        } else if i < 2 * cols {
                            1. / 1048576.
                        } else {
                            ((i * 37 % 1021) as f32 - 510.) / 128.
                        }
                    })
                    .collect();
                let weights: Vec<_> = (0..cols).map(|i| (i % 7) as f32 / 8. - 0.5).collect();
                let x = tensor(vec![rows, cols], &values);
                let w = tensor(vec![cols], &weights);
                let mut expected = tensor(vec![rows, cols], &vec![0.; values.len()]);
                channel_l2_norm(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &x,
                        weight: &w,
                        out: &mut expected,
                        eps,
                    },
                )
                .unwrap();
                ctx.synchronize().unwrap();
                let before_silu = bits(&expected);
                let normalized = expected.clone();
                let mut args = PointwiseArgs::new(
                    PointwiseSemantic::BiasActivation,
                    &normalized,
                    &mut expected,
                );
                args.activation = PointwiseActivation::Silu;
                pointwise(&ctx, args).unwrap();
                let mut out = tensor(vec![rows, cols], &vec![0.; values.len()]);
                channel_l2_norm_silu(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &x,
                        weight: &w,
                        out: &mut out,
                        eps,
                    },
                )
                .unwrap();
                let mut alias = tensor(vec![rows, cols], &values);
                let alias_input = alias.clone();
                let graph = capture(&ctx, || {
                    channel_l2_norm_silu(
                        &ctx,
                        ChannelL2NormArgs {
                            input: &alias_input,
                            weight: &w,
                            out: &mut alias,
                            eps,
                        },
                    )
                })
                .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    bits(&out),
                    bits(&expected),
                    "rows={rows},cols={cols},eps={eps}"
                );
                assert_eq!(bits(&alias), bits(&expected));
                for (actual, value) in bits(&out).into_iter().zip(before_silu) {
                    let value = bf16::from_bits(value).to_f32() as f64;
                    let reference = value / (1. + (-value).exp());
                    let actual = bf16::from_bits(actual).to_f32() as f64;
                    assert!((actual - reference).abs() <= 0.004 * reference.abs() + 1e-7, "rows={rows},cols={cols},eps={eps},value={value},actual={actual},reference={reference}");
                }
                assert!(channel_l2_norm_silu(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &x,
                        weight: &w,
                        out: &mut out,
                        eps: -1.
                    }
                )
                .is_err());
            }
        }
    }
}

#[test]
fn channel_l2_silu_packed_alignment_tail_and_bf16_edges() {
    use apxinf_cuda_new::{
        channel_l2_norm_silu, pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
    };
    let ctx = CudaContext::new(0).unwrap();
    let view = |shape: Vec<usize>, values: &[f32], offset: usize| {
        let buffer = CudaBuffer::alloc(values.len() * 2 + offset, 0)
            .unwrap()
            .view(offset, values.len() * 2)
            .unwrap();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect();
        buffer.copy_from_host(&bytes).unwrap();
        buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
    };
    let patterns = [
        0u16, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0x3f81, 0xbf80, 0x3f7f,
    ];
    for cols in [96, 192] {
        for (xo, wo, yo) in [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (4, 8, 14)] {
            let rows = 1025;
            let values: Vec<_> = (0..rows * cols)
                .map(|i| bf16::from_bits(patterns[i % patterns.len()]).to_f32())
                .collect();
            let weights: Vec<_> = (0..cols).map(|i| ((i % 7) as f32 - 3.) / 4.).collect();
            let x = view(vec![rows, cols], &values, xo);
            let w = view(vec![cols], &weights, wo);
            let mut expected = tensor(vec![rows, cols], &vec![0.; values.len()]);
            channel_l2_norm(
                &ctx,
                ChannelL2NormArgs {
                    input: &x,
                    weight: &w,
                    out: &mut expected,
                    eps: 1e-12,
                },
            )
            .unwrap();
            let normalized = expected.clone();
            let mut args = PointwiseArgs::new(
                PointwiseSemantic::BiasActivation,
                &normalized,
                &mut expected,
            );
            args.activation = PointwiseActivation::Silu;
            pointwise(&ctx, args).unwrap();
            let mut out = view(vec![rows, cols], &vec![0.; values.len()], yo);
            let mut alias = view(vec![rows, cols], &values, xo);
            let alias_input = alias.clone();
            let graph = capture(&ctx, || {
                channel_l2_norm_silu(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &x,
                        weight: &w,
                        out: &mut out,
                        eps: 1e-12,
                    },
                )?;
                channel_l2_norm_silu(
                    &ctx,
                    ChannelL2NormArgs {
                        input: &alias_input,
                        weight: &w,
                        out: &mut alias,
                        eps: 1e-12,
                    },
                )
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&out), bits(&expected), "offsets={xo}/{wo}/{yo}");
            assert_eq!(
                bits(&alias),
                bits(&expected),
                "inplace offsets={xo}/{wo}/{yo}"
            );
        }
    }
}
