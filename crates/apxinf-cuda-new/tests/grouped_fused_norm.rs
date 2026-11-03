use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::ResidualGate;
use apxinf_cuda_new::{
    capture, fused_add_norm, norm, residual_add, Affine, BiasOrder, CudaBuffer, CudaContext,
    ExecutionSession, FusedAddNormArgs, NormArgs, NormKind, NormNumerics, ResidualAddArgs,
};
use half::bf16;
fn tensor(shape: Vec<usize>, values: &[f32], offset: usize) -> Tensor {
    let b = CudaBuffer::alloc(values.len() * 2 + offset, 0)
        .unwrap()
        .view(offset, values.len() * 2)
        .unwrap();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
        .collect();
    b.copy_from_host(&bytes).unwrap();
    b.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}
fn bits(t: &Tensor) -> Vec<u8> {
    let mut v = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut v)
        .unwrap();
    v
}
fn args<'a>(
    x: &'a Tensor,
    r: &'a Tensor,
    g: &'a Tensor,
    s: &'a Tensor,
    h: &'a Tensor,
    y: &'a Tensor,
    shift: bool,
) -> FusedAddNormArgs<'a> {
    FusedAddNormArgs {
        input: x,
        residual: r,
        bias: None,
        gate: Some(ResidualGate::GroupedStyle {
            style: g,
            numerics: NormNumerics::StagedBf16,
        }),
        hidden: h,
        normalized: Some(y),
        kind: NormKind::Layer,
        affine: Affine::GroupedAdaptive {
            style: s,
            shift_first: shift,
        },
        quantized: None,
        numerics: NormNumerics::StagedBf16,
        bias_order: BiasOrder::SingleRound,
        eps: 1e-6,
    }
}
fn reference(
    ctx: &CudaContext,
    x: &Tensor,
    r: &Tensor,
    g: &Tensor,
    s: &Tensor,
    h: &Tensor,
    y: &Tensor,
    shift: bool,
) {
    residual_add(
        ctx,
        ResidualAddArgs {
            input: x,
            bias: None,
            residual: r,
            gate: Some(ResidualGate::GroupedStyle {
                style: g,
                numerics: NormNumerics::StagedBf16,
            }),
            hidden: h,
            bias_order: BiasOrder::SingleRound,
        },
    )
    .unwrap();
    norm(
        ctx,
        NormArgs {
            input: h,
            output: Some(y),
            kind: NormKind::Layer,
            affine: Affine::GroupedAdaptive {
                style: s,
                shift_first: shift,
            },
            quantized: None,
            numerics: NormNumerics::StagedBf16,
            eps: 1e-6,
        },
    )
    .unwrap();
}
#[test]
fn grouped_fused_norm_matches_staged_chain_and_closed_form() {
    let ctx = CudaContext::new(0).unwrap();
    for (rows, cols, groups, offset) in [
        (6, 7, 3, 2),
        (6, 16, 3, 0),
        (6, 2048, 3, 2),
        (6, 2048, 3, 4),
        (1, 2048, 1, 0),
        (7, 2048, 7, 0),
        (14, 2048, 2, 0),
        (1764, 2048, 9, 0),
    ] {
        for parts in [2, 3] {
            for shift in [false, true] {
                let values: Vec<_> = (0..rows * cols)
                    .map(|i| (i as i32 % 17 - 8) as f32 / 8.)
                    .collect();
                let residual: Vec<_> = (0..rows * cols)
                    .map(|i| (i as i32 % 13 - 6) as f32 / 16.)
                    .collect();
                let gate: Vec<_> = (0..groups * 3 * cols)
                    .map(|i| (i as i32 % 11 - 5) as f32 / 4.)
                    .collect();
                let style: Vec<_> = (0..groups * parts * cols)
                    .map(|i| (i as i32 % 7 - 3) as f32 / 8.)
                    .collect();
                let x = tensor(vec![rows, cols], &values, offset);
                let r = tensor(vec![rows, cols], &residual, offset);
                let g = tensor(vec![groups, 3, cols], &gate, offset);
                let s = tensor(vec![groups, parts, cols], &style, offset);
                let h = tensor(vec![rows, cols], &vec![0.; rows * cols], offset);
                let y = tensor(vec![rows, cols], &vec![0.; rows * cols], offset);
                reference(&ctx, &x, &r, &g, &s, &h, &y, shift);
                ctx.synchronize().unwrap();
                let expected_h = bits(&h);
                let expected_y = bits(&y);
                // The hidden result overwrites the residual, as in a transformer block.
                fused_add_norm(&ctx, args(&x, &r, &g, &s, &r, &h, shift)).unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(bits(&r), expected_h);
                assert_eq!(bits(&h), expected_y);
            }
        }
    }
    // Independent closed form: equally many +/- values yield mean0; normalized
    // values round to +/-1. Group-dependent gates and affine values are distinct.
    for (rows, cols, offset) in [(6, 16, 2), (6, 2048, 0)] {
        for parts in [2, 3] {
            for shift in [false, true] {
                let x = tensor(
                    vec![rows, cols],
                    &(0..rows * cols)
                        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                        .collect::<Vec<_>>(),
                    offset,
                );
                let mut gate = vec![0.; 3 * 3 * cols];
                let mut style = vec![0.; 3 * parts * cols];
                let mut eh = vec![0.; rows * cols];
                let mut ey = eh.clone();
                for group in 0..3 {
                    for c in 0..cols {
                        gate[(group * 3 + 2) * cols + c] = (group + 1) as f32;
                        style[(group * parts + usize::from(shift)) * cols + c] = group as f32 * 0.5;
                        style[(group * parts + usize::from(!shift)) * cols + c] =
                            group as f32 * 0.25;
                    }
                    for row in group * 2..group * 2 + 2 {
                        for c in 0..cols {
                            let sign = if c % 2 == 0 { 1. } else { -1. };
                            eh[row * cols + c] = sign * ((group + 2) as f32 * 0.5);
                            ey[row * cols + c] =
                                sign * (1. + group as f32 * 0.5) + group as f32 * 0.25;
                        }
                    }
                }
                let g = tensor(vec![3, 3, cols], &gate, offset);
                let s = tensor(vec![3, parts, cols], &style, offset);
                let h = tensor(vec![rows, cols], &vec![0.; rows * cols], offset);
                let y = tensor(vec![rows, cols], &vec![0.; rows * cols], offset);
                fused_add_norm(&ctx, args(&x, &x, &g, &s, &h, &y, shift)).unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(bits(&h), bits(&tensor(vec![rows, cols], &eh, 0)));
                assert_eq!(bits(&y), bits(&tensor(vec![rows, cols], &ey, 0)));
            }
        }
    }
}
#[test]
fn grouped_fused_norm_graph_retention_and_contracts() {
    let ctx = CudaContext::new(0).unwrap();
    let cols = 2048;
    let x = tensor(vec![6, cols], &vec![0.5; 6 * cols], 0);
    let r = tensor(vec![6, cols], &vec![0.25; 6 * cols], 0);
    let g = tensor(vec![3, 3, cols], &vec![1.; 9 * cols], 0);
    let s = tensor(vec![3, 2, cols], &vec![0.25; 6 * cols], 0);
    let h = tensor(vec![6, cols], &vec![0.; 6 * cols], 0);
    let y = tensor(vec![6, cols], &vec![0.; 6 * cols], 0);
    assert!(fused_add_norm(&ctx, args(&x, &r, &g, &s, &h, &h, true)).is_err());
    assert!(fused_add_norm(&ctx, args(&x, &r, &g, &s, &s, &y, true)).is_err());
    let wrong = tensor(vec![1, 3, cols], &vec![1.; 3 * cols], 0);
    assert!(fused_add_norm(&ctx, args(&x, &r, &wrong, &s, &h, &y, true)).is_err());
    let mut bad = args(&x, &r, &g, &s, &h, &y, true);
    bad.numerics = NormNumerics::Flexible;
    assert!(fused_add_norm(&ctx, bad).is_err());
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    session
        .prepare(|| fused_add_norm(&ctx, args(&x, &r, &g, &s, &h, &y, true)))
        .unwrap();
    let graph = capture(&ctx, || {
        session.run(|| fused_add_norm(&ctx, args(&x, &r, &g, &s, &h, &y, true)))
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    // Mutate both styles after capture; neither binding is folded into constants.
    let newg = tensor(
        vec![3, 3, cols],
        &(0..9 * cols)
            .map(|i| (i % 5) as f32 / 4.)
            .collect::<Vec<_>>(),
        0,
    );
    let news = tensor(
        vec![3, 2, cols],
        &(0..6 * cols)
            .map(|i| (i % 7) as f32 / 8.)
            .collect::<Vec<_>>(),
        0,
    );
    apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &newg, &g).unwrap();
    apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &news, &s).unwrap();
    reference(&ctx, &x, &r, &g, &s, &h, &y, true);
    ctx.synchronize().unwrap();
    let eh = bits(&h);
    let ey = bits(&y);
    drop(session);
    drop(x);
    drop(r);
    drop(g);
    drop(s);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(bits(&h), eh);
    assert_eq!(bits(&y), ey);
}

#[test]
fn grouped_fused_norm_preserves_bf16_rounding_edges() {
    let ctx = CudaContext::new(0).unwrap();
    let (rows, cols, groups) = (6, 2048, 3);
    let patterns = [
        0u16, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
        0x3f81, 0xbf81, 0x3eff, 0xbeff,
    ];
    let values = |count: usize, offset: usize| -> Vec<f32> {
        (0..count)
            .map(|i| bf16::from_bits(patterns[(i + offset) % patterns.len()]).to_f32())
            .collect()
    };
    for parts in [2, 3] {
        for shift in [false, true] {
            let x = tensor(vec![rows, cols], &values(rows * cols, 0), 0);
            let r = tensor(vec![rows, cols], &values(rows * cols, 5), 0);
            let g = tensor(vec![groups, 3, cols], &values(groups * 3 * cols, 9), 0);
            let s = tensor(
                vec![groups, parts, cols],
                &values(groups * parts * cols, 13),
                0,
            );
            let h = tensor(vec![rows, cols], &vec![0.; rows * cols], 0);
            let y = tensor(vec![rows, cols], &vec![0.; rows * cols], 0);
            reference(&ctx, &x, &r, &g, &s, &h, &y, shift);
            ctx.synchronize().unwrap();
            let expected_h = bits(&h);
            let expected_y = bits(&y);
            fused_add_norm(&ctx, args(&x, &r, &g, &s, &r, &h, shift)).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&r), expected_h, "staged residual rounding");
            assert_eq!(bits(&h), expected_y, "staged affine rounding");
        }
    }
}
