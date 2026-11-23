use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture, channel_l2_norm_silu, fused_add_channel_l2_norm_silu, pointwise, residual_add,
    BiasOrder, ChannelL2NormArgs, CudaBuffer, CudaContext, FusedAddChannelL2NormArgs,
    PointwiseArgs, PointwiseSemantic, ResidualAddArgs,
};
use half::bf16;
fn upload(t: &Tensor, words: &[u16]) {
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_from_host(
            &words
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
}
fn tensor(shape: Vec<usize>, words: &[u16], offset: usize) -> Tensor {
    let b = CudaBuffer::alloc(words.len() * 2 + offset, 0)
        .unwrap()
        .view(offset, words.len() * 2)
        .unwrap();
    let t = b.as_tensor(Shape::new(shape), DType::BF16).unwrap();
    upload(&t, words);
    t
}
fn bits(t: &Tensor) -> Vec<u16> {
    let mut b = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut b)
        .unwrap();
    b.chunks_exact(2)
        .map(|v| u16::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}
fn check(
    ctx: &CudaContext,
    cols: usize,
    rows: usize,
    eps: f32,
    mode: usize,
    alias: usize,
    offsets: [usize; 6],
) {
    let n = cols * rows;
    let x = tensor(vec![rows, cols], &vec![0; n], offsets[0]);
    let r = tensor(vec![rows, cols], &vec![0; n], offsets[1]);
    let b = tensor(vec![cols], &vec![0; cols], offsets[2]);
    let w = tensor(vec![cols], &vec![0; cols], offsets[3]);
    let h = match alias {
        1 => x.clone(),
        2 => r.clone(),
        _ => tensor(vec![rows, cols], &vec![0; n], offsets[4]),
    };
    let mut y = match alias {
        3 => x.clone(),
        4 => r.clone(),
        _ => tensor(vec![rows, cols], &vec![0; n], offsets[5]),
    };
    let mut rh = tensor(vec![rows, cols], &vec![0; n], 0);
    let mut ry = tensor(vec![rows, cols], &vec![0; n], 0);
    let graph = capture(ctx, || {
        fused_add_channel_l2_norm_silu(
            ctx,
            FusedAddChannelL2NormArgs {
                input: &x,
                bias: (mode & 1 != 0).then_some(&b),
                residual: (mode & 2 != 0).then_some(&r),
                hidden: &h,
                weight: &w,
                normalized: &mut y,
                eps,
            },
        )
    })
    .unwrap();
    let edges = [
        0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0x3f81, 0xbf80, 0x3f7f,
    ];
    // Mutate every semantic operand, then replay the same graph. Caller retains
    // all tensors; capture alone does not promise ownership of their storage.
    for seed in 0..2 {
        let xdata: Vec<_> = (0..n)
            .map(|i| {
                if i < cols {
                    edges[(i + seed) % edges.len()]
                } else {
                    bf16::from_f32(((i * 37 + seed * 19) % 1021) as f32 / 128. - 4.).to_bits()
                }
            })
            .collect();
        let rdata: Vec<_> = (0..n)
            .map(|i| bf16::from_f32(((i + seed) % 13) as f32 / 32. - 0.25).to_bits())
            .collect();
        let bdata: Vec<_> = (0..cols)
            .map(|i| bf16::from_f32(((i + seed) % 9) as f32 / 64. - 0.0625).to_bits())
            .collect();
        let wdata: Vec<_> = (0..cols)
            .map(|i| bf16::from_f32(((i + seed) % 7) as f32 / 8. - 0.5).to_bits())
            .collect();
        upload(&x, &xdata);
        upload(&r, &rdata);
        upload(&b, &bdata);
        upload(&w, &wdata);
        if mode & 2 != 0 {
            residual_add(
                ctx,
                ResidualAddArgs {
                    input: &x,
                    bias: (mode & 1 != 0).then_some(&b),
                    residual: &r,
                    gate: None,
                    hidden: &rh,
                    bias_order: BiasOrder::BiasThenResidual,
                },
            )
            .unwrap();
        } else if mode & 1 != 0 {
            let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &x, &mut rh);
            args.bias = (mode & 1 != 0).then_some(&b);
            pointwise(ctx, args).unwrap();
        } else {
            upload(&rh, &xdata);
        }
        channel_l2_norm_silu(
            ctx,
            ChannelL2NormArgs {
                input: &rh,
                weight: &w,
                out: &mut ry,
                eps,
            },
        )
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(bits(&h),bits(&rh),"hidden cols={cols},rows={rows},mode={mode},alias={alias},offsets={offsets:?},seed={seed}");
        assert_eq!(bits(&y),bits(&ry),"normalized cols={cols},rows={rows},mode={mode},alias={alias},offsets={offsets:?},seed={seed}");
    }
}
#[test]
fn channel_fused_update_matches_separate_ops_and_mutable_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for cols in [3, 96, 192, 384] {
        for rows in [1, 1023, 1024, 1025] {
            for eps in [1e-12, 0.5] {
                for mode in 0..4 {
                    for alias in 0..5 {
                        check(&ctx, cols, rows, eps, mode, alias, [0; 6]);
                    }
                }
            }
        }
    }
}
#[test]
fn channel_fused_update_each_binding_misalignment() {
    let ctx = CudaContext::new(0).unwrap();
    for cols in [96, 192] {
        for binding in 0..6 {
            let mut offsets = [0; 6];
            offsets[binding] = 2;
            for mode in 0..4 {
                check(&ctx, cols, 1025, 1e-12, mode, 0, offsets);
            }
        }
    }
}
#[test]
fn channel_fused_update_rejects_invalid_contracts() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(vec![2, 96], &vec![0; 192], 0);
    let r = x.clone();
    let b = tensor(vec![96], &vec![0; 96], 0);
    let w = b.clone();
    let h = tensor(vec![2, 96], &vec![0; 192], 0);
    let mut y = h.clone();
    let call = |input: &Tensor,
                bias: Option<&Tensor>,
                residual: Option<&Tensor>,
                hidden: &Tensor,
                weight: &Tensor,
                output: &mut Tensor,
                eps| {
        fused_add_channel_l2_norm_silu(
            &ctx,
            FusedAddChannelL2NormArgs {
                input,
                bias,
                residual,
                hidden,
                weight,
                normalized: output,
                eps,
            },
        )
    };
    assert!(call(&x, Some(&b), Some(&r), &h, &w, &mut y, 1e-12).is_err());
    y = tensor(vec![2, 96], &vec![0; 192], 0);
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(call(&x, Some(&b), Some(&r), &h, &w, &mut y, eps).is_err());
    }
    let wrong = tensor(vec![95], &vec![0; 95], 0);
    assert!(call(&x, Some(&wrong), Some(&r), &h, &w, &mut y, 1e-12).is_err());
    assert!(call(&x, Some(&b), Some(&wrong), &h, &w, &mut y, 1e-12).is_err());
    assert!(call(&x, Some(&b), Some(&r), &h, &wrong, &mut y, 1e-12).is_err());
    let f32_input = CudaBuffer::alloc(192 * 4, 0)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 96]), DType::F32)
        .unwrap();
    assert!(call(&f32_input, Some(&b), Some(&r), &h, &w, &mut y, 1e-12).is_err());
    let backing = CudaBuffer::alloc(386, 0).unwrap();
    let a = backing
        .view(0, 384)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 96]), DType::BF16)
        .unwrap();
    let partial = backing
        .view(2, 384)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 96]), DType::BF16)
        .unwrap();
    assert!(call(&a, Some(&b), Some(&r), &partial, &w, &mut y, 1e-12).is_err());
    assert!(call(&x, Some(&b), Some(&a), &partial, &w, &mut y, 1e-12).is_err());
    let mut partial_out = partial.clone();
    assert!(call(&a, Some(&b), Some(&r), &h, &w, &mut partial_out, 1e-12).is_err());
    assert!(call(&x, Some(&b), Some(&r), &a, &w, &mut partial_out, 1e-12).is_err());
    let vector = backing
        .view(0, 192)
        .unwrap()
        .as_tensor(Shape::new(vec![96]), DType::BF16)
        .unwrap();
    assert!(call(&x, Some(&vector), Some(&r), &a, &w, &mut y, 1e-12).is_err());
    assert!(call(&x, Some(&b), Some(&r), &a, &vector, &mut y, 1e-12).is_err());
}
