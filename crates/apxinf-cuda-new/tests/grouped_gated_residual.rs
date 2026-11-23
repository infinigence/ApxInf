use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture, ops::ResidualGate, residual_add, BiasOrder, CudaBuffer, CudaContext, NormNumerics,
    ResidualAddArgs,
};
use half::bf16;
fn tensor(shape: Vec<usize>, values: &[f32], offset: usize) -> Tensor {
    let buffer = CudaBuffer::alloc(values.len() * 2 + offset, 0)
        .unwrap()
        .view(offset, values.len() * 2)
        .unwrap();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
        .collect();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}
fn bits(t: &Tensor) -> Vec<u16> {
    let mut bytes = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn run(
    ctx: &CudaContext,
    x: &Tensor,
    r: &Tensor,
    g: &Tensor,
    o: &Tensor,
) -> apxinf_core::Result<()> {
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
            hidden: o,
            bias_order: BiasOrder::SingleRound,
        },
    )
}
#[test]
fn grouped_gate_matches_cpu_across_alignments_aliases_and_replay() {
    let ctx = CudaContext::new(0).unwrap();
    let mut cases = vec![
        (1, 1, 1, [0; 4]),
        (18, 257, 6, [2; 4]),
        (1764, 2048, 9, [0; 4]),
        (1764, 2048, 9, [2; 4]),
    ];
    for mask in 0..16 {
        cases.push((
            12,
            96,
            3,
            std::array::from_fn(|i| if mask & (1 << i) != 0 { 2 } else { 0 }),
        ));
    }
    for (rows, cols, groups, offsets) in cases {
        let n = rows * cols;
        let a: Vec<_> = (0..n)
            .map(|i| round((i as i32 * 37 % 1009 - 504) as f32 / 127.))
            .collect();
        let r: Vec<_> = (0..n)
            .map(|i| round((i as i32 * 13 % 997 - 498) as f32 / 131.))
            .collect();
        let g: Vec<_> = (0..groups * 3 * cols)
            .map(|i| round((i as i32 * 17 % 509 - 254) as f32 / 193.))
            .collect();
        let expected = |res: &[f32]| -> Vec<u16> {
            (0..n)
                .map(|i| {
                    let ix = (i / cols / (rows / groups) * 3 + 2) * cols + i % cols;
                    bf16::from_f32(res[i] + round(a[i] * g[ix])).to_bits()
                })
                .collect()
        };
        let x = tensor(vec![rows, cols], &a, offsets[0]);
        let residual = tensor(vec![rows, cols], &r, offsets[1]);
        let style = tensor(vec![groups, 3, cols], &g, offsets[2]);
        let out = tensor(vec![rows, cols], &vec![0.; n], offsets[3]);
        run(&ctx, &x, &residual, &style, &out).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(bits(&out), expected(&r));
        let graph = capture(&ctx, || run(&ctx, &x, &residual, &style, &out)).unwrap();
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&out), expected(&r));
        }
        // In-place input, in-place residual and both inputs sharing output are legal.
        for which in 0..3 {
            let xx = tensor(vec![rows, cols], &a, offsets[0]);
            let rr = tensor(vec![rows, cols], &r, offsets[1]);
            let (res, out) = match which {
                0 => (&rr, &xx),
                1 => (&rr, &rr),
                _ => (&xx, &xx),
            };
            let graph = capture(&ctx, || run(&ctx, &xx, res, &style, out)).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(out), expected(if which == 2 { &a } else { &r }));
        }
        // Captured style and residual addresses must read changed contents.
        CudaBuffer::from_tensor(&style)
            .unwrap()
            .copy_from_host(&vec![0; style.size_in_bytes()])
            .unwrap();
        let rb: Vec<_> = a
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
            .collect();
        CudaBuffer::from_tensor(&residual)
            .unwrap()
            .copy_from_host(&rb)
            .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            bits(&out),
            a.iter()
                .map(|&v| bf16::from_f32(v + round(v * 0.)).to_bits())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn grouped_gate_rejects_invalid_contracts() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(vec![4, 8], &vec![1.; 32], 0);
    let r = tensor(vec![4, 8], &vec![1.; 32], 0);
    let o = tensor(vec![4, 8], &vec![0.; 32], 0);
    for shape in [vec![3, 8], vec![3, 3, 8], vec![2, 2, 8], vec![2, 3, 7]] {
        let g = tensor(shape.clone(), &vec![0.; shape.iter().product()], 0);
        assert!(run(&ctx, &x, &r, &g, &o).is_err());
    }
    let g = tensor(vec![2, 3, 8], &vec![0.; 48], 0);
    for numerics in [
        NormNumerics::Flexible,
        NormNumerics::StableDecode,
        NormNumerics::StablePlanning,
    ] {
        assert!(residual_add(
            &ctx,
            ResidualAddArgs {
                input: &x,
                bias: None,
                residual: &r,
                gate: Some(ResidualGate::GroupedStyle {
                    style: &g,
                    numerics
                }),
                hidden: &o,
                bias_order: BiasOrder::SingleRound
            }
        )
        .is_err());
    }
    let backing = CudaBuffer::alloc(96, 0).unwrap();
    let style = backing
        .as_tensor(Shape::new(vec![2, 3, 8]), DType::BF16)
        .unwrap();
    let out = backing
        .view(0, 64)
        .unwrap()
        .as_tensor(Shape::new(vec![4, 8]), DType::BF16)
        .unwrap();
    assert!(run(&ctx, &x, &r, &style, &out).is_err());
    let backing = CudaBuffer::alloc(66, 0).unwrap();
    let a = backing
        .view(0, 64)
        .unwrap()
        .as_tensor(Shape::new(vec![4, 8]), DType::BF16)
        .unwrap();
    let b = backing
        .view(2, 64)
        .unwrap()
        .as_tensor(Shape::new(vec![4, 8]), DType::BF16)
        .unwrap();
    assert!(run(&ctx, &a, &r, &g, &b).is_err());
    assert!(run(&ctx, &x, &a, &g, &b).is_err());
}
