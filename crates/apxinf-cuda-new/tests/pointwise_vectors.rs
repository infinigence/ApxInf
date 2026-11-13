use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture,
    ops::{pointwise, PointwiseArgs, PointwiseSemantic},
    CudaBuffer, CudaContext,
};
use half::bf16;
fn tensor(v: &[f32], offset: usize) -> Tensor {
    let storage = CudaBuffer::alloc(v.len() * 2 + offset, 0).unwrap();
    let buffer = storage.view(offset, v.len() * 2).unwrap();
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
        .collect();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
        .as_tensor(Shape::new(vec![1, v.len()]), DType::BF16)
        .unwrap()
}
fn bits(t: &Tensor) -> Vec<u16> {
    let mut v = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut v)
        .unwrap();
    v.chunks_exact(2)
        .map(|v| u16::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}
#[test]
fn binary_pointwise_matches_cpu_with_alignment_tails_and_inplace_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let mut cases = vec![
        (1, 0, 0, 0),
        (8, 0, 0, 0),
        (1023, 0, 0, 0),
        (1025, 0, 0, 0),
        (1764 * 2048, 0, 0, 0),
    ];
    for a in [0, 2] {
        for b in [0, 2] {
            for o in [0, 2] {
                cases.push((1024, a, b, o));
            }
        }
    }
    for (n, ao, bo, oo) in cases {
        let a: Vec<f32> = (0..n)
            .map(|i| bf16::from_f32(((i * 37 % 509) as f32 - 254.) / 127.).to_f32())
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| bf16::from_f32(((i * 17 % 511) as f32 - 255.) / 128.).to_f32())
            .collect();
        for semantic in [PointwiseSemantic::Multiply, PointwiseSemantic::EulerUpdate] {
            let dt = -0.137f32;
            let expected: Vec<u16> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    bf16::from_f32(if semantic == PointwiseSemantic::Multiply {
                        x * y
                    } else {
                        dt.mul_add(y, x)
                    })
                    .to_bits()
                })
                .collect();
            let at = tensor(&a, ao);
            let bt = tensor(&b, bo);
            let mut out = tensor(&vec![0.; n], oo);
            let mut args = PointwiseArgs::new(semantic, &at, &mut out);
            args.secondary = Some(&bt);
            args.dt = dt;
            pointwise(&ctx, args).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                bits(&out),
                expected,
                "n={n},offsets={ao},{bo},{oo},semantic={semantic:?}"
            );
            let inplace = tensor(&a, ao);
            let graph = capture(&ctx, || {
                let mut output = inplace.clone();
                let mut args = PointwiseArgs::new(semantic, &inplace, &mut output);
                args.secondary = Some(&bt);
                args.dt = dt;
                pointwise(&ctx, args)
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&inplace), expected);
        }
    }
}
