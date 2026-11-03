use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture, norm, Affine, CudaBuffer, CudaContext, NormArgs, NormKind, NormNumerics,
};
use half::bf16;

fn tensor(ctx: &CudaContext, shape: Vec<usize>, values: &[f32], offset: usize) -> Tensor {
    let allocation = CudaBuffer::alloc(values.len() * 2 + offset, 0).unwrap();
    let buffer = allocation.view(offset, values.len() * 2).unwrap();
    let data: Vec<u8> = values
        .iter()
        .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
        .collect();
    buffer.copy_from_host_on(&data, ctx.stream()).unwrap();
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
fn round(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}
fn grouped(
    ctx: &CudaContext,
    x: &Tensor,
    style: &Tensor,
    out: &Tensor,
    shift_first: bool,
) -> apxinf_core::Result<()> {
    norm(
        ctx,
        NormArgs {
            input: x,
            output: Some(out),
            kind: NormKind::Layer,
            affine: Affine::GroupedAdaptive { style, shift_first },
            numerics: NormNumerics::StagedBf16,
            quantized: None,
            eps: 1e-6,
        },
    )
}

#[test]
fn grouped_affine_preserves_staged_bf16_and_graph_input_updates() {
    let ctx = CudaContext::new(0).unwrap();
    for (rows, cols, groups) in [(1, 7, 1), (18, 257, 6), (1764, 2048, 9)] {
        for parts in [2, 3] {
            for shift_first in [false, true] {
                // Two-byte offset deliberately exercises non-vector-aligned bindings.
                let values: Vec<_> = (0..rows * cols)
                    .map(|i| round((i as i32 * 37 % 1009 - 504) as f32 / 127.))
                    .collect();
                let styles: Vec<_> = (0..groups * parts * cols)
                    .map(|i| round((i as i32 * 17 % 509 - 254) as f32 / 193.))
                    .collect();
                let x = tensor(&ctx, vec![rows, cols], &values, 2);
                let style = tensor(&ctx, vec![groups, parts, cols], &styles, 2);
                let out = tensor(&ctx, vec![rows, cols], &vec![0.; rows * cols], 2);
                let ln = tensor(&ctx, vec![rows, cols], &vec![0.; rows * cols], 2);
                let one = tensor(&ctx, vec![cols], &vec![1.; cols], 0);
                let zero = tensor(&ctx, vec![cols], &vec![0.; cols], 0);
                norm(&ctx, NormArgs::layer(&x, &one, &zero, &ln, 1e-6)).unwrap();
                ctx.synchronize().unwrap();
                let base = bits(&ln);
                let expected: Vec<_> = base
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let st = (i / cols / (rows / groups)) * parts * cols + i % cols;
                        let shift = styles[st + if shift_first { 0 } else { cols }];
                        let scale = round(styles[st + if shift_first { cols } else { 0 }] + 1.);
                        bf16::from_f32(round(bf16::from_bits(v).to_f32() * scale) + shift).to_bits()
                    })
                    .collect();
                grouped(&ctx, &x, &style, &out, shift_first).unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    bits(&out),
                    expected,
                    "shape={rows}x{cols},parts={parts},shift_first={shift_first}"
                );
                // Independent f64 population-variance reference checks the full formula,
                // beyond exact differential comparison with the existing reduction.
                for row in 0..rows {
                    let slice = &values[row * cols..(row + 1) * cols];
                    let mean = slice.iter().map(|&x| x as f64).sum::<f64>() / cols as f64;
                    let var = slice
                        .iter()
                        .map(|&x| (x as f64 - mean).powi(2))
                        .sum::<f64>()
                        / cols as f64;
                    for col in 0..cols {
                        let st = (row / (rows / groups)) * parts * cols + col;
                        let n = round(((slice[col] as f64 - mean) / (var + 1e-6).sqrt()) as f32);
                        let scale = round(styles[st + if shift_first { cols } else { 0 }] + 1.);
                        let shift = styles[st + if shift_first { 0 } else { cols }];
                        let reference = round(round(n * scale) + shift);
                        let actual = bf16::from_bits(expected[row * cols + col]).to_f32();
                        assert!(
                            (actual - reference).abs() <= 0.04,
                            "CPU reference: {actual} vs {reference}"
                        );
                    }
                }
                let graph = capture(&ctx, || grouped(&ctx, &x, &style, &out, shift_first)).unwrap();
                for _ in 0..3 {
                    graph.replay().unwrap();
                    ctx.synchronize().unwrap();
                    assert_eq!(bits(&out), expected);
                }
                // Upload on the graph stream: pageable default-stream copies may
                // return before DMA finishes, and this stream is nonblocking.
                // Constant new input has exact zero normalized values; result is shift.
                let new_values: Vec<u8> = (0..rows * cols)
                    .flat_map(|_| bf16::from_f32(2.).to_bits().to_ne_bytes())
                    .collect();
                CudaBuffer::from_tensor(&x)
                    .unwrap()
                    .copy_from_host_on(&new_values, ctx.stream())
                    .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let expected_shift: Vec<_> = (0..rows * cols)
                    .map(|i| {
                        let st = (i / cols / (rows / groups)) * parts * cols
                            + i % cols
                            + if shift_first { 0 } else { cols };
                        bf16::from_f32(styles[st]).to_bits()
                    })
                    .collect();
                assert_eq!(bits(&out), expected_shift);
                // Exact in-place operation has the same result and no cross-row hazard.
                grouped(&ctx, &x, &style, &x, shift_first).unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(bits(&x), expected_shift);
                CudaBuffer::from_tensor(&x)
                    .unwrap()
                    .copy_from_host_on(&new_values, ctx.stream())
                    .unwrap();
                CudaBuffer::from_tensor(&style)
                    .unwrap()
                    .copy_from_host_on(&vec![0; style.size_in_bytes()], ctx.stream())
                    .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(bits(&out), vec![0; rows * cols], "captured style updates");
            }
        }
    }
}

#[test]
fn grouped_affine_rejects_invalid_shapes_numerics_and_aliases() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(&ctx, vec![4, 8], &vec![1.; 32], 0);
    let out = tensor(&ctx, vec![4, 8], &vec![0.; 32], 0);
    for shape in [vec![2, 8], vec![3, 2, 8], vec![2, 4, 8], vec![2, 2, 7]] {
        let s = tensor(&ctx, shape.clone(), &vec![0.; shape.iter().product()], 0);
        assert!(grouped(&ctx, &x, &s, &out, true).is_err());
    }
    let s = tensor(&ctx, vec![2, 2, 8], &vec![0.; 32], 0);
    assert!(grouped(&ctx, &x, &s, &s, true).is_err());
    for numerics in [
        NormNumerics::Flexible,
        NormNumerics::StableDecode,
        NormNumerics::StablePlanning,
    ] {
        assert!(norm(
            &ctx,
            NormArgs {
                input: &x,
                output: Some(&out),
                kind: NormKind::Layer,
                affine: Affine::GroupedAdaptive {
                    style: &s,
                    shift_first: true
                },
                numerics,
                quantized: None,
                eps: 1e-6
            }
        )
        .is_err());
    }
    let storage = CudaBuffer::alloc(66, 0).unwrap();
    let a = storage
        .view(0, 64)
        .unwrap()
        .as_tensor(Shape::new(vec![4, 8]), DType::BF16)
        .unwrap();
    let b = storage
        .view(2, 64)
        .unwrap()
        .as_tensor(Shape::new(vec![4, 8]), DType::BF16)
        .unwrap();
    assert!(grouped(&ctx, &a, &s, &b, true).is_err());
}

#[test]
fn grouped_affine_preserves_zero_signs() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(&ctx, vec![1, 1], &[-0.0], 0);
    let style = tensor(&ctx, vec![1, 2, 1], &[-0.0, 0.0], 0);
    let out = tensor(&ctx, vec![1, 1], &[1.0], 0);
    grouped(&ctx, &x, &style, &out, true).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(bits(&out), vec![0]);
}
