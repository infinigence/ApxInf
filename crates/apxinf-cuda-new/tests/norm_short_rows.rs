//! Short-row RMS covers tail warps, zero rows, epsilon, and graph/in-place use.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{capture, ops, CudaBuffer, CudaContext};

fn tensor(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| match dtype {
            DType::F16 => half::f16::from_f32(v).to_bits().to_ne_bytes(),
            _ => half::bf16::from_f32(v).to_bits().to_ne_bytes(),
        })
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}
fn read(t: &Tensor) -> Vec<f32> {
    let mut bytes = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_ne_bytes(b.try_into().unwrap());
            match t.dtype() {
                DType::F16 => half::f16::from_bits(bits).to_f32(),
                _ => half::bf16::from_bits(bits).to_f32(),
            }
        })
        .collect()
}
#[test]
fn short_rms_matches_reference_and_in_place_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        for rows in [1, 7, 8, 9, 17, 8192] {
            for eps in [1e-6, 0.125] {
                let values: Vec<f32> = (0..rows * 128)
                    .map(|i| {
                        if i < 128 {
                            0.
                        } else {
                            ((i * 37 % 509) as f32 - 254.) / 128.
                        }
                    })
                    .collect();
                let weights: Vec<f32> = (0..128).map(|i| (i as f32 - 64.) / 64.).collect();
                let input = tensor(dtype, vec![rows, 128], &values);
                let alias = tensor(dtype, vec![rows, 128], &values);
                let weight = tensor(dtype, vec![128], &weights);
                let output = tensor(dtype, vec![rows, 128], &vec![0.; rows * 128]);
                ops::norm::norm(&ctx, ops::NormArgs::rms(&input, &weight, &output, eps)).unwrap();
                let graph = capture(&ctx, || {
                    ops::norm::norm(&ctx, ops::NormArgs::rms(&alias, &weight, &alias, eps))
                })
                .unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let actual = read(&output);
                assert_eq!(actual, read(&alias));
                let rounded_input = read(&input);
                for (r, row) in rounded_input.chunks_exact(128).enumerate() {
                    let inv = 1.
                        / (row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / 128.
                            + eps as f64)
                            .sqrt();
                    for c in 0..128 {
                        let expected = (row[c] as f64 * inv * weights[c] as f64) as f32;
                        let tolerance = if dtype == DType::F16 { 0.001 } else { 0.008 };
                        assert!(
                            (actual[r * 128 + c] - expected).abs()
                                <= tolerance * expected.abs().max(1.),
                            "dtype={dtype:?} rows={rows} row={r} col={c}"
                        );
                    }
                }
            }
        }
    }
}
