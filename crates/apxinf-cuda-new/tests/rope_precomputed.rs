use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{capture, ops::rope::apply_precomputed, CudaBuffer, CudaContext};
use half::bf16;
fn tensor(shape: Vec<usize>, values: &[f32], dtype: DType) -> Tensor {
    let bytes: Vec<u8> = if dtype == DType::F32 {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    } else {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect()
    };
    let b = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    b.copy_from_host(&bytes).unwrap();
    b.as_tensor(Shape::new(shape), dtype).unwrap()
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
fn precomputed_rotation_matches_staged_cpu_and_in_place_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for (tokens, heads, dim) in [
        (1, 1, 2),
        (17, 3, 64),
        (1764, 16, 128),
        (9, 2, 256),
        (2, 1, 2048),
    ] {
        let n = tokens * heads * dim;
        let values: Vec<f32> = (0..n)
            .map(|i| bf16::from_f32((i as i32 * 37 % 509 - 254) as f32 / 127.).to_f32())
            .collect();
        let mut table = vec![0.; tokens * dim];
        for t in 0..tokens {
            for c in 0..dim / 2 {
                let angle = (t * 31 + c) as f32 * 0.0017;
                table[t * dim + c] = angle.cos();
                table[t * dim + dim / 2 + c] = angle.sin();
            }
        }
        let x = tensor(vec![tokens, heads, dim], &values, DType::BF16);
        let inplace = tensor(vec![tokens, heads, dim], &values, DType::BF16);
        let output = tensor(vec![tokens, heads, dim], &vec![0.; n], DType::BF16);
        let coeff = tensor(vec![tokens, 2, dim / 2], &table, DType::F32);
        apply_precomputed(&ctx, &x, &coeff, &output).unwrap();
        let graph = capture(&ctx, || apply_precomputed(&ctx, &inplace, &coeff, &inplace)).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let actual = bits(&output);
        assert_eq!(actual, bits(&inplace));
        for i in 0..n {
            let c = i % dim;
            let t = i / (heads * dim);
            let j = i / dim * dim + (c + dim / 2) % dim;
            let a = values[i] * table[t * dim + c % (dim / 2)];
            let sin = table[t * dim + dim / 2 + c % (dim / 2)] * if c < dim / 2 { -1. } else { 1. };
            let b = values[j] * sin;
            assert_eq!(
                actual[i],
                bf16::from_f32(a + b).to_bits(),
                "shape={tokens},{heads},{dim},i={i}"
            );
        }
        assert!(apply_precomputed(
            &ctx,
            &x,
            &coeff.reshape(vec![tokens, dim]).unwrap(),
            &output
        )
        .is_err());
    }
}
#[test]
fn precomputed_rotation_rejects_partial_alias_and_bad_dtype() {
    let ctx = CudaContext::new(0).unwrap();
    let storage = CudaBuffer::alloc(258, 0).unwrap();
    let x = storage
        .view(0, 256)
        .unwrap()
        .as_tensor(Shape::new(vec![1, 1, 128]), DType::BF16)
        .unwrap();
    let shifted = storage
        .view(2, 256)
        .unwrap()
        .as_tensor(x.shape().clone(), DType::BF16)
        .unwrap();
    let table = tensor(vec![1, 2, 64], &vec![1.; 128], DType::F32);
    assert!(apply_precomputed(&ctx, &x, &table, &shifted).is_err());
    let table_storage = CudaBuffer::from_tensor(&table).unwrap();
    let table_output = table_storage
        .view(0, 256)
        .unwrap()
        .as_tensor(x.shape().clone(), DType::BF16)
        .unwrap();
    assert!(apply_precomputed(&ctx, &x, &table, &table_output).is_err());
    let bad = tensor(vec![1, 2, 64], &vec![1.; 128], DType::BF16);
    assert!(apply_precomputed(&ctx, &x, &bad, &x).is_err());
    let x32 = tensor(vec![1, 1, 128], &vec![1.; 128], DType::F32);
    assert!(apply_precomputed(&ctx, &x32, &table, &x32).is_err());
}
