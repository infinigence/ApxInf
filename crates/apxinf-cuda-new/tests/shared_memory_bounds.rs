use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::{relative_attention, RelativeAttentionArgs};
use apxinf_cuda_new::{norm, Affine, CudaBuffer, CudaContext, NormArgs, NormKind, NormNumerics};
use half::f16;

const MAX_BOUND: usize = (48 * 1024 - 32 * 4) / 4;

fn tensor(shape: Vec<usize>, fill: impl Fn(usize) -> f32) -> Tensor {
    let count: usize = shape.iter().product();
    let bytes: Vec<u8> = (0..count)
        .flat_map(|i| f16::from_f32(fill(i)).to_bits().to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::F16).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    let mut bytes = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|p| f16::from_bits(u16::from_ne_bytes([p[0], p[1]])).to_f32())
        .collect()
}

fn run_relative_attention(ctx: &CudaContext, length: usize) -> apxinf_core::Result<Vec<f32>> {
    let head_dim = 32;
    let window = 4;
    let qkv = |salt: usize| {
        tensor(vec![1, length, 1, head_dim], move |i| {
            (((i + salt) * 7 % 23) as f32 - 11.0) / 16.0
        })
    };
    let (q, k, v) = (qkv(1), qkv(2), qkv(3));
    let table = |salt: usize| {
        tensor(vec![2 * window + 1, head_dim], move |i| {
            (((i + salt) * 5 % 17) as f32 - 8.0) / 16.0
        })
    };
    let (rk, rv) = (table(1), table(2));
    let mut out = tensor(vec![1, length, 1, head_dim], |_| 0.0);
    let mut args = RelativeAttentionArgs::new(&q, &k, &v, &rk, &rv, &mut out);
    args.window = window;
    relative_attention(ctx, args)?;
    ctx.synchronize().unwrap();
    Ok(host(&out))
}

fn run_grn(ctx: &CudaContext, cols: usize) -> apxinf_core::Result<Vec<f32>> {
    let input = tensor(vec![1, cols], |i| ((i * 13 % 31) as f32 - 15.0) / 16.0);
    let style = tensor(vec![2 * cols], |i| if i < cols { 0.5 } else { 0.25 });
    let out = tensor(vec![1, cols], |_| 0.0);
    norm(
        ctx,
        NormArgs {
            input: &input,
            output: Some(&out),
            kind: NormKind::Grn,
            affine: Affine::Adaptive { style: &style },
            quantized: None,
            numerics: NormNumerics::Flexible,
            eps: 0.0,
        },
    )?;
    ctx.synchronize().unwrap();
    Ok(host(&out))
}

#[test]
fn relative_attention_launches_at_the_admitted_bound_and_rejects_one_past_it() {
    let ctx = CudaContext::new(0).unwrap();
    let at = run_relative_attention(&ctx, MAX_BOUND).unwrap();
    assert_eq!(at.len(), MAX_BOUND * 32);
    assert!(at.iter().all(|x| x.is_finite()));
    assert!(at.iter().any(|x| *x != 0.0));
    let below = run_relative_attention(&ctx, MAX_BOUND - 1).unwrap();
    assert!(below.iter().all(|x| x.is_finite()));
    assert!(run_relative_attention(&ctx, MAX_BOUND + 1).is_err());
    assert!(run_relative_attention(&ctx, 12288).is_err());
}

#[test]
fn grn_launches_at_the_admitted_bound_and_rejects_one_past_it() {
    let ctx = CudaContext::new(0).unwrap();
    let at = run_grn(&ctx, MAX_BOUND).unwrap();
    assert_eq!(at.len(), MAX_BOUND);
    assert!(at.iter().all(|x| x.is_finite()));
    assert!(at.iter().any(|x| *x != 0.0));
    assert!(run_grn(&ctx, MAX_BOUND + 1).is_err());
    assert!(run_grn(&ctx, 12288).is_err());
}
