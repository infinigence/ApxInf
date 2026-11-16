use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    ops::{attention, AttentionArgs, ExecutionSession},
    CudaBuffer, CudaContext, PreparedPhase,
};
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
#[test]
fn wide_head_attention_matches_cpu_and_retains_captured_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let batch = 2;
    let nq = 129;
    let nk = 131;
    let d = 384;
    let make = |n, seed| {
        (0..n)
            .map(|i| bf16::from_f32(((i * seed % 127) as f32 - 63.) / 64.).to_f32())
            .collect::<Vec<_>>()
    };
    let q = make(batch * nq * d, 37);
    let k = make(batch * nk * d, 17);
    let v = make(batch * nk * d, 53);
    let qt = tensor(vec![batch, nq, 1, d], &q);
    let kt = tensor(vec![batch, nk, 1, d], &k);
    let vt = tensor(vec![batch, nk, 1, d], &v);
    let mut out = tensor(vec![batch, nq, 1, d], &vec![0.; q.len()]);
    let scale = 0.0375;
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(4096, 0).unwrap(),
        || {
            let mut args = AttentionArgs::new(&qt, &kt, &vt, &mut out);
            args.scale = scale;
            args.policy.online_tune = false;
            args.policy.cache_dir = Some(String::new());
            attention(&ctx, args)
        },
    )
    .unwrap();
    drop(qt);
    drop(kt);
    drop(vt);
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let mut bytes = vec![0; out.size_in_bytes()];
    CudaBuffer::from_tensor(&out)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    let actual: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    for b in 0..batch {
        for r in 0..nq {
            let mut scores = vec![0f32; nk];
            for c in 0..nk {
                let mut dot = 0f32;
                for j in 0..d {
                    dot = q[(b * nq + r) * d + j].mul_add(k[(b * nk + c) * d + j], dot);
                }
                scores[c] = dot * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for j in 0..d {
                let mut value = 0f32;
                for c in 0..nk {
                    value = scores[c].mul_add(v[(b * nk + c) * d + j], value);
                }
                assert!(
                    (actual[(b * nq + r) * d + j] - value).abs() < 0.002,
                    "b={b},r={r},d={j}"
                );
            }
        }
    }
}

#[test]
fn tuned_wide_head_attention_matches_cpu_and_retains_captured_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let batch = 4;
    let nq = 129;
    let nk = 131;
    let d = 384;
    let make = |n, seed| {
        (0..n)
            .map(|i| bf16::from_f32(((i * seed % 127) as f32 - 63.) / 64.).to_f32())
            .collect::<Vec<_>>()
    };
    let q = make(batch * nq * d, 37);
    let k = make(batch * nk * d, 17);
    let v = make(batch * nk * d, 53);
    let qt = tensor(vec![batch, nq, 1, d], &q);
    let kt = tensor(vec![batch, nk, 1, d], &k);
    let vt = tensor(vec![batch, nk, 1, d], &v);
    let mut out = tensor(vec![batch, nq, 1, d], &vec![0.; q.len()]);
    let scale = 0.0375;
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(4096, 0).unwrap(),
        || {
            let mut args = AttentionArgs::new(&qt, &kt, &vt, &mut out);
            args.scale = scale;
            args.policy.online_tune = true;
            args.policy.cache_dir = Some(String::new());
            attention(&ctx, args)
        },
    )
    .unwrap();
    drop(qt);
    drop(kt);
    drop(vt);
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let mut bytes = vec![0; out.size_in_bytes()];
    CudaBuffer::from_tensor(&out)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    let actual: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect();
    for b in 0..batch {
        for r in 0..nq {
            let mut scores = vec![0f32; nk];
            for c in 0..nk {
                let mut dot = 0f32;
                for j in 0..d {
                    dot = q[(b * nq + r) * d + j].mul_add(k[(b * nk + c) * d + j], dot);
                }
                scores[c] = dot * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for j in 0..d {
                let mut value = 0f32;
                for c in 0..nk {
                    value = scores[c].mul_add(v[(b * nk + c) * d + j], value);
                }
                assert!(
                    (actual[(b * nq + r) * d + j] - value).abs() < 0.002,
                    "b={b},r={r},d={j}"
                );
            }
        }
    }
}
