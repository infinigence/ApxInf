//! Cosmos3-Edge MoT MLP up-projection: fused `GEMM + relu(x)^2` epilogue.
//!
//! The fused semantic must reproduce `plain GEMM -> BF16 -> relu^2 -> BF16`
//! exactly, because the activation is computed on the BF16-rounded projection.
//! Dyadic operands make every FP32 partial sum exact, so both paths must agree
//! bit for bit even though the GEMM kernels accumulate in a different order.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::{gemm, gemm_relu_squared, GemmArgs, GemmPolicy, WeightVersion};
use apxinf_cuda_new::{
    pointwise, CudaBuffer, CudaContext, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
};
use half::bf16;

const M: usize = 1971;
const N: usize = 9216;
const K: usize = 2048;

fn tensor(shape: Vec<usize>, values: &[bf16]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|v| v.to_bits().to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn bits(t: &Tensor) -> Vec<u8> {
    let mut b = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut b)
        .unwrap();
    b
}

fn policy(tune: bool) -> GemmPolicy {
    // `APXINF_TEST_RECIPES` pins the persistable recipes (for example the
    // production directory) so a profiling run executes exactly one kernel per
    // semantic instead of replaying the tuner's candidate sweep. Empty means
    // in-process only.
    let cache_dir = std::env::var("APXINF_TEST_RECIPES")
        .ok()
        .or_else(|| Some(String::new()));
    GemmPolicy {
        online_tune: tune,
        cache_dir,
        ..Default::default()
    }
}

fn fused(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor, p: &GemmPolicy) {
    let mut o = out.clone();
    let mut args = GemmArgs::new(a, b, &mut o);
    args.policy = p.clone();
    // Match the production immutable-weight key so a pinned recipe directory
    // skips the tuner (and the production recipes apply verbatim).
    args.weight_version = Some(WeightVersion::new(1));
    gemm_relu_squared(ctx, args).unwrap();
}

fn unfused(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor, p: &GemmPolicy) {
    let mut o = out.clone();
    let mut args = GemmArgs::new(a, b, &mut o);
    args.policy = p.clone();
    args.weight_version = Some(WeightVersion::new(1));
    gemm(ctx, args).unwrap();
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, out, &mut o);
    args.activation = PointwiseActivation::ReluSquared;
    pointwise(ctx, args).unwrap();
}

/// Small dyadic values: every product and partial sum is exactly representable
/// in FP32, so BF16 rounding cannot depend on accumulation order.
fn dyadic(seed: usize, len: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let step = (i * 37 + seed * 101) % 7;
            bf16::from_f32((step as f32 - 3.0) * 0.25)
        })
        .collect()
}

fn randomish(seed: usize, len: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let step = ((i * 2654435761usize + seed * 40503) % 2003) as f32;
            bf16::from_f32((step - 1001.0) / 512.0)
        })
        .collect()
}

fn compare(fused: &[u8], reference: &[u8]) -> (usize, f32, f32, f64) {
    let mut diff = 0;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (x, y) in fused.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let x = bf16::from_bits(u16::from_ne_bytes([x[0], x[1]])).to_f32();
        let y = bf16::from_bits(u16::from_ne_bytes([y[0], y[1]])).to_f32();
        if x.to_bits() != y.to_bits() {
            diff += 1;
        }
        max_abs = max_abs.max((x - y).abs());
        max_rel = max_rel.max((x - y).abs() / y.abs().max(1e-30));
        dot += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    (diff, max_abs, max_rel, dot / (aa.sqrt() * bb.sqrt()))
}

#[test]
fn fused_relu2_is_bitwise_identical_on_exact_operands() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(vec![M, K], &dyadic(1, M * K));
    let b = tensor(vec![K, N], &dyadic(2, K * N));
    let out = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    let expected = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    // Tune the fused semantic so the CUTLASS epilogue kernel is selected; the
    // unfused path keeps its own tuned recipe.
    fused(&ctx, &a, &b, &out, &policy(true));
    unfused(&ctx, &a, &b, &expected, &policy(true));
    ctx.synchronize().unwrap();
    assert_eq!(bits(&out), bits(&expected));
}

#[test]
fn fused_relu2_matches_reference_on_random_operands() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(vec![M, K], &randomish(3, M * K));
    let b = tensor(vec![K, N], &randomish(4, K * N));
    let out = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    let expected = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    fused(&ctx, &a, &b, &out, &policy(true));
    unfused(&ctx, &a, &b, &expected, &policy(true));
    ctx.synchronize().unwrap();
    let (diff, max_abs, max_rel, cosine) = compare(&bits(&out), &bits(&expected));
    eprintln!(
        "fused relu2 vs GEMM+pointwise: diffs={diff}/{} max_abs={max_abs:.6} max_rel={max_rel:.6} cosine={cosine:.9}",
        M * N
    );
    assert_eq!(
        diff, 0,
        "pinned fused recipe must preserve GEMM + relu² bits"
    );
    assert!(cosine > 0.999999);
    assert!(max_rel < 1e-2);
    // relu^2 is non-negative, and a zero accumulator must stay zero.
    let values: Vec<_> = bits(&out)
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32())
        .collect();
    assert!(values.iter().all(|v| *v >= 0.0));
}

#[test]
#[ignore]
fn fused_relu2_timing() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(vec![M, K], &randomish(5, M * K));
    let b = tensor(vec![K, N], &randomish(6, K * N));
    let out = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    let expected = tensor(vec![M, N], &vec![bf16::ZERO; M * N]);
    let p = policy(true);
    // Warm-up also runs the tuner and persists the selected recipe.
    for _ in 0..5 {
        fused(&ctx, &a, &b, &out, &p);
        unfused(&ctx, &a, &b, &expected, &p);
    }
    ctx.synchronize().unwrap();
    let iterations = 30;
    let time = |run: &mut dyn FnMut()| {
        ctx.synchronize().unwrap();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            run();
        }
        ctx.synchronize().unwrap();
        start.elapsed().as_secs_f64() * 1e3 / iterations as f64
    };
    let fused_ms = time(&mut || fused(&ctx, &a, &b, &out, &p));
    let unfused_ms = time(&mut || unfused(&ctx, &a, &b, &expected, &p));
    eprintln!(
        "M={M} N={N} K={K}: fused={fused_ms:.3} ms  gemm+relu2={unfused_ms:.3} ms  saved={:.3} ms ({:+.1}%)",
        unfused_ms - fused_ms,
        (fused_ms / unfused_ms - 1.0) * 100.0
    );
}
