use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::{gemm, gemm_gelu_exact, GemmArgs, GemmPolicy, GemmWeightLayout};
use apxinf_cuda_new::{
    capture, pointwise, CudaBuffer, CudaContext, ExecutionSession, PointwiseActivation,
    PointwiseArgs, PointwiseSemantic,
};
use half::bf16;
fn tensor(shape: Vec<usize>, values: &[f32], offset: usize) -> Tensor {
    let buffer = CudaBuffer::alloc(values.len() * 2 + offset, 0)
        .unwrap()
        .view(offset, values.len() * 2)
        .unwrap();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
        .collect();
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
    GemmPolicy {
        online_tune: tune,
        cache_dir: Some(String::new()),
        ..Default::default()
    }
}
fn fused(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    p: &GemmPolicy,
) -> apxinf_core::Result<()> {
    let mut o = out.clone();
    let mut args = GemmArgs::new(a, b, &mut o);
    args.policy = p.clone();
    gemm_gelu_exact(ctx, args)
}
fn unfused(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) {
    let mut o = out.clone();
    let mut args = GemmArgs::new(a, b, &mut o);
    args.policy = policy(false);
    gemm(ctx, args).unwrap();
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, out, &mut o);
    args.activation = PointwiseActivation::GeluExact;
    pointwise(ctx, args).unwrap();
}
#[test]
fn exact_gelu_fallback_and_rejected_contracts() {
    let ctx = CudaContext::new(0).unwrap();
    for (m, k, n, offset) in [(3, 7, 5, 2), (8, 16, 16, 0)] {
        let av: Vec<_> = (0..m * k).map(|i| (i as i32 % 9 - 4) as f32 / 4.).collect();
        let bv: Vec<_> = (0..k * n).map(|i| (i as i32 % 7 - 3) as f32 / 8.).collect();
        let a = tensor(vec![m, k], &av, offset);
        let b = tensor(vec![k, n], &bv, offset);
        let out = tensor(vec![m, n], &vec![0.; m * n], offset);
        let expected = tensor(vec![m, n], &vec![0.; m * n], offset);
        unfused(&ctx, &a, &b, &expected);
        fused(&ctx, &a, &b, &out, &policy(false)).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(bits(&out), bits(&expected));
        let mut insufficient = policy(false);
        insufficient.workspace_limit = 0;
        assert!(fused(&ctx, &a, &b, &out, &insufficient).is_err());
        let mut denied = policy(false);
        denied.allow_fallback = false;
        assert!(fused(&ctx, &a, &b, &out, &denied).is_err());
        let mut o = out.clone();
        let mut bad = GemmArgs::new(&a, &b, &mut o);
        bad.alpha = 0.5;
        assert!(gemm_gelu_exact(&ctx, bad).is_err());
        let mut bad = GemmArgs::new(&a, &b, &mut o);
        bad.weight_layout = GemmWeightLayout::OutputMajor;
        assert!(gemm_gelu_exact(&ctx, bad).is_err());
    }
    let a = tensor(vec![4, 4], &[1.; 16], 0);
    let b = tensor(vec![4, 4], &[1.; 16], 0);
    let mut alias = a.clone();
    assert!(gemm_gelu_exact(&ctx, GemmArgs::new(&a, &b, &mut alias)).is_err());
}
#[test]
fn exact_gelu_tuning_graph_updates_and_retention() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (1764, 8192, 2048);
    let mut av = vec![0.; m * k];
    let mut bv = vec![0.; k * n];
    for i in 0..m {
        av[i * k] = (i % 5) as f32 - 2.;
    }
    for j in 0..n {
        bv[j] = (j % 5) as f32 / 4. - 0.5;
    }
    let a = tensor(vec![m, k], &av, 0);
    let b = tensor(vec![k, n], &bv, 0);
    let out = tensor(vec![m, n], &vec![0.; m * n], 0);
    let expected = tensor(vec![m, n], &vec![0.; m * n], 0);
    let session = ExecutionSession::with_capacity(256 * 1024 * 1024, 0).unwrap();
    let p = policy(true);
    session.prepare(|| fused(&ctx, &a, &b, &out, &p)).unwrap();
    let graph = capture(&ctx, || session.run(|| fused(&ctx, &a, &b, &out, &p))).unwrap();
    graph.replay().unwrap();
    unfused(&ctx, &a, &b, &expected);
    ctx.synchronize().unwrap();
    assert_eq!(bits(&out), bits(&expected));
    // Both mutable bindings change after capture; no immutable weight packing.
    let updated_a = tensor(vec![m, k], &av.iter().map(|x| -*x).collect::<Vec<_>>(), 0);
    let updated_b = tensor(
        vec![k, n],
        &bv.iter().map(|x| *x * 0.5).collect::<Vec<_>>(),
        0,
    );
    apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &updated_a, &a).unwrap();
    apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &updated_b, &b).unwrap();
    unfused(&ctx, &a, &b, &expected);
    ctx.synchronize().unwrap();
    let expected_bits = bits(&expected);
    drop(a);
    drop(b);
    drop(session);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(bits(&out), expected_bits);
}
