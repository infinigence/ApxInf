use super::*;
use crate::{transfers, CudaContext, ExecutionSession};
use apxinf_core::{DType, Shape, Tensor};

unsafe extern "C" {
    fn apxinf_attention_test_efficient_support_domain() -> std::ffi::c_int;
}

#[test]
fn efficient_attention_d384_native_offset_limits() {
    assert_eq!(
        unsafe { apxinf_attention_test_efficient_support_domain() },
        1
    );
}

#[test]
fn efficient_attention_d384_all_candidates_scalar_oracle_and_retained_graph() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    // Multiple batches/heads, partial query tile, and a second partial key tile.
    let (batch, qs, ks, heads, dim) = (2, 35, 137, 2, 384);
    let qv: Vec<_> = (0..batch * qs * heads * dim)
        .map(|i| ((i * 3 % 17) as f32 - 8.) / 16.)
        .collect();
    let kv: Vec<_> = (0..batch * ks * heads * dim)
        .map(|i| ((i * 5 % 19) as f32 - 9.) / 16.)
        .collect();
    let vv: Vec<_> = (0..batch * ks * heads * dim)
        .map(|i| 0.75 + ((i * 7 % 23) as f32 - 11.) / 32.)
        .collect();
    // Nondefault scaling exercises the binding rather than a hardcoded 1/sqrt(D).
    let scale = 0.0625f32;
    let mut expected = vec![0f32; qv.len()];
    for b in 0..batch {
        for h in 0..heads {
            for q in 0..qs {
                let qi = ((b * qs + q) * heads + h) * dim;
                let mut p: Vec<f64> = (0..ks)
                    .map(|k| {
                        let ki = ((b * ks + k) * heads + h) * dim;
                        (0..dim)
                            .map(|d| qv[qi + d] as f64 * kv[ki + d] as f64)
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for x in &mut p {
                    *x = (*x - max).exp();
                }
                let sum: f64 = p.iter().sum();
                for d in 0..dim {
                    expected[qi + d] = (0..ks)
                        .map(|k| p[k] / sum * vv[((b * ks + k) * heads + h) * dim + d] as f64)
                        .sum::<f64>() as f32;
                }
            }
        }
    }
    for dtype in [DType::F16, DType::BF16] {
        let upload = |shape, values: &[f32]| {
            let host = if dtype == DType::BF16 {
                Tensor::from_bf16(
                    shape,
                    &values
                        .iter()
                        .map(|&v| half::bf16::from_f32(v))
                        .collect::<Vec<_>>(),
                )
            } else {
                Tensor::from_f16(
                    shape,
                    &values
                        .iter()
                        .map(|&v| half::f16::from_f32(v))
                        .collect::<Vec<_>>(),
                )
            }
            .unwrap();
            transfers::to_cuda(&host, 0).unwrap()
        };
        let q = upload(vec![batch, qs, heads, dim], &qv);
        let k = upload(vec![batch, ks, heads, dim], &kv);
        let v = upload(vec![batch, ks, heads, dim], &vv);
        let retained = v.clone();
        let mut out = ctx
            .allocate_output(Shape::new(vec![batch, qs, heads, dim]), dtype)
            .unwrap();
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        args.scale = scale;
        args.policy.cache_dir = Some(String::new());
        args.policy.workspace_limit = 16 << 20;
        let normalized = crate::ops::attention::contracts::normalize(&ctx, args).unwrap();
        crate::ops::attention::execution::validate_candidates(&ctx, &normalized, &expected)
            .unwrap();
        let execution = crate::ops::attention::execution::prepare(&ctx, normalized).unwrap();
        if std::env::var_os("APXINF_TEST_REQUIRE_EFFICIENT_ATTENTION").is_some() {
            assert!(
                execution
                    .summary()
                    .contains("cutlass-memory-efficient-attention"),
                "{}",
                execution.summary()
            );
        }
        drop(execution);
        let run = |out: &mut Tensor| {
            let mut args = AttentionArgs::new(&q, &k, &v, out);
            args.scale = scale;
            args.policy.cache_dir = Some(String::new());
            args.policy.workspace_limit = 16 << 20;
            attention(&ctx, args)
        };
        let session = ExecutionSession::with_capacity(256, 0).unwrap();
        session.prepare(|| run(&mut out)).unwrap();
        let graph = crate::capture(&ctx, || session.run(|| run(&mut out))).unwrap();
        drop(session);
        drop(q);
        drop(k);
        drop(v);
        let changed = upload(
            vec![batch, ks, heads, dim],
            &vv.iter().map(|v| -v).collect::<Vec<_>>(),
        );
        transfers::copy_cuda_to_cuda(&ctx, &changed, &retained).unwrap();
        drop(changed);
        drop(retained);
        let mut first = None;
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
            if let Some(previous) = &first {
                assert_eq!(&actual, previous);
            }
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &e)| (a as f64 + e as f64).powi(2))
                .sum::<f64>();
            let norm = expected.iter().map(|&e| (e as f64).powi(2)).sum::<f64>();
            assert!(
                (error / norm).sqrt() < 0.004,
                "{dtype:?} relative error {}",
                (error / norm).sqrt()
            );
            first = Some(actual);
        }
    }
}
