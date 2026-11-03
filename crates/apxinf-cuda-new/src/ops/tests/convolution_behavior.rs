use super::*;
use crate::{transfers, CudaContext, ExecutionSession};
use apxinf_core::{DType, Shape, Tensor};

fn available() -> bool {
    let ready = convolution_cudnn_version().is_some();
    if std::env::var_os("APXINF_TEST_REQUIRE_CUDNN").is_some() {
        assert!(
            ready,
            "cuDNN provider is required for this target validation"
        );
    }
    ready
}
fn upload(ctx: &CudaContext, shape: Vec<usize>, values: &[f32], dtype: DType) -> Tensor {
    let host = match dtype {
        DType::F16 => Tensor::from_f16(
            shape,
            &values
                .iter()
                .map(|&x| half::f16::from_f32(x))
                .collect::<Vec<_>>(),
        ),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values
                .iter()
                .map(|&x| half::bf16::from_f32(x))
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    }
    .unwrap();
    transfers::to_cuda(&host, ctx.device_id()).unwrap()
}
fn rounded(value: f64, dtype: DType) -> f32 {
    if dtype == DType::F16 {
        half::f16::from_f64(value).to_f32()
    } else {
        half::bf16::from_f64(value).to_f32()
    }
}

#[test]
fn convolution_all_heuristics_real_shape_dyadic_oracle_capture_lifetime() {
    if !available() {
        return;
    }
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for (ci, co, h, w, frames) in [
        (3, 96, 352, 640, 1),
        (3, 96, 352, 640, 4),
        (192, 384, 88, 160, 1),
        (192, 384, 88, 160, 2),
        (384, 384, 88, 160, 1),
        (384, 384, 88, 160, 2),
    ] {
        let spatial = (frames + 2) * (h + 2) * (w + 2);
        for dtype in [DType::F16, DType::BF16] {
            let av: Vec<_> = (0..ci * spatial)
                .map(|i| ((i / spatial % 7) as f32 - 3.) / 16.)
                .collect();
            let bv: Vec<_> = (0..co * ci * 27)
                .map(|i| ((i / (ci * 27) % 5) as f32 - 2. + (i / 27 % ci % 3) as f32) / 32.)
                .collect();
            let a = upload(&ctx, vec![1, ci, frames + 2, h + 2, w + 2], &av, dtype);
            let retained = a.clone();
            let b = upload(&ctx, vec![co, ci, 3, 3, 3], &bv, dtype);
            let mut out = ctx
                .allocate_output(Shape::new(vec![1, co, frames, h, w]), dtype)
                .unwrap();
            let workspace = ctx
                .allocate_output(Shape::new(vec![256 << 20]), DType::I8)
                .unwrap();
            let expected: Vec<_> = (0..co)
                .map(|o| {
                    27. * (0..ci)
                        .map(|c| av[c * spatial] as f64 * bv[(o * ci + c) * 27] as f64)
                        .sum::<f64>()
                })
                .collect();
            let check = |out: &Tensor, sign: f64| {
                let values = transfers::to_cpu(out).unwrap().to_f32_vec().unwrap();
                for (i, &v) in values.iter().enumerate() {
                    assert_eq!(
                        v,
                        rounded(sign * expected[i / (frames * h * w)], dtype),
                        "{dtype:?} index={i}"
                    );
                }
            };
            let tested = crate::ops::convolution::test_configurations(
                &ctx,
                &a,
                &b,
                [1; 3],
                &out,
                &workspace,
                |summary| {
                    eprintln!("CONVOLUTION_ORACLE {dtype:?} ci={ci} frames={frames}: {summary}");
                    check(&out, 1.);
                    Ok(())
                },
            )
            .unwrap();
            assert!(tested > 0);
            let session = ExecutionSession::with_capacity(256, 0).unwrap();
            session
                .prepare(|| convolution3d(&ctx, &a, &b, [1; 3], &mut out, &workspace))
                .unwrap();
            let graph = crate::capture(&ctx, || {
                session.run(|| convolution3d(&ctx, &a, &b, [1; 3], &mut out, &workspace))
            })
            .unwrap();
            drop(a);
            drop(b);
            drop(workspace);
            drop(session);
            let changed = upload(
                &ctx,
                vec![1, ci, frames + 2, h + 2, w + 2],
                &av.iter().map(|v| -v).collect::<Vec<_>>(),
                dtype,
            );
            transfers::copy_cuda_to_cuda(&ctx, &changed, &retained).unwrap();
            drop(changed);
            drop(retained);
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                check(&out, -1.);
            }
        }
    }
}

#[test]
fn convolution_strides_scalar_oracle_and_invalid_contracts() {
    if !available() {
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let av: Vec<_> = (0..2 * 3 * 5 * 6 * 7)
        .map(|i| (i as i32 % 13 - 6) as f32 / 16.)
        .collect();
    let bv: Vec<_> = (0..4 * 3 * 3 * 2 * 3)
        .map(|i| (i as i32 % 11 - 5) as f32 / 32.)
        .collect();
    for dtype in [DType::F16, DType::BF16] {
        let a = upload(&ctx, vec![2, 3, 5, 6, 7], &av, dtype);
        let b = upload(&ctx, vec![4, 3, 3, 2, 3], &bv, dtype);
        let mut out = ctx
            .allocate_output(Shape::new(vec![2, 4, 2, 5, 3]), dtype)
            .unwrap();
        let workspace = ctx
            .allocate_output(Shape::new(vec![32 << 20]), DType::I8)
            .unwrap();
        convolution3d(&ctx, &a, &b, [2, 1, 2], &mut out, &workspace).unwrap();
        let actual = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
        for n in 0..2 {
            for o in 0..4 {
                for t in 0..2 {
                    for h in 0..5 {
                        for w in 0..3 {
                            let mut expected = 0f64;
                            for c in 0..3 {
                                for kt in 0..3 {
                                    for kh in 0..2 {
                                        for kw in 0..3 {
                                            let ai =
                                                ((((n * 3 + c) * 5 + t * 2 + kt) * 6 + h + kh) * 7)
                                                    + w * 2
                                                    + kw;
                                            let bi = ((((o * 3 + c) * 3 + kt) * 2 + kh) * 3) + kw;
                                            expected += av[ai] as f64 * bv[bi] as f64;
                                        }
                                    }
                                }
                            }
                            let i = ((((n * 4 + o) * 2 + t) * 5 + h) * 3) + w;
                            assert_eq!(actual[i], rounded(expected, dtype));
                        }
                    }
                }
            }
        }
        assert!(convolution3d(&ctx, &a, &b, [0, 1, 1], &mut out, &workspace).is_err());
        assert!(convolution3d(&ctx, &a, &b, [1; 3], &mut out, &workspace).is_err());
        let error = crate::capture(&ctx, || {
            convolution3d(&ctx, &a, &b, [2, 1, 2], &mut out, &workspace)
        })
        .err()
        .expect("capture must not create a plan");
        assert!(error.to_string().contains("cache miss"));
        let mut alias = a.clone();
        let one = upload(&ctx, vec![3, 3, 1, 1, 1], &vec![1.; 9], dtype);
        assert!(
            convolution3d(&ctx, &a, &one, [1; 3], &mut alias, &workspace)
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );
    }
}
