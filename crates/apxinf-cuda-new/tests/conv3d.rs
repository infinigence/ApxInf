use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture, Conv3dArgs, CudaBuffer, CudaContext, PreparedConv3d, Unfold3dGeometry,
};
use half::bf16;
fn tensor(d: &[usize], values: &[f32]) -> Tensor {
    let b = CudaBuffer::alloc(values.len() * 2, 0).unwrap();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
        .collect();
    b.copy_from_host(&bytes).unwrap();
    b.as_tensor(Shape::new(d.to_vec()), DType::BF16).unwrap()
}
fn read(t: &Tensor) -> Vec<f32> {
    let mut b = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut b)
        .unwrap();
    b.chunks_exact(2)
        .map(|x| bf16::from_bits(u16::from_ne_bytes(x.try_into().unwrap())).to_f32())
        .collect()
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 13 % 17) as f32 - 8.) / 16.).collect()
}
#[test]
fn conv3d_cpu_geometry_capture_and_lifetime() {
    let ctx = CudaContext::new(0).unwrap();
    // Unequal channel counts, causal prefix, asymmetric spatial padding, stride2,
    // no history and singleton kernel, plus a budget-constrained baseline.
    for online_tune in [false, true] {
        for (d, h, n, k, stride, before, after, budget) in [
            (
                [1, 28, 28, 3],
                2,
                96,
                [3, 3, 3],
                [1; 3],
                [0, 1, 1],
                [0, 1, 1],
                512 << 20,
            ),
            (
                [3, 5, 6, 8],
                2,
                16,
                [3, 3, 3],
                [1, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                512 << 20,
            ),
            (
                [4, 5, 6, 8],
                2,
                16,
                [3, 3, 3],
                [2, 2, 2],
                [2, 0, 2],
                [1, 1, 0],
                512 << 20,
            ),
            (
                [1, 5, 6, 96],
                0,
                8,
                [3, 3, 3],
                [1, 1, 1],
                [2, 1, 1],
                [0, 1, 1],
                512 << 20,
            ),
            (
                [4, 5, 6, 3],
                1,
                7,
                [3, 3, 3],
                [2, 2, 2],
                [1, 0, 2],
                [0, 1, 0],
                512 << 20,
            ),
            (
                [2, 3, 4, 8],
                0,
                8,
                [1, 1, 1],
                [1, 1, 1],
                [0; 3],
                [0; 3],
                512 << 20,
            ),
            (
                [1, 65, 65, 8],
                0,
                8,
                [1, 1, 1],
                [1; 3],
                [0; 3],
                [0; 3],
                16 << 20,
            ),
        ] {
            let xvalues = values(d.iter().product());
            let hvalues = values(h * d[1] * d[2] * d[3]);
            let weights = values(k.iter().product::<usize>() * d[3] * n);
            let x = if budget == 16 << 20 {
                let b = CudaBuffer::alloc(xvalues.len() * 2 + 2, 0)
                    .unwrap()
                    .view(2, xvalues.len() * 2)
                    .unwrap();
                let bytes: Vec<_> = xvalues
                    .iter()
                    .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
                    .collect();
                b.copy_from_host(&bytes).unwrap();
                b.as_tensor(Shape::new(d.to_vec()), DType::BF16).unwrap()
            } else {
                tensor(&d, &xvalues)
            };
            let history = (h > 0).then(|| {
                if d == [3, 5, 6, 8] {
                    // Aligned current input with unaligned history keeps the direct
                    // convolution but uses scalar materialization.
                    let b = CudaBuffer::alloc(hvalues.len() * 2 + 2, 0)
                        .unwrap()
                        .view(2, hvalues.len() * 2)
                        .unwrap();
                    let bytes: Vec<_> = hvalues
                        .iter()
                        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
                        .collect();
                    b.copy_from_host(&bytes).unwrap();
                    b.as_tensor(Shape::new(vec![h, d[1], d[2], d[3]]), DType::BF16)
                        .unwrap()
                } else {
                    tensor(&[h, d[1], d[2], d[3]], &hvalues)
                }
            });
            let weight = tensor(&[weights.len() / n, n], &weights);
            let mut od = [0usize; 4];
            od[3] = n;
            for i in 0..3 {
                od[i] = (d[i] + if i == 0 { h } else { 0 } + before[i] + after[i] - k[i])
                    / stride[i]
                    + 1;
            }
            let mut out = tensor(&od, &vec![0.; od.iter().product()]);
            let plan = PreparedConv3d::with_policy(
                &ctx,
                Conv3dArgs {
                    input: &x,
                    history: history.as_ref(),
                    weight: &weight,
                    geometry: Unfold3dGeometry {
                        kernel: k,
                        stride,
                        padding_before: before,
                        padding_after: after,
                    },
                    out: &mut out,
                    workspace_limit: budget,
                },
                &apxinf_cuda_new::Conv3dPolicy {
                    online_tune,
                    ..Default::default()
                },
            )
            .unwrap();
            eprintln!("{:?}: {}", d, plan.summary());
            if budget == 16 << 20 {
                assert!(plan.summary().contains("lowered-gemm"));
            }
            let reference = |x: &[f32], history: &[f32]| {
                let mut expected = vec![0.; od.iter().product()];
                for t in 0..od[0] {
                    for y in 0..od[1] {
                        for z in 0..od[2] {
                            for c in 0..n {
                                let mut sum = 0f32;
                                for kt in 0..k[0] {
                                    for ky in 0..k[1] {
                                        for kz in 0..k[2] {
                                            let it =
                                                (t * stride[0] + kt) as isize - before[0] as isize;
                                            let iy =
                                                (y * stride[1] + ky) as isize - before[1] as isize;
                                            let iz =
                                                (z * stride[2] + kz) as isize - before[2] as isize;
                                            if it < 0
                                                || it >= (d[0] + h) as isize
                                                || iy < 0
                                                || iy >= d[1] as isize
                                                || iz < 0
                                                || iz >= d[2] as isize
                                            {
                                                continue;
                                            }
                                            let (source, ix) = if (it as usize) < h {
                                                (history, it as usize)
                                            } else {
                                                (x, it as usize - h)
                                            };
                                            for ic in 0..d[3] {
                                                sum += source[((ix * d[1] + iy as usize) * d[2]
                                                    + iz as usize)
                                                    * d[3]
                                                    + ic]
                                                    * weights[(((kt * k[1] + ky) * k[2] + kz)
                                                        * d[3]
                                                        + ic)
                                                        * n
                                                        + c];
                                            }
                                        }
                                    }
                                }
                                expected[((t * od[1] + y) * od[2] + z) * n + c] =
                                    bf16::from_f32(sum).to_f32();
                            }
                        }
                    }
                }
                expected
            };
            plan.enqueue().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(read(&out), reference(&xvalues, &hvalues));
            let graph = capture(&ctx, || plan.enqueue()).unwrap();
            // Graph owns prepared resources even after the Rust plan is dropped.
            drop(plan);
            let updated: Vec<_> = xvalues.iter().map(|&x| -x).collect();
            let bytes: Vec<_> = updated
                .iter()
                .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
                .collect();
            CudaBuffer::from_tensor(&x)
                .unwrap()
                .copy_from_host(&bytes)
                .unwrap();
            let updated_history: Vec<_> = hvalues.iter().map(|&x| -x).collect();
            if let Some(history) = &history {
                let bytes: Vec<_> = updated_history
                    .iter()
                    .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
                    .collect();
                CudaBuffer::from_tensor(history)
                    .unwrap()
                    .copy_from_host(&bytes)
                    .unwrap();
            }
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(read(&out), reference(&updated, &updated_history));
            }
        }
    }
}
#[test]
fn conv3d_rejects_alias_shape_and_workspace() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(&[1, 2, 3, 8], &values(48));
    let w = tensor(&[8, 8], &values(64));
    let mut out = x.clone();
    let geometry = Unfold3dGeometry {
        kernel: [1; 3],
        stride: [1; 3],
        padding_before: [0; 3],
        padding_after: [0; 3],
    };
    assert!(PreparedConv3d::new(
        &ctx,
        Conv3dArgs {
            input: &x,
            history: None,
            weight: &w,
            geometry,
            out: &mut out,
            workspace_limit: 512 << 20
        }
    )
    .is_err());
    let mut out = tensor(&[1, 2, 3, 8], &vec![0.; 48]);
    assert!(PreparedConv3d::new(
        &ctx,
        Conv3dArgs {
            input: &x,
            history: None,
            weight: &w,
            geometry,
            out: &mut out,
            workspace_limit: 0
        }
    )
    .is_err());
    let bad = Unfold3dGeometry {
        stride: [0; 3],
        ..geometry
    };
    assert!(PreparedConv3d::new(
        &ctx,
        Conv3dArgs {
            input: &x,
            history: None,
            weight: &w,
            geometry: bad,
            out: &mut out,
            workspace_limit: 512 << 20
        }
    )
    .is_err());
}

#[test]
fn conv3d_tuning_recipe_restore_and_fallback_policy() {
    use apxinf_cuda_new::Conv3dPolicy;
    // Put task-specific runs under the checkout; CI can choose its own artifact root.
    let root = std::env::var_os("APXINF_TEST_ARTIFACT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../devlocal/conv3d-tests")
        });
    let cache = root.join(format!("recipes-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    let input_values = values(2 * 3 * 4 * 8);
    let weight_values = values(8 * 8);
    let input = tensor(&[2, 3, 4, 8], &input_values);
    let weight = tensor(&[8, 8], &weight_values);
    let mut output = tensor(&[2, 3, 4, 8], &vec![0.; input_values.len()]);
    let expected: Vec<_> = input_values
        .chunks_exact(8)
        .flat_map(|row| {
            let weights = &weight_values;
            (0..8).map(move |c| {
                bf16::from_f32((0..8).map(|k| row[k] * weights[k * 8 + c]).sum()).to_f32()
            })
        })
        .collect();
    let geometry = Unfold3dGeometry {
        kernel: [1; 3],
        stride: [1; 3],
        padding_before: [0; 3],
        padding_after: [0; 3],
    };
    let mut policy = Conv3dPolicy {
        online_tune: true,
        allow_fallback: false,
        cache_dir: Some(cache.to_string_lossy().into_owned()),
    };
    let prepare = |ctx: &CudaContext, out: &mut Tensor, policy: &Conv3dPolicy| {
        PreparedConv3d::with_policy(
            ctx,
            Conv3dArgs {
                input: &input,
                history: None,
                weight: &weight,
                geometry,
                out,
                workspace_limit: 32 << 20,
            },
            policy,
        )
    };
    let ctx = CudaContext::new(0).unwrap();
    let plan = prepare(&ctx, &mut output, &policy).unwrap();
    eprintln!("cold: {}", plan.summary());
    assert!(plan.summary().contains("tuned checked="));
    // Tuning has isolated output; only explicit enqueue may publish a result.
    assert!(read(&output).iter().all(|&v| v == 0.));
    plan.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&output), expected);
    drop(plan);
    drop(ctx);
    let files: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "recipe"))
        .collect();
    assert_eq!(files.len(), 1);
    let saved = std::fs::read_to_string(&files[0]).unwrap();
    let key = saved.lines().next().unwrap();
    policy.online_tune = false;
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let plan = prepare(&ctx, &mut output, &policy).unwrap();
    assert!(plan.summary().contains("cache=hit"));
    plan.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&output), expected);
    drop(plan);
    drop(ctx);
    // Restore each stable backend configuration through the normal recipe path.
    // Unsupported library/device/geometry must fail with fallback disabled.
    for config in 0..5 {
        std::fs::write(&files[0], format!("{key}\n2 1 1 {config}\n")).unwrap();
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        match prepare(&ctx, &mut output, &policy) {
            Ok(plan) => {
                assert!(plan.summary().contains(&format!("config={config}")));
                plan.enqueue().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(read(&output), expected);
                let graph = capture(&ctx, || plan.enqueue()).unwrap();
                drop(plan);
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(read(&output), expected);
            }
            Err(error) => eprintln!("config {config} unsupported: {error}"),
        }
    }
    // Malformed, stale-version and invalid configuration all miss exactly.
    for recipe in ["broken", "2 1 999 0", "2 1 1 999"] {
        std::fs::write(&files[0], format!("{key}\n{recipe}\n")).unwrap();
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        assert!(prepare(&ctx, &mut output, &policy).is_err());
        let fallback = Conv3dPolicy {
            allow_fallback: true,
            ..policy.clone()
        };
        let plan = prepare(&ctx, &mut output, &fallback).unwrap();
        assert!(plan.summary().contains("fallback=untuned"));
        plan.enqueue().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(read(&output), expected);
        assert_eq!(
            std::fs::read_to_string(&files[0]).unwrap(),
            format!("{key}\n{recipe}\n")
        );
    }
    std::fs::remove_dir_all(cache).unwrap();
}
