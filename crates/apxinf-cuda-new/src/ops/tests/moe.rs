//! Fixed-shape MoE helpers against an independent CPU reference: eager,
//! retained-graph replay with changed inputs, odd shapes, ties and rejection.
use apxinf_core::{DType, Shape, Tensor};
use half::bf16;

use crate::ops::{moe_route, moe_swiglu_scaled};
use crate::{capture, CudaBuffer, CudaContext, ExecutionSession};

fn upload(dims: &[usize], bytes: &[u8], offset: usize) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len() + offset, 0)
        .unwrap()
        .view(offset, bytes.len())
        .unwrap();
    buffer.copy_from_host(bytes).unwrap();
    let dtype = if bytes.len() == dims.iter().product::<usize>() * 2 {
        DType::BF16
    } else {
        DType::F32
    };
    buffer.as_tensor(Shape::new(dims.to_vec()), dtype).unwrap()
}
fn bf16_tensor(dims: &[usize], v: &[f32], offset: usize) -> Tensor {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
        .collect();
    upload(dims, &bytes, offset)
}
fn f32_tensor(dims: &[usize], v: &[f32], offset: usize) -> Tensor {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    upload(dims, &bytes, offset)
}
fn read_bytes(t: &Tensor) -> Vec<u8> {
    let mut v = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut v)
        .unwrap();
    v
}
fn read_f32(t: &Tensor) -> Vec<f32> {
    read_bytes(t)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
fn read_bf16(t: &Tensor) -> Vec<f32> {
    read_bytes(t)
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
fn rounded(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn overwrite(dst: &Tensor, src: &Tensor) {
    CudaBuffer::from_tensor(dst)
        .unwrap()
        .copy_from_host(&read_bytes(src))
        .unwrap();
}
/// Deterministic pseudo-random values in roughly [-1, 1].
fn values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            rounded(((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale)
        })
        .collect()
}

/// Straightforward f64 reference of the router contract.
fn route_reference(
    x: &[f32],
    gate: &[f32],
    bias: &[f32],
    tokens: usize,
    experts: usize,
    dim: usize,
    top_k: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f64>) {
    let mut out = vec![0.0f32; tokens * experts];
    let mut scores_all = vec![0.0f64; tokens * experts];
    for t in 0..tokens {
        let scores: Vec<f64> = (0..experts)
            .map(|e| {
                let logit: f64 = (0..dim)
                    .map(|d| x[t * dim + d] as f64 * gate[e * dim + d] as f64)
                    .sum();
                1.0 / (1.0 + (-logit).exp())
            })
            .collect();
        scores_all[t * experts..(t + 1) * experts].copy_from_slice(&scores);
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            let (va, vb) = (scores[a] + bias[a] as f64, scores[b] + bias[b] as f64);
            vb.partial_cmp(&va).unwrap().then(a.cmp(&b))
        });
        let picked = &order[..top_k];
        let total: f64 = picked.iter().map(|&e| scores[e]).sum();
        for &e in picked {
            out[t * experts + e] = (scores[e] / (total + 1e-20) * scale as f64) as f32;
        }
    }
    (out, scores_all)
}

/// Compare dense routing rows. A selection may legitimately differ only when the
/// reference's k-th and (k+1)-th candidates are within float noise of each other.
fn assert_routes_match(
    got: &[f32],
    expected: &[f32],
    scores: &[f64],
    bias: &[f32],
    tokens: usize,
    experts: usize,
    top_k: usize,
    label: &str,
) {
    for t in 0..tokens {
        let row = t * experts;
        let mut keyed: Vec<f64> = (0..experts)
            .map(|e| scores[row + e] + bias[e] as f64)
            .collect();
        keyed.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let margin = if top_k < experts {
            keyed[top_k - 1] - keyed[top_k]
        } else {
            1.0
        };
        let nonzero = |v: &[f32]| {
            (0..experts)
                .filter(|&e| v[row + e] != 0.0)
                .collect::<Vec<_>>()
        };
        if margin > 1e-4 {
            assert_eq!(
                nonzero(got),
                nonzero(expected),
                "{label} token {t} selection"
            );
            for e in 0..experts {
                let (g, r) = (got[row + e], expected[row + e]);
                assert!(
                    (g - r).abs() <= 2e-5 * r.abs().max(1.0),
                    "{label} token {t} expert {e}: {g} vs {r}"
                );
            }
        } else {
            assert_eq!(
                nonzero(got).len(),
                top_k,
                "{label} token {t} selected count"
            );
        }
    }
}

#[test]
fn moe_route_matches_cpu_reference_graph_replay_and_odd_shapes() {
    let ctx = CudaContext::new(0).unwrap();
    // (tokens, experts, dim, top_k, scale, x offset)
    let cases = [
        (51, 32, 768, 4, 4.0f32, 0),
        (1, 32, 768, 4, 4.0, 0),
        (7, 32, 768, 4, 1.0, 0),
        (5, 64, 24, 8, 2.5, 0),
        (3, 8, 8, 1, 1.0, 0),
        (4, 5, 16, 5, 1.0, 0),
        (4096, 32, 64, 4, 4.0, 0),
    ];
    for (case, &(tokens, experts, dim, top_k, scale, offset)) in cases.iter().enumerate() {
        let x = values(tokens * dim, 1 + case as u32, 1.0);
        let gate = values(experts * dim, 100 + case as u32, 0.2);
        let bias = values(experts, 200 + case as u32, 0.05);
        let xt = bf16_tensor(&[tokens, dim], &x, offset);
        let gt = bf16_tensor(&[experts, dim], &gate, 0);
        let bt = f32_tensor(&[experts], &bias, 0);
        let (expected, scores) =
            route_reference(&x, &gate, &bias, tokens, experts, dim, top_k, scale);
        let out = moe_route(&ctx, &xt, &gt, &bt, top_k, scale).unwrap();
        ctx.synchronize().unwrap();
        let label = format!("T{tokens} E{experts} D{dim} k{top_k}");
        assert_eq!(out.shape().dims(), [tokens, experts]);
        assert_routes_match(
            &read_f32(&out),
            &expected,
            &scores,
            &bias,
            tokens,
            experts,
            top_k,
            &label,
        );
        // every row's weights sum to the routed scale
        for (t, row) in read_f32(&out).chunks(experts).enumerate() {
            let total: f32 = row.iter().sum();
            assert!(
                (total - scale).abs() <= 1e-4 * scale,
                "{label} token {t} sum {total}"
            );
        }
        // Eager repeats are bit-identical (fixed reduction order).
        let again = moe_route(&ctx, &xt, &gt, &bt, top_k, scale).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(read_bytes(&out), read_bytes(&again), "{label} determinism");

        if tokens <= 51 {
            // Graph replay: capture inside a session so the output has a stable arena address.
            let session = ExecutionSession::with_capacity(1 << 20, 0).unwrap();
            let slot = std::cell::RefCell::new(None);
            let graph = crate::ops::with_session(&session, || {
                capture(&ctx, || {
                    *slot.borrow_mut() = Some(moe_route(&ctx, &xt, &gt, &bt, top_k, scale)?);
                    Ok(())
                })
            })
            .unwrap();
            let captured = slot.into_inner().unwrap();
            let new_x = values(tokens * dim, 900 + case as u32, 1.0);
            let new_gate = values(experts * dim, 950 + case as u32, 0.2);
            let new_bias = values(experts, 990 + case as u32, 0.05);
            overwrite(&xt, &bf16_tensor(&[tokens, dim], &new_x, 0));
            overwrite(&gt, &bf16_tensor(&[experts, dim], &new_gate, 0));
            overwrite(&bt, &f32_tensor(&[experts], &new_bias, 0));
            let (expected, scores) = route_reference(
                &new_x, &new_gate, &new_bias, tokens, experts, dim, top_k, scale,
            );
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_routes_match(
                    &read_f32(&captured),
                    &expected,
                    &scores,
                    &new_bias,
                    tokens,
                    experts,
                    top_k,
                    &format!("{label} replay"),
                );
            }
        }
    }
}

#[test]
fn moe_route_breaks_ties_toward_the_lower_expert() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, experts, dim, top_k) = (3, 8, 16, 3);
    // x = 0 -> every logit is 0 and every score is exactly 0.5; zero bias -> a full tie.
    let xt = bf16_tensor(&[tokens, dim], &vec![0.0; tokens * dim], 0);
    let gt = bf16_tensor(&[experts, dim], &values(experts * dim, 7, 1.0), 0);
    let bt = f32_tensor(&[experts], &vec![0.0; experts], 0);
    let out = moe_route(&ctx, &xt, &gt, &bt, top_k, 3.0).unwrap();
    ctx.synchronize().unwrap();
    for row in read_f32(&out).chunks(experts) {
        // experts 0,1,2 selected, each 0.5 / 1.5 * 3.0 = 1.0
        assert_eq!(row, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    // A bias that makes experts 5 and 6 tie for the last slot picks 5.
    let mut bias = vec![0.0; experts];
    bias[2] = 0.3;
    bias[4] = 0.3;
    bias[5] = 0.2;
    bias[6] = 0.2;
    let bt = f32_tensor(&[experts], &bias, 0);
    let out = moe_route(&ctx, &xt, &gt, &bt, 3, 3.0).unwrap();
    ctx.synchronize().unwrap();
    for row in read_f32(&out).chunks(experts) {
        assert_eq!(row, [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }
}

#[test]
fn moe_route_rejects_malformed_arguments() {
    let ctx = CudaContext::new(0).unwrap();
    let x = bf16_tensor(&[2, 16], &vec![0.5; 32], 0);
    let g = bf16_tensor(&[4, 16], &vec![0.5; 64], 0);
    let b = f32_tensor(&[4], &[0.0; 4], 0);
    assert!(moe_route(&ctx, &x, &g, &b, 2, 1.0).is_ok());
    assert!(moe_route(&ctx, &x, &g, &b, 0, 1.0).is_err(), "top_k 0");
    assert!(
        moe_route(&ctx, &x, &g, &b, 5, 1.0).is_err(),
        "top_k > experts"
    );
    assert!(
        moe_route(&ctx, &x, &g, &b, 2, f32::NAN).is_err(),
        "NaN scale"
    );
    let wide = bf16_tensor(&[2, 12], &vec![0.5; 24], 0);
    let wide_gate = bf16_tensor(&[4, 12], &vec![0.5; 48], 0);
    assert!(
        moe_route(&ctx, &wide, &wide_gate, &b, 2, 1.0).is_err(),
        "D % 8"
    );
    let other_gate = bf16_tensor(&[4, 24], &vec![0.5; 96], 0);
    assert!(
        moe_route(&ctx, &x, &other_gate, &b, 2, 1.0).is_err(),
        "gate D mismatch"
    );
    let short_bias = f32_tensor(&[3], &[0.0; 3], 0);
    assert!(
        moe_route(&ctx, &x, &g, &short_bias, 2, 1.0).is_err(),
        "bias length"
    );
    let bf16_bias = bf16_tensor(&[4], &[0.0; 4], 0);
    assert!(
        moe_route(&ctx, &x, &g, &bf16_bias, 2, 1.0).is_err(),
        "bias dtype"
    );
    let f32_x = f32_tensor(&[2, 16], &vec![0.5; 32], 0);
    assert!(
        moe_route(&ctx, &f32_x, &g, &b, 2, 1.0).is_err(),
        "input dtype"
    );
    let misaligned = bf16_tensor(&[2, 16], &vec![0.5; 32], 2);
    assert!(
        moe_route(&ctx, &misaligned, &g, &b, 2, 1.0).is_err(),
        "alignment"
    );
    let many = bf16_tensor(&[65, 16], &vec![0.5; 65 * 16], 0);
    let b65 = f32_tensor(&[65], &vec![0.0; 65], 0);
    assert!(moe_route(&ctx, &x, &many, &b65, 2, 1.0).is_err(), "E > 64");
    let tall = bf16_tensor(&[4097, 16], &vec![0.5; 4097 * 16], 0);
    assert!(moe_route(&ctx, &tall, &g, &b, 2, 1.0).is_err(), "T > 4096");
}

fn swiglu_reference(
    g: &[f32],
    u: &[f32],
    w: &[f32],
    tokens: usize,
    cols: usize,
    experts: usize,
) -> Vec<f32> {
    let block = cols / experts;
    (0..tokens * cols)
        .map(|i| {
            let (t, c) = (i / cols, i % cols);
            let g = g[i] as f64;
            ((g / (1.0 + (-g).exp())) * u[i] as f64 * w[t * experts + c / block] as f64) as f32
        })
        .collect()
}

fn assert_swiglu_close(got: &[f32], expected: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &r)) in got.iter().zip(expected).enumerate() {
        // one BF16 rounding of an f32-evaluated product: 2^-8 relative plus a tiny absolute floor
        let tolerance = r.abs() * 0.0079 + 1e-6;
        assert!((g - r).abs() <= tolerance, "{label}[{i}]: {g} vs {r}");
    }
}

#[test]
fn moe_swiglu_scaled_matches_cpu_reference_graph_replay_and_scalar_path() {
    let ctx = CudaContext::new(0).unwrap();
    // (tokens, columns, experts, gate offset): the last two force the scalar path
    // through a misaligned pointer and a block width that is not a multiple of 8.
    let cases = [
        (51, 16384, 32, 0),
        (1, 16384, 32, 0),
        (7, 256, 4, 0),
        (3, 64, 8, 0),
        (5, 96, 4, 2),
        (4, 60, 5, 0),
    ];
    for (case, &(tokens, cols, experts, offset)) in cases.iter().enumerate() {
        let n = tokens * cols;
        let g = values(n, 11 + case as u32, 4.0);
        let u = values(n, 31 + case as u32, 2.0);
        let mut w = values(tokens * experts, 51 + case as u32, 1.0);
        // sparse routing weights: most experts are exactly zero, as the router produces them
        for (i, v) in w.iter_mut().enumerate() {
            if i % 3 != 0 {
                *v = 0.0;
            }
        }
        let (gt, ut, wt) = (
            bf16_tensor(&[tokens, cols], &g, offset),
            bf16_tensor(&[tokens, cols], &u, 0),
            f32_tensor(&[tokens, experts], &w, 0),
        );
        let out = moe_swiglu_scaled(&ctx, &gt, &ut, &wt).unwrap();
        ctx.synchronize().unwrap();
        let label = format!("T{tokens} N{cols} E{experts}");
        assert_eq!(out.dtype(), DType::BF16);
        assert_swiglu_close(
            &read_bf16(&out),
            &swiglu_reference(&g, &u, &w, tokens, cols, experts),
            &label,
        );
        // unselected experts are exact zeros (including negative-zero-free)
        let block = cols / experts;
        for (i, v) in read_bf16(&out).iter().enumerate() {
            if w[(i / cols) * experts + (i % cols) / block] == 0.0 {
                assert_eq!(*v, 0.0, "{label}[{i}] must be zero");
            }
        }

        if tokens <= 7 || cols == 16384 {
            let session = ExecutionSession::with_capacity(8 << 20, 0).unwrap();
            let slot = std::cell::RefCell::new(None);
            let graph = crate::ops::with_session(&session, || {
                capture(&ctx, || {
                    *slot.borrow_mut() = Some(moe_swiglu_scaled(&ctx, &gt, &ut, &wt)?);
                    Ok(())
                })
            })
            .unwrap();
            let captured = slot.into_inner().unwrap();
            let (g2, u2) = (
                values(n, 71 + case as u32, 4.0),
                values(n, 81 + case as u32, 2.0),
            );
            let w2: Vec<f32> = values(tokens * experts, 91 + case as u32, 1.0)
                .iter()
                .map(|v| v.abs())
                .collect();
            overwrite(&gt, &bf16_tensor(&[tokens, cols], &g2, 0));
            overwrite(&ut, &bf16_tensor(&[tokens, cols], &u2, 0));
            overwrite(&wt, &f32_tensor(&[tokens, experts], &w2, 0));
            let expected = swiglu_reference(&g2, &u2, &w2, tokens, cols, experts);
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_swiglu_close(&read_bf16(&captured), &expected, &format!("{label} replay"));
            }
        }
    }
}

#[test]
fn moe_swiglu_scaled_rejects_malformed_arguments() {
    let ctx = CudaContext::new(0).unwrap();
    let g = bf16_tensor(&[2, 16], &vec![0.5; 32], 0);
    let u = bf16_tensor(&[2, 16], &vec![0.5; 32], 0);
    let w = f32_tensor(&[2, 4], &[1.0; 8], 0);
    assert!(moe_swiglu_scaled(&ctx, &g, &u, &w).is_ok());
    let w3 = f32_tensor(&[2, 3], &[1.0; 6], 0);
    assert!(moe_swiglu_scaled(&ctx, &g, &u, &w3).is_err(), "N % E != 0");
    let w_tokens = f32_tensor(&[3, 4], &[1.0; 12], 0);
    assert!(
        moe_swiglu_scaled(&ctx, &g, &u, &w_tokens).is_err(),
        "weight rows != tokens"
    );
    let up_other = bf16_tensor(&[2, 8], &vec![0.5; 16], 0);
    assert!(
        moe_swiglu_scaled(&ctx, &g, &up_other, &w).is_err(),
        "gate/up shape"
    );
    let f32_up = f32_tensor(&[2, 16], &vec![0.5; 32], 0);
    assert!(
        moe_swiglu_scaled(&ctx, &g, &f32_up, &w).is_err(),
        "up dtype"
    );
    let bf16_w = bf16_tensor(&[2, 4], &[1.0; 8], 0);
    assert!(
        moe_swiglu_scaled(&ctx, &g, &u, &bf16_w).is_err(),
        "weight dtype"
    );
    let rank3 = bf16_tensor(&[1, 2, 16], &vec![0.5; 32], 0);
    assert!(moe_swiglu_scaled(&ctx, &rank3, &u, &w).is_err(), "rank");
}

/// `cargo test --release -p apxinf-cuda-new --lib -- moe_bench --ignored --nocapture`
/// Times the real LingBot-VLA 2.0 shapes as captured-graph replays, so allocation is excluded and the
/// number is the kernel plus one graph launch, which is how the model executes them.
#[test]
#[ignore = "microbenchmark; run explicitly"]
fn moe_bench_real_shapes() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, experts, dim, top_k, cols) = (51usize, 32usize, 768usize, 4usize, 16384usize);
    let x = bf16_tensor(&[tokens, dim], &values(tokens * dim, 1, 1.0), 0);
    let gate = bf16_tensor(&[experts, dim], &values(experts * dim, 2, 0.2), 0);
    let bias = f32_tensor(&[experts], &values(experts, 3, 0.05), 0);
    let weights = f32_tensor(&[tokens, experts], &values(tokens * experts, 6, 1.0), 0);
    let g = bf16_tensor(&[tokens, cols], &values(tokens * cols, 4, 4.0), 0);
    let u = bf16_tensor(&[tokens, cols], &values(tokens * cols, 5, 2.0), 0);
    // `repeat` kernels per graph amortise the ~µs graph launch so the per-kernel time is visible.
    let repeat = 50;
    let time = |label: &str, f: &dyn Fn() -> apxinf_core::Result<()>| {
        let session = ExecutionSession::with_capacity(256 << 20, 0).unwrap();
        let graph = crate::ops::with_session(&session, || {
            capture(&ctx, || {
                for _ in 0..repeat {
                    f()?;
                }
                Ok(())
            })
        })
        .unwrap();
        for _ in 0..5 {
            graph.replay().unwrap();
        }
        ctx.synchronize().unwrap();
        let mut micros = Vec::new();
        for _ in 0..100 {
            let start = std::time::Instant::now();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            micros.push(start.elapsed().as_secs_f64() * 1e6 / repeat as f64);
        }
        micros.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "BENCH {label}: median {:.2} us/call  p10 {:.2}  p90 {:.2}  (graph of {repeat})",
            micros[50], micros[10], micros[90]
        );
    };
    time("moe_route T51 E32 D768 k4", &|| {
        moe_route(&ctx, &x, &gate, &bias, top_k, 4.0).map(|_| ())
    });
    time("moe_swiglu_scaled T51 N16384 E32", &|| {
        moe_swiglu_scaled(&ctx, &g, &u, &weights).map(|_| ())
    });
}
