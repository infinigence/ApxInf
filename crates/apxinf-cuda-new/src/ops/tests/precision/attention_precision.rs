//! Independent scalar reference for attention's arithmetic boundaries. f64
//! accumulation avoids sharing the GPU reduction implementation or rounding.
use super::*;
use crate::ops::attention::additive_attention::normalize_additive;
use half::{bf16, f16};

fn rounded(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => bf16::from_f32(x).to_f32(),
        DType::F16 => f16::from_f32(x).to_f32(),
        _ => x,
    }
}

fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    tq: usize,
    tk: usize,
    hq: usize,
    hk: usize,
    d: usize,
    scale: f32,
    broadcast: bool,
    dtype: DType,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.; b * tq * hq * d];
    for batch in 0..b {
        for head in 0..hq {
            for row in 0..tq {
                let qoffset = ((batch * tq + row) * hq + head) * d;
                let kh = head / (hq / hk);
                let mut scores = vec![f64::NEG_INFINITY; tk];
                for col in 0..tk {
                    if causal && col > tk - tq + row {
                        continue;
                    }
                    let koff = ((batch * tk + col) * hk + kh) * d;
                    let dot: f64 = (0..d)
                        .map(|i| q[qoffset + i] as f64 * k[koff + i] as f64)
                        .sum();
                    let mut score = (dot * scale as f64) as f32;

                    if let Some(bias) = bias {
                        score += bias[(((if broadcast { 0 } else { batch }) * hq + head) * tq
                            + row)
                            * tk
                            + col];
                    }
                    scores[col] = score as f64;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if max == f64::NEG_INFINITY {
                    continue;
                }
                let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let probs: Vec<f32> = exp
                    .iter()
                    .map(|p| {
                        let p = (p / sum) as f32;
                        p
                    })
                    .collect();
                for dim in 0..d {
                    let sum: f64 = (0..tk)
                        .map(|col| {
                            probs[col] as f64 * v[((batch * tk + col) * hk + kh) * d + dim] as f64
                        })
                        .sum();
                    out[qoffset + dim] = rounded(sum as f32, dtype);
                }
            }
        }
    }
    out
}

fn report_real_shape_timing(
    ctx: &CudaContext,
    mut norm: crate::ops::attention::contracts::Normalized,
) {
    norm.policy.cache_dir = Some(String::new());
    norm.policy.online_tune = true;
    norm.policy.allow_fallback = false;
    let shape = format!(
        "B{} Q{} K{} H{}/{} D{} dtype{}",
        norm.spec.batch,
        norm.spec.query_tokens,
        norm.spec.key_tokens,
        norm.spec.query_heads,
        norm.spec.kv_heads,
        norm.spec.head_dim,
        norm.spec.dtype
    );
    let prepared = super::super::attention_execution::prepare(ctx, norm).unwrap();
    eprintln!("attention real-shape {shape}: {}", prepared.summary());
}

fn data(n: usize, salt: usize, dtype: DType) -> Vec<f32> {
    (0..n)
        .map(|i| rounded((((i * 17 + salt) * 13 % 127) as f32 - 63.0) / 64.0, dtype))
        .collect()
}

fn input(shape: Vec<usize>, data: &[f32], dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => f32_tensor(0, shape, data),
        DType::F16 => f16_tensor(0, shape, data),
        _ => tensor(0, shape, data),
    }
}

#[test]
fn attention_f32_reduction_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    // Unlike the rational golden data, these mixed-exponent values require
    // rounding during QK accumulation and exercise cancellation when a
    // candidate changes from a serial dot product to a warp reduction.
    let random = |n: usize, mut state: u32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let x = (state >> 8) as f32 / 8_388_608.0 - 1.0;
                x * [0.125, 0.5, 1.0, 2.0][i % 4]
            })
            .collect()
    };
    for d in [128, 640] {
        let (b, tq, tk, h) = (1, 19, 37, 3);
        let qv = random(tq * h * d, 0x17a3_5b7d);
        let kv = random(tk * h * d, 0x93bc_a57f);
        let vv = random(tk * h * d, 0xef92_301b);
        let q = input(vec![b, tq, h, d], &qv, DType::F32);
        let k = input(vec![b, tk, h, d], &kv, DType::F32);
        let v = input(vec![b, tk, h, d], &vv, DType::F32);
        let mut out = zeros_tensor(0, vec![b, tq, h, d], DType::F32);
        let args = AttentionArgs::new(&q, &k, &v, &mut out);
        let expected = reference(
            &qv,
            &kv,
            &vv,
            None,
            b,
            tq,
            tk,
            h,
            h,
            d,
            args.scale,
            false,
            DType::F32,
            false,
        );
        let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
        super::super::attention_execution::validate_candidates(&ctx, &norm, &expected).unwrap();
    }
}

#[test]
fn attention_f32_real_shapes_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    // Wan video, action self-attention, action context, VAE spatial attention,
    // and a small causal GQA shape. D640 intentionally exceeds FA2's domain.
    for (b, tq, tk, hq, hk, d, causal) in [
        (1, 98, 98, 24, 24, 128, false),
        (1, 98, 129, 24, 24, 128, false),
        (1, 16, 1, 2, 2, 128, false),
        (1, 17, 7, 2, 2, 128, false),
        (1, 31, 31, 2, 2, 128, false),
        (1, 33, 33, 2, 2, 128, false),
        (1, 127, 159, 2, 2, 128, false),
        (1, 128, 160, 2, 2, 128, false),
        (1, 129, 160, 2, 2, 128, false),
        (1, 32, 161, 2, 2, 128, false),
        (1, 32, 130, 2, 2, 128, true),
        (2, 32, 130, 2, 2, 128, false),
        (2, 3, 7, 4, 4, 9, true),
        (2, 3, 7, 4, 4, 9, false),
        (1, 1, 1024, 1, 1, 7, false),
        (1, 1, 1025, 1, 1, 7, false),
        (1, 32, 130, 24, 24, 128, false),
        (1, 32, 129, 24, 24, 128, false),
        (1, 392, 392, 1, 1, 640, false),
        (2, 3, 7, 4, 2, 9, true),
    ] {
        let qv = data(b * tq * hq * d, 1, DType::F32);
        let kv = data(b * tk * hk * d, 7, DType::F32);
        let vv = data(b * tk * hk * d, 19, DType::F32);
        let q = input(vec![b, tq, hq, d], &qv, DType::F32);
        let k = input(vec![b, tk, hk, d], &kv, DType::F32);
        let v = input(vec![b, tk, hk, d], &vv, DType::F32);
        let mut out = zeros_tensor(0, vec![b, tq, hq, d], DType::F32);
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        if causal {
            args = args.causal();
        }
        let expected = reference(
            &qv,
            &kv,
            &vv,
            None,
            b,
            tq,
            tk,
            hq,
            hk,
            d,
            args.scale,
            false,
            DType::F32,
            causal,
        );
        let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
        super::super::attention_execution::validate_candidates(&ctx, &norm, &expected).unwrap();
        report_real_shape_timing(&ctx, norm);
    }
}

#[test]
fn attention_f32_cublas_respects_scratch_budget() {
    let ctx = CudaContext::new(0).unwrap();
    // D64 stays outside the zero-workspace short-D128 candidate's domain.
    let (tq, tk, h, d) = (32, 130, 24, 64);
    let q = input(
        vec![1, tq, h, d],
        &data(tq * h * d, 1, DType::F32),
        DType::F32,
    );
    let k = input(
        vec![1, tk, h, d],
        &data(tk * h * d, 7, DType::F32),
        DType::F32,
    );
    let v = input(
        vec![1, tk, h, d],
        &data(tk * h * d, 19, DType::F32),
        DType::F32,
    );
    let mut out = zeros_tensor(0, vec![1, tq, h, d], DType::F32);
    // The provider reserves the shared handle's 4 MiB workspace plus F32 P.
    let required = 4 * 1024 * 1024 + h * tq * tk * 4;
    for (budget, expected) in [
        (required, "cublas-f32-attention"),
        (required - 1, "custom-warp-attention"),
    ] {
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy.workspace_limit = budget;
        args.policy.cache_dir = Some(String::new());
        args.policy.online_tune = false;
        let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
        let execution = super::super::attention_execution::prepare(&ctx, norm).unwrap();
        assert!(
            execution.summary().contains(expected),
            "{}",
            execution.summary()
        );
        execution.enqueue_for_test().unwrap();
        ctx.synchronize().unwrap();
    }
}

#[test]
fn attention_short_f32_needs_no_global_workspace() {
    let ctx = CudaContext::new(0).unwrap();
    let (tq, tk, h, d) = (32, 130, 2, 128);
    let qv = data(tq * h * d, 1, DType::F32);
    let kv = data(tk * h * d, 7, DType::F32);
    let vv = data(tk * h * d, 19, DType::F32);
    let q = input(vec![1, tq, h, d], &qv, DType::F32);
    let k = input(vec![1, tk, h, d], &kv, DType::F32);
    let v = input(vec![1, tk, h, d], &vv, DType::F32);
    let mut out = zeros_tensor(0, vec![1, tq, h, d], DType::F32);
    let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
    let expected = reference(
        &qv,
        &kv,
        &vv,
        None,
        1,
        tq,
        tk,
        h,
        h,
        d,
        args.scale,
        false,
        DType::F32,
        false,
    );
    args.policy.workspace_limit = 0;
    args.policy.cache_dir = Some(String::new());
    args.policy.online_tune = false;
    let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
    let execution = super::super::attention_execution::prepare(&ctx, norm).unwrap();
    assert!(execution.summary().contains("custom-short-f32-attention"));
    execution.enqueue_for_test().unwrap();
    ctx.synchronize().unwrap();
    let buffer = crate::CudaBuffer::from_tensor(&out).unwrap();
    let mut bytes = vec![0; expected.len() * 4];
    buffer.copy_to_host(&mut bytes).unwrap();
    for (chunk, expected) in bytes.chunks_exact(4).zip(expected) {
        let actual = f32::from_ne_bytes(chunk.try_into().unwrap());
        assert!(actual.is_finite() && (actual - expected).abs() < 2e-6);
    }
}

#[test]
fn attention_short_f32_unaligned_bindings_fall_back() {
    use apxinf_core::Shape;

    let ctx = CudaContext::new(0).unwrap();
    let make = |tokens: usize, offset: usize| {
        let count = tokens * 128;
        let buffer = crate::CudaBuffer::alloc(offset + count * 4, 0).unwrap();
        let view = buffer.view(offset, count * 4).unwrap();
        let bytes: Vec<u8> = (0..count).flat_map(|_| 0.25f32.to_ne_bytes()).collect();
        view.copy_from_host(&bytes).unwrap();
        view.as_tensor(Shape::new(vec![1, tokens, 1, 128]), DType::F32)
            .unwrap()
    };
    for shifted in 0..4 {
        let q = make(17, if shifted == 0 { 4 } else { 0 });
        let k = make(7, if shifted == 1 { 4 } else { 0 });
        let v = make(7, if shifted == 2 { 4 } else { 0 });
        let mut out = make(17, if shifted == 3 { 4 } else { 0 });
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
        let execution = super::super::attention_execution::prepare(&ctx, norm).unwrap();
        assert!(execution.summary().contains("cublas-f32-attention"));
        execution.enqueue_for_test().unwrap();
        ctx.synchronize().unwrap();
        let mut bytes = vec![0; 17 * 128 * 4];
        crate::CudaBuffer::from_tensor(&out)
            .unwrap()
            .copy_to_host(&mut bytes)
            .unwrap();
        for chunk in bytes.chunks_exact(4) {
            let actual = f32::from_ne_bytes(chunk.try_into().unwrap());
            assert!(actual.is_finite() && (actual - 0.25).abs() < 2e-6);
        }
    }
}

#[test]
fn attention_bf16_vae_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (t, d) = (392, 640);
    let qv = data(t * d, 1, DType::BF16);
    let kv = data(t * d, 7, DType::BF16);
    let vv = data(t * d, 19, DType::BF16);
    let q = input(vec![1, t, 1, d], &qv, DType::BF16);
    let k = input(vec![1, t, 1, d], &kv, DType::BF16);
    let v = input(vec![1, t, 1, d], &vv, DType::BF16);
    let mut out = zeros_tensor(0, vec![1, t, 1, d], DType::BF16);
    let args = AttentionArgs::new(&q, &k, &v, &mut out);
    let expected = reference(
        &qv,
        &kv,
        &vv,
        None,
        1,
        t,
        t,
        1,
        1,
        d,
        args.scale,
        false,
        DType::BF16,
        false,
    );
    let norm = super::super::attention_contracts::normalize(&ctx, args).unwrap();
    super::super::attention_execution::validate_candidates(&ctx, &norm, &expected).unwrap();
    report_real_shape_timing(&ctx, norm);
}

#[test]
fn additive_attention_persisted_recipe_restores_in_new_process() {
    const CACHE_ENV: &str = "APXINF_ATTENTION_TEST_CACHE";
    const STAGE_ENV: &str = "APXINF_ATTENTION_TEST_STAGE";
    if let Ok(cache) = std::env::var(CACHE_ENV) {
        let stage = std::env::var(STAGE_ENV).unwrap();
        let ctx = CudaContext::new(0).unwrap();
        let q = input(vec![1, 2, 2, 4], &data(16, 1, DType::BF16), DType::BF16);
        let k = input(vec![1, 3, 2, 4], &data(24, 7, DType::BF16), DType::BF16);
        let v = input(vec![1, 3, 2, 4], &data(24, 19, DType::BF16), DType::BF16);
        let bias = f32_tensor(0, vec![1, 2, 2, 3], &data(12, 3, DType::F32));
        let mut out = zeros_tensor(0, vec![1, 2, 2, 4], DType::BF16);
        let mut attention = AttentionArgs::new(&q, &k, &v, &mut out);
        attention.policy.cache_dir = Some(cache);
        attention.policy.online_tune = stage == "cold" || stage == "repair";
        attention.policy.allow_fallback = false;
        let norm = normalize_additive(
            &ctx,
            AttentionBiasArgs {
                attention,
                bias: &bias,
            },
        )
        .unwrap();
        let prepared = super::super::attention_execution::prepare(&ctx, norm);
        if stage == "invalid" {
            assert!(prepared.is_err());
            return;
        }
        let prepared = prepared.unwrap();
        eprintln!("additive {stage}: {}", prepared.summary());
        if stage == "warm" {
            assert!(prepared.summary().contains("source=recipe"));
        }
        prepared.enqueue_for_test().unwrap();
        ctx.synchronize().unwrap();
        return;
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let cache = root
        .join("devlocal/wam-pr/results")
        .join(format!("attention-recipe-{}", std::process::id()));
    std::fs::create_dir_all(&cache).unwrap();
    let invoke = |stage: &str| {
        let result=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","ops::tests::precision::attention_precision::additive_attention_persisted_recipe_restores_in_new_process","--nocapture"])
            .env(CACHE_ENV,&cache).env(STAGE_ENV,stage).output().unwrap();
        assert!(
            result.status.success(),
            "stage {stage}: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    };
    invoke("cold");
    invoke("warm");
    let recipes: Vec<_> = std::fs::read_dir(&cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("recipe"))
        .collect();
    assert_eq!(recipes.len(), 1);
    std::fs::write(&recipes[0], "invalid recipe data").unwrap();
    invoke("invalid");
    invoke("repair");
    invoke("warm");
}

#[test]
fn additive_attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        {
            for broadcast in [true, false] {
                let (b, tq, tk, hq, hk, d) = (2, 3, 7, 4, 2, 9);
                let qv = data(b * tq * hq * d, 1, dtype);
                let kv = data(b * tk * hk * d, 7, dtype);
                let vv = data(b * tk * hk * d, 19, dtype);
                let mut bv = data(
                    (if broadcast { 1 } else { b }) * hq * tq * tk,
                    3,
                    DType::F32,
                );
                // Finite positional bias plus padding, and an entirely masked row.
                for row in bv.chunks_mut(tk) {
                    row[tk - 1] = f32::NEG_INFINITY;
                }
                bv[..tk].fill(f32::NEG_INFINITY);
                let q = input(vec![b, tq, hq, d], &qv, dtype);
                let k = input(vec![b, tk, hk, d], &kv, dtype);
                let v = input(vec![b, tk, hk, d], &vv, dtype);
                let bias = f32_tensor(0, vec![if broadcast { 1 } else { b }, hq, tq, tk], &bv);
                let mut out = zeros_tensor(0, vec![b, tq, hq, d], dtype);
                let mut attention = AttentionArgs::new(&q, &k, &v, &mut out).causal();
                attention.scale = 1.0;
                let expected = reference(
                    &qv,
                    &kv,
                    &vv,
                    Some(&bv),
                    b,
                    tq,
                    tk,
                    hq,
                    hk,
                    d,
                    1.0,
                    broadcast,
                    dtype,
                    true,
                );
                let norm = normalize_additive(
                    &ctx,
                    AttentionBiasArgs {
                        attention,
                        bias: &bias,
                    },
                )
                .unwrap();
                super::super::attention_execution::validate_candidates(&ctx, &norm, &expected)
                    .unwrap();
            }
        }
    }
    // Real UMT5-XXL encoder shape, F32 score/probability arithmetic.
    let (t, h, d) = (128, 64, 64);
    let qv = data(t * h * d, 1, DType::BF16);
    let kv = data(t * h * d, 7, DType::BF16);
    let vv = data(t * h * d, 19, DType::BF16);
    let bv = data(h * t * t, 3, DType::F32);
    let q = input(vec![1, t, h, d], &qv, DType::BF16);
    let k = input(vec![1, t, h, d], &kv, DType::BF16);
    let v = input(vec![1, t, h, d], &vv, DType::BF16);
    let bias = f32_tensor(0, vec![1, h, t, t], &bv);
    let mut out = zeros_tensor(0, vec![1, t, h, d], DType::BF16);
    let mut attention = AttentionArgs::new(&q, &k, &v, &mut out);
    attention.scale = 1.0;
    let expected = reference(
        &qv,
        &kv,
        &vv,
        Some(&bv),
        1,
        t,
        t,
        h,
        h,
        d,
        1.0,
        true,
        DType::BF16,
        false,
    );
    let norm = normalize_additive(
        &ctx,
        AttentionBiasArgs {
            attention,
            bias: &bias,
        },
    )
    .unwrap();
    super::super::attention_execution::validate_candidates(&ctx, &norm, &expected).unwrap();
    report_real_shape_timing(&ctx, norm);
}
