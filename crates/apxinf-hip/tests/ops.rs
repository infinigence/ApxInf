//! Every implemented operator, on a real HIP device, against `CpuBackend`.
//!
//! Compiled only when the crate was built with ROCm, and requires device
//! `hip:0` at run time — the same convention as apxinf-cuda's tests, where
//! building the crate is the GPU gate.
//!
//! Tolerances are fixed here, before any result was seen, and are not to be
//! loosened to make a failing kernel pass. `CpuBackend` is F32-only, so every
//! reference is computed in F32 on inputs that are exactly representable in
//! the dtype under test.
//!
//! - F32 elementwise, RMSNorm, attention: |hip - ref| <= 1e-5 * max(1, |ref|).
//!   Differences come only from device `expf`/`rsqrtf` and summation order.
//! - F32 RoPE: 1e-4. The tests use positions past 700, where the angle is
//!   hundreds of radians and host libm and device `cosf`/`sinf` reduce the
//!   argument differently, by about angle * f32::EPSILON ≈ 4e-5.
//! - F32 GEMM: 1e-4 * max(1, |ref|), for K up to 160 with inputs in [-1, 1).
//! - BF16, any operator: 2 BF16 ulps, i.e. 2^-7 * max(1, |ref|). Each kernel
//!   rounds once, which is half an ulp; the second ulp covers the device math.
//! - Embedding and transfers: bit-exact. They move bits; they do not compute.

#![cfg(apxinf_hip_runtime)]

use apxinf_core::{
    Backend, CpuBackend, DType, Device, KvCache, NextTokenLogits, RngKey, SamplingBackend,
    Tensor, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_hip::HipBackend;
use half::bf16;

const F32_TOL: f32 = 1e-5;
const F32_GEMM_TOL: f32 = 1e-4;
const BF16_TOL: f32 = 1.0 / 128.0;

fn hip() -> HipBackend {
    HipBackend::new(0).expect("hip:0 must be available to run the device tests")
}

/// Reproducible values in [-1, 1), independent of platform.
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// A host tensor of `dtype` whose values are exactly representable in it,
/// plus the same values as F32 for the CPU reference.
fn input(dims: &[usize], dtype: DType, seed: u64) -> (Tensor, Tensor) {
    let raw = values(dims.iter().product(), seed);
    match dtype {
        DType::F32 => {
            let t = Tensor::from_f32(dims.to_vec(), &raw).unwrap();
            (t.clone(), t)
        }
        DType::BF16 => {
            let b: Vec<bf16> = raw.iter().map(|&v| bf16::from_f32(v)).collect();
            let exact: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
            (
                Tensor::from_bf16(dims.to_vec(), &b).unwrap(),
                Tensor::from_f32(dims.to_vec(), &exact).unwrap(),
            )
        }
        _ => unreachable!(),
    }
}

fn on_device(hip: &HipBackend, t: &Tensor) -> Tensor {
    hip.to_device(t).unwrap()
}

fn host_f32(hip: &HipBackend, t: &Tensor) -> Vec<f32> {
    hip.to_cpu(t).unwrap().to_f32_vec().unwrap()
}

fn tolerance(dtype: DType, f32_tol: f32) -> f32 {
    if dtype == DType::BF16 {
        BF16_TOL
    } else {
        f32_tol
    }
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(err.is_finite() || (g.is_nan() && w.is_nan()), "{what}: element {i} = {g}, want {w}");
        if err > worst.1 {
            worst = (i, err);
        }
    }
    assert!(
        worst.1 <= tol,
        "{what}: element {} = {}, want {}, relative error {:.3e} > {:.1e}",
        worst.0,
        got[worst.0],
        want[worst.0],
        worst.1,
        tol
    );
}

const DTYPES: [DType; 2] = [DType::F32, DType::BF16];

// ── Device and transfers ─────────────────────────────────────────────

#[test]
fn reports_its_device_and_capabilities() {
    let hip = hip();
    assert_eq!(hip.device(), Device::Hip(0));
    assert!(hip.device().is_gpu());
    let caps = hip.caps();
    assert!(caps.base_arch().starts_with("gfx"), "arch {}", caps.arch);
    assert!(matches!(caps.warp_size, 32 | 64), "warp size {}", caps.warp_size);
    assert!(caps.compute_units > 0);
    eprintln!(
        "hip:0 = {} ({}), wave{}, {} CUs, {} MiB, memory pools: {}",
        caps.name,
        caps.arch,
        caps.warp_size,
        caps.compute_units,
        caps.total_memory >> 20,
        caps.memory_pools
    );
}

#[test]
fn transfers_round_trip_bit_exact_for_every_dtype() {
    let hip = hip();
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::F8E4M3] {
        let bytes: Vec<u8> = (0..24 * dtype.size_in_bytes()).map(|i| (i * 37 + 11) as u8).collect();
        let host = Tensor::from_raw(vec![4, 6].into(), dtype, Device::Cpu, bytes.clone()).unwrap();
        let device = on_device(&hip, &host);
        assert_eq!(device.device(), Device::Hip(0));
        let back = hip.to_cpu(&device).unwrap();
        assert_eq!(back.shape().dims(), &[4, 6]);
        assert_eq!(back.storage().as_cpu().unwrap(), &bytes[..], "{dtype}");
    }
}

#[test]
fn host_tensors_are_rejected_by_device_operators() {
    let hip = hip();
    let host = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(hip.silu(&host).is_err());
    assert!(hip.add(&host, &host).is_err());
}

// ── Elementwise ──────────────────────────────────────────────────────

#[test]
fn elementwise_operators_match_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    for dtype in DTYPES {
        let tol = tolerance(dtype, F32_TOL);
        let (a, a_ref) = input(&[7, 33], dtype, 1);
        let (b, b_ref) = input(&[7, 33], dtype, 2);
        let (da, db) = (on_device(&hip, &a), on_device(&hip, &b));

        let silu = hip.silu(&da).unwrap();
        assert_eq!(silu.dtype(), dtype);
        assert_close(&host_f32(&hip, &silu), cpu.silu(&a_ref).unwrap().as_f32().unwrap(), tol, "silu");
        assert_close(
            &host_f32(&hip, &hip.add(&da, &db).unwrap()),
            cpu.add(&a_ref, &b_ref).unwrap().as_f32().unwrap(),
            tol,
            "add",
        );
        assert_close(
            &host_f32(&hip, &hip.mul(&da, &db).unwrap()),
            cpu.mul(&a_ref, &b_ref).unwrap().as_f32().unwrap(),
            tol,
            "mul",
        );
        assert_close(
            &host_f32(&hip, &hip.scale(&da, -2.5).unwrap()),
            cpu.scale(&a_ref, -2.5).unwrap().as_f32().unwrap(),
            tol,
            "scale",
        );
    }
}

/// apxinf-cuda reads out of bounds when operand sizes differ; this backend
/// refuses.
#[test]
fn mismatched_operands_are_rejected() {
    let hip = hip();
    let a = on_device(&hip, &input(&[4, 4], DType::F32, 3).0);
    let b = on_device(&hip, &input(&[4, 3], DType::F32, 4).0);
    assert!(hip.add(&a, &b).is_err());
    let c = on_device(&hip, &input(&[4, 4], DType::BF16, 5).0);
    assert!(hip.mul(&a, &c).is_err());
    let f16 = on_device(&hip, &Tensor::zeros(vec![4, 4], DType::F16));
    let message = hip.silu(&f16).unwrap_err().to_string();
    assert!(message.contains("silu"), "unexpected: {message}");
}

// ── RMSNorm ──────────────────────────────────────────────────────────

#[test]
fn rms_norm_matches_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    for dtype in DTYPES {
        // 300 columns exceeds the 256-thread block, so every thread loops.
        for (rows, cols) in [(5, 64), (3, 300), (1, 7)] {
            let (x, x_ref) = input(&[rows, cols], dtype, 10 + cols as u64);
            let (w, w_ref) = input(&[cols], dtype, 20 + cols as u64);
            let got = hip.rms_norm(&on_device(&hip, &x), &on_device(&hip, &w), 1e-6).unwrap();
            assert_eq!(got.shape().dims(), &[rows, cols]);
            let want = cpu.rms_norm(&x_ref, &w_ref, 1e-6).unwrap();
            assert_close(
                &host_f32(&hip, &got),
                want.as_f32().unwrap(),
                tolerance(dtype, F32_TOL),
                &format!("rms_norm {dtype} [{rows},{cols}]"),
            );
        }
    }
}

/// Qwen3's per-head QK norm reshapes [seq, heads, head_dim] to rows of
/// head_dim; every leading dimension folds into rows.
#[test]
fn rms_norm_folds_leading_dimensions_into_rows() {
    let hip = hip();
    let cpu = CpuBackend;
    let (x, x_ref) = input(&[3, 4, 16], DType::F32, 30);
    let (w, w_ref) = input(&[16], DType::F32, 31);
    let got = hip.rms_norm(&on_device(&hip, &x), &on_device(&hip, &w), 1e-6).unwrap();
    assert_eq!(got.shape().dims(), &[3, 4, 16]);
    let want = cpu.rms_norm(&x_ref.reshape(vec![12, 16]).unwrap(), &w_ref, 1e-6).unwrap();
    assert_close(&host_f32(&hip, &got), want.as_f32().unwrap(), F32_TOL, "rms_norm rank 3");
}

// ── GEMM ─────────────────────────────────────────────────────────────

#[test]
fn matmul_matches_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    for dtype in DTYPES {
        // Odd sizes on every axis, so a transposed or swapped operand cannot pass.
        for (m, k, n) in [(1, 64, 96), (37, 96, 53), (5, 160, 3)] {
            let (a, a_ref) = input(&[m, k], dtype, 40 + m as u64);
            let (b, b_ref) = input(&[k, n], dtype, 50 + n as u64);
            let got = hip.matmul(&on_device(&hip, &a), &on_device(&hip, &b)).unwrap();
            assert_eq!(got.shape().dims(), &[m, n]);
            assert_eq!(got.dtype(), dtype);
            let want = cpu.matmul(&a_ref, &b_ref).unwrap();
            assert_close(
                &host_f32(&hip, &got),
                want.as_f32().unwrap(),
                tolerance(dtype, F32_GEMM_TOL),
                &format!("matmul {dtype} {m}x{k}x{n}"),
            );
        }
    }
}

#[test]
fn matmul_folds_leading_dimensions_and_rejects_bad_shapes() {
    let hip = hip();
    let cpu = CpuBackend;
    let (a, a_ref) = input(&[2, 3, 8], DType::F32, 60);
    let (b, b_ref) = input(&[8, 5], DType::F32, 61);
    let got = hip.matmul(&on_device(&hip, &a), &on_device(&hip, &b)).unwrap();
    assert_eq!(got.shape().dims(), &[2, 3, 5]);
    let want = cpu.matmul(&a_ref.reshape(vec![6, 8]).unwrap(), &b_ref).unwrap();
    assert_close(&host_f32(&hip, &got), want.as_f32().unwrap(), F32_GEMM_TOL, "matmul rank 3");

    let wrong_k = on_device(&hip, &input(&[7, 5], DType::F32, 62).0);
    assert!(hip.matmul(&on_device(&hip, &a), &wrong_k).is_err());
}

// ── RoPE ─────────────────────────────────────────────────────────────

#[test]
fn rope_matches_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    for dtype in DTYPES {
        for (seq, heads, head_dim, offset) in [(5, 4, 64, 0u32), (1, 8, 128, 731), (3, 2, 6, 17)] {
            let (x, x_ref) = input(&[seq, heads, head_dim], dtype, 70 + offset as u64);
            let got = hip
                .rope(&on_device(&hip, &x), heads, head_dim, 1_000_000.0, offset)
                .unwrap();
            let want = cpu.rope(&x_ref, heads, head_dim, 1_000_000.0, offset).unwrap();
            // Large positions make the angle large, so the F32 tolerance has
            // to absorb cos/sin argument reduction differences.
            assert_close(
                &host_f32(&hip, &got),
                want.as_f32().unwrap(),
                tolerance(dtype, 1e-4),
                &format!("rope {dtype} seq {seq} heads {heads} dim {head_dim} offset {offset}"),
            );
        }
    }
}

// ── Embedding ────────────────────────────────────────────────────────

#[test]
fn embedding_copies_rows_bit_exact() {
    let hip = hip();
    for dtype in DTYPES {
        let (table, _) = input(&[11, 9], dtype, 80);
        let ids = [3u32, 0, 10, 3, 7];
        let got = hip.to_cpu(&hip.embedding(&on_device(&hip, &table), &ids).unwrap()).unwrap();
        assert_eq!(got.shape().dims(), &[5, 9]);
        assert_eq!(got.dtype(), dtype);
        let width = 9 * dtype.size_in_bytes();
        let table_bytes = table.storage().as_cpu().unwrap();
        let got_bytes = got.storage().as_cpu().unwrap();
        for (row, &id) in ids.iter().enumerate() {
            assert_eq!(
                &got_bytes[row * width..(row + 1) * width],
                &table_bytes[id as usize * width..(id as usize + 1) * width],
                "{dtype} row {row}"
            );
        }
    }
}

#[test]
fn embedding_rejects_out_of_range_ids() {
    let hip = hip();
    let table = on_device(&hip, &input(&[11, 9], DType::F32, 81).0);
    let message = hip.embedding(&table, &[2, 11]).unwrap_err().to_string();
    assert!(message.contains("11"), "unexpected: {message}");
}

// ── KV cache and attention ───────────────────────────────────────────

/// Append `[len, kv_heads, dim]` K/V blocks to both caches for `layer`.
fn append_both(
    hip: &HipBackend,
    hip_kv: &mut dyn KvCache,
    cpu_kv: &mut dyn KvCache,
    layer: usize,
    dims: [usize; 3],
    dtype: DType,
    seed: u64,
) {
    let (k, k_ref) = input(&dims, dtype, seed);
    let (v, v_ref) = input(&dims, dtype, seed + 1);
    hip.kv_append(hip_kv, layer, &on_device(hip, &k), &on_device(hip, &v), dims[0]).unwrap();
    CpuBackend.kv_append(cpu_kv, layer, &k_ref, &v_ref, dims[0]).unwrap();
}

/// A full request — prefill, then several decode steps — against the CPU
/// backend, for MHA and GQA. kv_len 300 exceeds the 128-key tile, so the
/// online softmax must carry its running max and sum across tiles.
#[test]
fn prefill_then_decode_matches_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    for dtype in DTYPES {
        for (heads, kv_heads, dim, prompt) in [(4, 4, 64, 7), (8, 2, 128, 300)] {
            let tol = tolerance(dtype, F32_TOL);
            let layers = 2;
            let max_seq = prompt + 8;
            let mut hip_kv = hip.create_kv_cache(layers, kv_heads, dim, max_seq);
            let mut cpu_kv = cpu.create_kv_cache(layers, kv_heads, dim, max_seq);

            for (step, q_len) in std::iter::once(prompt).chain([1, 1, 1]).enumerate() {
                let kv_len = hip_kv.seq_len() + q_len;
                for layer in 0..layers {
                    let seed = 1000 * step as u64 + 10 * layer as u64;
                    append_both(&hip, &mut *hip_kv, &mut *cpu_kv, layer, [q_len, kv_heads, dim], dtype, seed);
                    let (q, q_ref) = input(&[q_len, heads, dim], dtype, seed + 5);
                    let dq = on_device(&hip, &q);
                    let (got, want) = if q_len == 1 {
                        (
                            hip.sdpa_decode(&dq, &mut *hip_kv, layer, heads, kv_heads, dim, kv_len, max_seq),
                            cpu.sdpa_decode(&q_ref, &mut *cpu_kv, layer, heads, kv_heads, dim, kv_len, max_seq),
                        )
                    } else {
                        (
                            hip.sdpa_prefill(&dq, &mut *hip_kv, layer, heads, kv_heads, dim, kv_len, max_seq),
                            cpu.sdpa_prefill(&q_ref, &mut *cpu_kv, layer, heads, kv_heads, dim, kv_len, max_seq),
                        )
                    };
                    let got = got.unwrap();
                    assert_eq!(got.shape().dims(), &[q_len, heads * dim]);
                    assert_close(
                        &host_f32(&hip, &got),
                        want.unwrap().as_f32().unwrap(),
                        tol,
                        &format!("{dtype} H{heads}/KV{kv_heads} D{dim} step {step} layer {layer}"),
                    );
                }
                hip_kv.advance(q_len);
                cpu_kv.advance(q_len);
            }
        }
    }
}

/// A second prefill chunk on a non-empty cache: query row i sees the earlier
/// chunk plus its own prefix — the kv_offset > 0 causal case.
#[test]
fn chunked_prefill_matches_the_cpu_backend() {
    let hip = hip();
    let cpu = CpuBackend;
    let (heads, kv_heads, dim) = (4, 2, 32);
    let mut hip_kv = hip.create_kv_cache(1, kv_heads, dim, 64);
    let mut cpu_kv = cpu.create_kv_cache(1, kv_heads, dim, 64);
    for (chunk, len) in [(0u64, 9usize), (1, 6)] {
        let kv_len = hip_kv.seq_len() + len;
        append_both(&hip, &mut *hip_kv, &mut *cpu_kv, 0, [len, kv_heads, dim], DType::F32, 500 + chunk);
        let (q, q_ref) = input(&[len, heads, dim], DType::F32, 600 + chunk);
        let got = hip
            .sdpa_prefill(&on_device(&hip, &q), &mut *hip_kv, 0, heads, kv_heads, dim, kv_len, 64)
            .unwrap();
        let want = cpu
            .sdpa_prefill(&q_ref, &mut *cpu_kv, 0, heads, kv_heads, dim, kv_len, 64)
            .unwrap();
        assert_close(&host_f32(&hip, &got), want.as_f32().unwrap(), F32_TOL, &format!("chunk {chunk}"));
        hip_kv.advance(len);
        cpu_kv.advance(len);
    }
}

#[test]
fn kv_cache_rejects_overflow_and_unwritten_reads() {
    let hip = hip();
    let (kv_heads, dim) = (2, 8);
    let mut kv = hip.create_kv_cache(2, kv_heads, dim, 4);
    let block = |len| on_device(&hip, &input(&[len, kv_heads, dim], DType::F32, 700).0);

    // Five positions do not fit in four.
    assert!(hip.kv_append(&mut *kv, 0, &block(5), &block(5), 5).is_err());

    hip.kv_append(&mut *kv, 0, &block(3), &block(3), 3).unwrap();
    let q = on_device(&hip, &input(&[1, kv_heads, dim], DType::F32, 701).0);
    // Layer 0 holds three positions, so four is a read of unwritten memory.
    assert!(hip.sdpa_decode(&q, &mut *kv, 0, kv_heads, kv_heads, dim, 4, 4).is_err());
    // Layer 1 has had nothing appended.
    assert!(hip.sdpa_decode(&q, &mut *kv, 1, kv_heads, kv_heads, dim, 1, 4).is_err());
    // The cache now holds F32; a BF16 append is a different cache.
    let bf = on_device(&hip, &input(&[1, kv_heads, dim], DType::BF16, 702).0);
    assert!(hip.kv_append(&mut *kv, 1, &bf, &bf, 1).is_err());

    kv.clear().unwrap();
    assert_eq!(kv.seq_len(), 0);
    assert!(hip.sdpa_decode(&q, &mut *kv, 0, kv_heads, kv_heads, dim, 1, 4).is_err());
}

// ── Sampling ─────────────────────────────────────────────────────────

#[test]
fn greedy_sampling_from_device_logits_matches_the_host() {
    let hip = hip();
    let vocab = 1000;
    let mut logits = values(2 * vocab, 900);
    logits[vocab + 417] = 50.0; // the last row's maximum
    let host = Tensor::from_f32(vec![2, vocab], &logits).unwrap();
    let device = on_device(&hip, &host);

    let spec = TokenSamplingSpec { vocab_size: vocab, max_sequence_len: 16 };
    let params = TokenSamplingParams::default();
    let mut samplers = [
        hip.create_token_sampler(spec).unwrap(),
        CpuBackend.create_token_sampler(spec).unwrap(),
    ];
    for sampler in &mut samplers {
        sampler
            .begin(TokenSamplingInit { prompt_token_ids: &[], params: &params, rng: RngKey::default() })
            .unwrap();
    }
    let from_device = samplers[0].sample(NextTokenLogits::last(&device, vocab).unwrap()).unwrap();
    let from_host = samplers[1].sample(NextTokenLogits::last(&host, vocab).unwrap()).unwrap();
    assert_eq!(from_device.token_id, 417);
    assert_eq!(from_device.token_id, from_host.token_id);
}

#[test]
fn normal_generator_fills_device_output_in_place_like_the_host() {
    let hip = hip();
    for dtype in DTYPES {
        let output = on_device(&hip, &Tensor::zeros(vec![3, 17], dtype));
        let address = output.storage().as_gpu().unwrap().ptr();
        let mut generator = hip.create_normal_generator(output).unwrap();
        let mut host = CpuBackend.create_normal_generator(Tensor::zeros(vec![3, 17], dtype)).unwrap();
        for seed in [1u64, 2] {
            let rng = RngKey::new(seed, 0, 0);
            let got = hip.to_cpu(generator.generate(rng).unwrap()).unwrap();
            let want = host.generate(rng).unwrap();
            assert_eq!(got.storage().as_cpu(), want.storage().as_cpu(), "{dtype} seed {seed}");
        }
        // Same allocation every time: a captured graph would still see it.
        assert_eq!(generator.output().storage().as_gpu().unwrap().ptr(), address);
    }
}
