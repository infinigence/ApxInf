//! `apxinf_hip::native` operators on a real HIP device against F32 host
//! references, at pi05's geometry.
//!
//! Compiled only when the crate was built with ROCm, and requires `hip:0`.
//!
//! The references reproduce the cuda-new kernels' rounding points — in the
//! fused residual norms, `hidden` is rounded to BF16 before it is normalized —
//! so what remains is device math and summation order. Tolerances are fixed
//! here, before any result was seen, and are not to be loosened to make a
//! failing kernel pass. Errors are |hip - ref| / max(1, |ref|):
//!
//! - Elementwise, norms, gathers, RoPE: 2 BF16 ulps (2^-7). Each kernel rounds
//!   once, half an ulp; the second ulp covers device `tanhf`/`expf`/`rsqrtf`
//!   and RoPE's `sincosf` at positions past 500.
//! - GEMM and attention: 1e-2. BF16 inputs, F32 accumulation in a different
//!   order (hipBLAS tiles, online softmax), one rounding at the store.
//! - Copies and the arena: exact.

#![cfg(apxinf_hip_runtime)]

use std::sync::Arc;

use apxinf_core::{DType, Tensor};
use apxinf_hip::native::{
    self, Activation, AttentionGeometry, HipArena, HipDeviceBytes, ImageLayout, QkvRope,
};
use apxinf_hip::HipContext;
use half::bf16;

const BF16_TOL: f32 = 1.0 / 128.0;
const ACCUMULATE_TOL: f32 = 1e-2;
const EPS: f32 = 1e-6;

fn ctx() -> Arc<HipContext> {
    Arc::new(HipContext::new(0).expect("hip:0 must be available to run the device tests"))
}

/// Reproducible values in [-1, 1), independent of platform.
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn r(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// BF16-exact values in [-scale, scale): the device tensor and the same
/// values as F32 for the reference.
fn input(ctx: &HipContext, dims: &[usize], seed: u64, scale: f32) -> (Tensor, Vec<f32>) {
    let exact: Vec<f32> = values(dims.iter().product(), seed)
        .iter()
        .map(|v| r(v * scale))
        .collect();
    let b: Vec<bf16> = exact.iter().map(|&v| bf16::from_f32(v)).collect();
    let host = Tensor::from_bf16(dims.to_vec(), &b).unwrap();
    (ctx.to_device(&host).unwrap(), exact)
}

fn host(ctx: &HipContext, t: &Tensor) -> Vec<f32> {
    ctx.synchronize().unwrap();
    ctx.to_cpu(t).unwrap().to_f32_vec().unwrap()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(err.is_finite(), "{what}: element {i} = {g}, want {w}");
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

fn arena(ctx: &Arc<HipContext>) -> HipArena {
    HipArena::new(ctx, 256 << 20).unwrap()
}

fn gelu_tanh(v: f32) -> f32 {
    0.5 * v * (1.0 + (0.797_884_6 * (v + 0.044715 * v * v * v)).tanh())
}

fn rms(row: &[f32]) -> f32 {
    1.0 / (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + EPS).sqrt()
}

fn mean_inverse_std(row: &[f32]) -> (f32, f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + EPS).sqrt())
}

// ── Arena and transfers ──────────────────────────────────────────────

#[test]
fn arena_reuses_addresses_after_reset_and_refuses_overflow() {
    let ctx = ctx();
    let arena = HipArena::new(&ctx, 1 << 20).unwrap();
    let address = |t: &Tensor| t.storage().as_gpu().unwrap().ptr();
    let first: Vec<usize> = [vec![3, 5], vec![7], vec![1, 1, 9]]
        .into_iter()
        .map(|d| address(&arena.tensor(d, DType::BF16).unwrap()))
        .collect();
    assert!(first.iter().all(|a| a % 256 == 0), "alignment: {first:?}");
    arena.reset();
    let second: Vec<usize> = [vec![3, 5], vec![7], vec![1, 1, 9]]
        .into_iter()
        .map(|d| address(&arena.tensor(d, DType::BF16).unwrap()))
        .collect();
    assert_eq!(
        first, second,
        "same allocation order must give the same addresses"
    );
    assert!(
        arena.tensor(vec![1 << 20], DType::BF16).is_err(),
        "2 MiB from a 1 MiB arena"
    );
    assert!(arena.peak() <= arena.capacity());
}

#[test]
fn copy_rows_places_rows_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (a, a_host) = input(&ctx, &[3, 8], 1, 1.0);
    let (b, b_host) = input(&ctx, &[2, 8], 2, 1.0);
    let out = arena.tensor(vec![6, 8], DType::BF16).unwrap();
    native::copy_rows(&ctx, &a, &out, 0).unwrap();
    native::copy_rows(&ctx, &b, &out, 3).unwrap();
    let got = host(&ctx, &out);
    assert_eq!(&got[..24], &a_host[..]);
    assert_eq!(&got[24..40], &b_host[..]);
    assert!(
        native::copy_rows(&ctx, &b, &out, 5).is_err(),
        "rows past the end"
    );
}

// ── GEMM ─────────────────────────────────────────────────────────────

fn gemm_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let x = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += x * b[p * n + j];
            }
        }
    }
    c
}

#[test]
fn gemm_matches_the_reference_at_prefix_and_action_shapes() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // A prefix-length M with a reduced K/N, and the action expert's M = 10.
    for (m, k, n) in [(522, 512, 384), (10, 1024, 2560), (512, 1152, 72)] {
        let (a, ah) = input(&ctx, &[m, k], 10 + m as u64, 1.0);
        let (b, bh) = input(&ctx, &[k, n], 20 + n as u64, 0.05);
        let out = arena.tensor(vec![m, n], DType::BF16).unwrap();
        native::gemm(&ctx, &a, &b, &out).unwrap();
        assert_close(
            &host(&ctx, &out),
            &gemm_reference(&ah, &bh, m, k, n),
            ACCUMULATE_TOL,
            &format!("gemm {m}x{k}x{n}"),
        );
    }
    let (a, _) = input(&ctx, &[4, 8], 1, 1.0);
    let (b, _) = input(&ctx, &[9, 4], 2, 1.0);
    assert!(native::gemm(
        &ctx,
        &a,
        &b,
        &arena.tensor(vec![4, 4], DType::BF16).unwrap()
    )
    .is_err());
}

// ── Norms ────────────────────────────────────────────────────────────

#[test]
fn rms_and_layer_norm_match_the_reference() {
    let ctx = ctx();
    let arena = arena(&ctx);
    for (rows, cols) in [(522, 2048), (512, 1152)] {
        let (x, xh) = input(&ctx, &[rows, cols], 3, 2.0);
        let (w, wh) = input(&ctx, &[cols], 4, 1.0);
        let (b, bh) = input(&ctx, &[cols], 5, 0.5);
        let out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();

        native::rms_norm(&ctx, &x, &w, EPS, &out).unwrap();
        let mut want = Vec::with_capacity(rows * cols);
        for row in xh.chunks(cols) {
            let s = rms(row);
            want.extend(row.iter().zip(&wh).map(|(v, w)| v * s * w));
        }
        assert_close(&host(&ctx, &out), &want, BF16_TOL, "rms_norm");

        native::layer_norm(&ctx, &x, &w, &b, EPS, &out).unwrap();
        let mut want = Vec::with_capacity(rows * cols);
        for row in xh.chunks(cols) {
            let (mean, inv) = mean_inverse_std(row);
            want.extend(
                row.iter()
                    .enumerate()
                    .map(|(c, v)| (v - mean) * inv * wh[c] + bh[c]),
            );
        }
        assert_close(&host(&ctx, &out), &want, BF16_TOL, "layer_norm");
    }
}

#[test]
fn adaptive_rms_norm_reads_scale_then_shift() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (rows, cols) = (10, 1024);
    let (x, xh) = input(&ctx, &[rows, cols], 6, 2.0);
    // A full [3 * cols] modulation vector; the norm reads the first 2 * cols.
    let (style, sh) = input(&ctx, &[3 * cols], 7, 0.5);
    let out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    native::ada_rms_norm(&ctx, &x, &style, EPS, &out).unwrap();
    let mut want = Vec::new();
    for row in xh.chunks(cols) {
        let s = rms(row);
        want.extend(
            row.iter()
                .enumerate()
                .map(|(c, v)| v * s * (1.0 + sh[c]) + sh[cols + c]),
        );
    }
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "ada_rms_norm");
}

#[test]
fn fused_residual_norms_normalize_the_rounded_hidden() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (rows, cols) = (522, 2048);
    let (p, ph) = input(&ctx, &[rows, cols], 8, 1.0);
    let (res, rh) = input(&ctx, &[rows, cols], 9, 4.0);
    let (bias, bh) = input(&ctx, &[cols], 10, 0.25);
    let (w, wh) = input(&ctx, &[cols], 11, 1.0);
    let (nb, nbh) = input(&ctx, &[cols], 12, 0.5);
    let hidden = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    let normalized = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    let hidden_ref: Vec<f32> = (0..rows * cols)
        .map(|i| r(ph[i] + rh[i] + bh[i % cols]))
        .collect();

    native::bias_residual(&ctx, &p, Some(&bias), &res, &hidden).unwrap();
    assert_close(&host(&ctx, &hidden), &hidden_ref, BF16_TOL, "bias_residual");

    native::bias_residual_rms_norm(&ctx, &p, Some(&bias), &res, &w, EPS, &hidden, &normalized)
        .unwrap();
    let mut want = Vec::new();
    for row in hidden_ref.chunks(cols) {
        let s = rms(row);
        want.extend(row.iter().zip(&wh).map(|(v, w)| v * s * w));
    }
    assert_close(
        &host(&ctx, &hidden),
        &hidden_ref,
        BF16_TOL,
        "bias_residual_rms_norm hidden",
    );
    assert_close(
        &host(&ctx, &normalized),
        &want,
        BF16_TOL,
        "bias_residual_rms_norm normalized",
    );

    native::bias_residual_layer_norm(&ctx, &p, None, &res, &w, &nb, EPS, &hidden, &normalized)
        .unwrap();
    let no_bias: Vec<f32> = (0..rows * cols).map(|i| r(ph[i] + rh[i])).collect();
    let mut want = Vec::new();
    for row in no_bias.chunks(cols) {
        let (mean, inv) = mean_inverse_std(row);
        want.extend(
            row.iter()
                .enumerate()
                .map(|(c, v)| (v - mean) * inv * wh[c] + nbh[c]),
        );
    }
    assert_close(
        &host(&ctx, &hidden),
        &no_bias,
        BF16_TOL,
        "bias_residual_layer_norm hidden",
    );
    assert_close(
        &host(&ctx, &normalized),
        &want,
        BF16_TOL,
        "bias_residual_layer_norm normalized",
    );
}

#[test]
fn ada_gate_residual_rms_norm_gates_with_the_third_segment() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (rows, cols) = (10, 1024);
    let (p, ph) = input(&ctx, &[rows, cols], 13, 1.0);
    let (res, rh) = input(&ctx, &[rows, cols], 14, 4.0);
    let (gate, gh) = input(&ctx, &[3 * cols], 15, 1.0);
    let (norm, nh) = input(&ctx, &[3 * cols], 16, 0.5);
    let hidden = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    let normalized = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    native::ada_gate_residual_rms_norm(&ctx, &p, &res, &gate, &norm, EPS, &hidden, &normalized)
        .unwrap();
    let hidden_ref: Vec<f32> = (0..rows * cols)
        .map(|i| r(rh[i] + ph[i] * gh[2 * cols + i % cols]))
        .collect();
    let mut want = Vec::new();
    for row in hidden_ref.chunks(cols) {
        let s = rms(row);
        want.extend(
            row.iter()
                .enumerate()
                .map(|(c, v)| v * s * (1.0 + nh[c]) + nh[cols + c]),
        );
    }
    assert_close(
        &host(&ctx, &hidden),
        &hidden_ref,
        BF16_TOL,
        "ada gate hidden",
    );
    assert_close(
        &host(&ctx, &normalized),
        &want,
        BF16_TOL,
        "ada gate normalized",
    );
}

// ── Pointwise and gathers ────────────────────────────────────────────

#[test]
fn pointwise_operators_match_the_reference() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (rows, cols) = (512, 4304);
    let (x, xh) = input(&ctx, &[rows, cols], 17, 4.0);
    let (bias, bh) = input(&ctx, &[cols], 18, 0.5);
    let out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    for (activation, f) in [
        (Activation::None, (|v| v) as fn(f32) -> f32),
        (Activation::GeluTanh, gelu_tanh),
        (Activation::Silu, |v: f32| v / (1.0 + (-v).exp())),
    ] {
        native::bias_activation(&ctx, &x, Some(&bias), activation, &out).unwrap();
        let want: Vec<f32> = (0..rows * cols).map(|i| f(xh[i] + bh[i % cols])).collect();
        assert_close(
            &host(&ctx, &out),
            &want,
            BF16_TOL,
            &format!("bias_activation {activation:?}"),
        );
    }

    let inner = 4096;
    let (gu, guh) = input(&ctx, &[10, 2 * inner], 19, 3.0);
    let out = arena.tensor(vec![10, inner], DType::BF16).unwrap();
    native::geglu(&ctx, &gu, &out).unwrap();
    let want: Vec<f32> = (0..10 * inner)
        .map(|i| {
            let (row, col) = (i / inner, i % inner);
            gelu_tanh(guh[row * 2 * inner + col]) * guh[row * 2 * inner + inner + col]
        })
        .collect();
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "geglu");

    let (state, sh) = input(&ctx, &[10, 32], 20, 2.0);
    let (velocity, vh) = input(&ctx, &[10, 32], 21, 2.0);
    let out = arena.tensor(vec![10, 32], DType::BF16).unwrap();
    native::euler_update(&ctx, &state, &velocity, -0.1, &out).unwrap();
    let want: Vec<f32> = sh.iter().zip(&vh).map(|(s, v)| s - 0.1 * v).collect();
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "euler_update");
}

#[test]
fn embedding_scales_rows_by_sqrt_width_and_zeroes_unknown_ids() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (vocab, width) = (300, 2048);
    let (table, th) = input(&ctx, &[vocab, width], 22, 1.0);
    let ids: Vec<u32> = vec![0, 299, 7, 7, 150, 1_000_000];
    let bytes = HipDeviceBytes::new(&ctx, 4 * 200).unwrap();
    bytes.upload_u32(&ids).unwrap();
    let out = arena.tensor(vec![ids.len(), width], DType::BF16).unwrap();
    native::embedding(&ctx, &table, &bytes, &out).unwrap();
    let scale = (width as f32).sqrt();
    let th = &th;
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            (0..width).map(move |c| {
                if (id as usize) < vocab {
                    th[id as usize * width + c] * scale
                } else {
                    0.0
                }
            })
        })
        .collect();
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "embedding");
}

#[test]
fn position_embedding_repeats_per_view() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (per_view, cols) = (256, 1152);
    let (p, ph) = input(&ctx, &[2 * per_view, cols], 23, 1.0);
    let (pos, posh) = input(&ctx, &[per_view, cols], 24, 1.0);
    let (bias, bh) = input(&ctx, &[cols], 25, 0.5);
    let out = arena.tensor(vec![2 * per_view, cols], DType::BF16).unwrap();
    native::add_position(&ctx, &p, Some(&bias), &pos, per_view, &out).unwrap();
    let want: Vec<f32> = (0..2 * per_view * cols)
        .map(|i| ph[i] + posh[((i / cols) % per_view) * cols + i % cols] + bh[i % cols])
        .collect();
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "add_position");
}

#[test]
fn rgb_to_patches_matches_both_layouts() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (views, size, patch) = (2usize, 224usize, 14usize);
    let bytes_host: Vec<u8> = (0..views * size * size * 3)
        .map(|i| ((i * 73 + i / 11 + 19) & 0xff) as u8)
        .collect();
    let images = HipDeviceBytes::new(&ctx, bytes_host.len()).unwrap();
    images.upload(&bytes_host).unwrap();
    let per_side = size / patch;
    let width = 3 * patch * patch;
    let out = arena
        .tensor(vec![views * per_side * per_side, width], DType::BF16)
        .unwrap();
    for layout in [ImageLayout::Nhwc, ImageLayout::Nchw] {
        native::rgb_to_patches(&ctx, &images, layout, views, size, patch, &out).unwrap();
        let mut want = Vec::with_capacity(out.numel());
        for view in 0..views {
            for py in 0..per_side {
                for px in 0..per_side {
                    for channel in 0..3 {
                        for dy in 0..patch {
                            for dx in 0..patch {
                                let (y, x) = (py * patch + dy, px * patch + dx);
                                let index = match layout {
                                    ImageLayout::Nhwc => {
                                        ((view * size + y) * size + x) * 3 + channel
                                    }
                                    ImageLayout::Nchw => {
                                        ((view * 3 + channel) * size + y) * size + x
                                    }
                                };
                                want.push(bytes_host[index] as f32 / 255.0 * 2.0 - 1.0);
                            }
                        }
                    }
                }
            }
        }
        assert_close(
            &host(&ctx, &out),
            &want,
            BF16_TOL,
            &format!("rgb_to_patches {layout:?}"),
        );
    }
}

// ── QKV split and RoPE ───────────────────────────────────────────────

fn rotate(
    first: f32,
    second: f32,
    position: usize,
    pair: usize,
    half: usize,
    theta: f32,
) -> (f32, f32) {
    let angle = position as f32 * theta.powf(-(pair as f32) / half as f32);
    let (s, c) = angle.sin_cos();
    (first * c - second * s, second * c + first * s)
}

#[test]
fn split_qkv_rope_rotates_q_and_k_into_a_cache_row_offset() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // The action expert's call: 10 tokens at positions 522.., written into a
    // [532, 1, 256] cache at row 522.
    let (tokens, q_heads, kv_heads, head_dim, offset, rows) = (10, 8, 1, 256, 522, 532);
    let fused = (q_heads + 2 * kv_heads) * head_dim;
    let (qkv, qh) = input(&ctx, &[tokens, fused], 26, 2.0);
    let (bias, bh) = input(&ctx, &[fused], 27, 0.5);
    let q = arena
        .tensor(vec![tokens, q_heads, head_dim], DType::BF16)
        .unwrap();
    let k = arena
        .tensor(vec![rows, kv_heads, head_dim], DType::BF16)
        .unwrap();
    let v = arena
        .tensor(vec![rows, kv_heads, head_dim], DType::BF16)
        .unwrap();
    let rope = QkvRope {
        q_heads,
        kv_heads,
        head_dim,
        theta: 10_000.0,
        position_offset: offset,
    };
    native::split_qkv_rope(&ctx, &qkv, Some(&bias), &rope, &q, &k, &v, offset).unwrap();

    let half = head_dim / 2;
    let packed = |t: usize, column: usize| qh[t * fused + column] + bh[column];
    let (mut want_q, mut want_k, mut want_v) = (
        vec![0.0; q.numel()],
        vec![0.0; head_dim * tokens],
        vec![0.0; head_dim * tokens],
    );
    for t in 0..tokens {
        for head in 0..q_heads + 2 * kv_heads {
            let base = head * head_dim;
            for p in 0..half {
                let (a, b) = (packed(t, base + p), packed(t, base + half + p));
                if head < q_heads {
                    let (x, y) = rotate(a, b, offset + t, p, half, 10_000.0);
                    want_q[(t * q_heads + head) * head_dim + p] = x;
                    want_q[(t * q_heads + head) * head_dim + half + p] = y;
                } else if head < q_heads + kv_heads {
                    let (x, y) = rotate(a, b, offset + t, p, half, 10_000.0);
                    want_k[t * head_dim + p] = x;
                    want_k[t * head_dim + half + p] = y;
                } else {
                    want_v[t * head_dim + p] = a;
                    want_v[t * head_dim + half + p] = b;
                }
            }
        }
    }
    assert_close(&host(&ctx, &q), &want_q, BF16_TOL, "rope q");
    assert_close(
        &host(&ctx, &k)[offset * head_dim..],
        &want_k,
        BF16_TOL,
        "rope k at the cache offset",
    );
    assert_close(
        &host(&ctx, &v)[offset * head_dim..],
        &want_v,
        BF16_TOL,
        "v at the cache offset",
    );
    let short = arena
        .tensor(vec![rows - 1, kv_heads, head_dim], DType::BF16)
        .unwrap();
    assert!(
        native::split_qkv_rope(&ctx, &qkv, None, &rope, &q, &short, &v, offset).is_err(),
        "cache too short"
    );
}

#[test]
fn split_qkv_bias_splits_the_vision_projection() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (tokens, width) = (512, 1152);
    let (qkv, qh) = input(&ctx, &[tokens, 3 * width], 28, 2.0);
    let (bias, bh) = input(&ctx, &[3 * width], 29, 0.5);
    let parts: Vec<Tensor> = (0..3)
        .map(|_| arena.tensor(vec![tokens, 16, 72], DType::BF16).unwrap())
        .collect();
    native::split_qkv_bias(&ctx, &qkv, Some(&bias), &parts[0], &parts[1], &parts[2]).unwrap();
    for (which, part) in parts.iter().enumerate() {
        let want: Vec<f32> = (0..tokens * width)
            .map(|i| {
                let column = which * width + i % width;
                qh[(i / width) * 3 * width + column] + bh[column]
            })
            .collect();
        assert_close(
            &host(&ctx, part),
            &want,
            BF16_TOL,
            &format!("split part {which}"),
        );
    }
}

// ── Attention ────────────────────────────────────────────────────────

/// Reference attention with the layout and head grouping of `native::attention`.
fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &AttentionGeometry,
    rows: usize,
) -> Vec<f32> {
    let AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
    } = *g;
    let mut out = vec![0.0f32; q.len()];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for b in 0..batch {
        for i in 0..q_tokens {
            for h in 0..q_heads {
                let kv = h / (q_heads / kv_heads);
                let qb = ((b * q_tokens + i) * q_heads + h) * head_dim;
                let key = |j: usize| ((b * rows + j) * kv_heads + kv) * head_dim;
                let scores: Vec<f32> = (0..kv_tokens)
                    .map(|j| {
                        (0..head_dim)
                            .map(|d| q[qb + d] * k[key(j) + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = weights.iter().sum();
                for d in 0..head_dim {
                    out[qb + d] = (0..kv_tokens)
                        .map(|j| weights[j] * v[key(j) + d])
                        .sum::<f32>()
                        / total;
                }
            }
        }
    }
    out
}

#[test]
fn attention_matches_the_reference_for_every_pi05_shape() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // (geometry, cache rows): vision MHA per view (d = 72, two views); a
    // bidirectional MQA prefix (d = 256, 8:1); the action expert's 10 queries
    // over the prefix plus its own rows, from a cache with spare rows; and a
    // key count that is not a multiple of the 128-key tile.
    let cases = [
        (
            AttentionGeometry {
                batch: 2,
                q_tokens: 256,
                q_heads: 16,
                kv_heads: 16,
                head_dim: 72,
                kv_tokens: 256,
            },
            256,
        ),
        (
            AttentionGeometry {
                batch: 1,
                q_tokens: 200,
                q_heads: 8,
                kv_heads: 1,
                head_dim: 256,
                kv_tokens: 200,
            },
            200,
        ),
        (
            AttentionGeometry {
                batch: 1,
                q_tokens: 10,
                q_heads: 8,
                kv_heads: 1,
                head_dim: 256,
                kv_tokens: 532,
            },
            540,
        ),
        (
            AttentionGeometry {
                batch: 1,
                q_tokens: 3,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 64,
                kv_tokens: 300,
            },
            300,
        ),
    ];
    for (index, (g, rows)) in cases.iter().enumerate() {
        let (q, qh) = input(
            &ctx,
            &[g.batch, g.q_tokens, g.q_heads, g.head_dim],
            40 + index as u64,
            2.0,
        );
        let (k, kh) = input(
            &ctx,
            &[g.batch * rows, g.kv_heads, g.head_dim],
            50 + index as u64,
            2.0,
        );
        let (v, vh) = input(
            &ctx,
            &[g.batch * rows, g.kv_heads, g.head_dim],
            60 + index as u64,
            1.0,
        );
        let out = arena
            .tensor(
                vec![g.batch, g.q_tokens, g.q_heads, g.head_dim],
                DType::BF16,
            )
            .unwrap();
        native::attention(&ctx, &q, &k, &v, g, &out).unwrap();
        let want = attention_reference(&qh, &kh, &vh, g, *rows);
        assert_close(
            &host(&ctx, &out),
            &want,
            ACCUMULATE_TOL,
            &format!("attention case {index} {g:?}"),
        );
    }
    let g = AttentionGeometry {
        batch: 1,
        q_tokens: 10,
        q_heads: 8,
        kv_heads: 1,
        head_dim: 256,
        kv_tokens: 541,
    };
    let (q, _) = input(&ctx, &[1, 10, 8, 256], 1, 1.0);
    let (k, _) = input(&ctx, &[540, 1, 256], 2, 1.0);
    let out = arena.tensor(vec![1, 10, 8, 256], DType::BF16).unwrap();
    assert!(
        native::attention(&ctx, &q, &k, &k, &g, &out).is_err(),
        "more keys than the cache holds"
    );
}

// ── Contracts ────────────────────────────────────────────────────────

#[test]
fn non_bf16_and_host_operands_are_rejected() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let f32_device = ctx
        .to_device(&Tensor::from_f32(vec![4, 8], &values(32, 1)).unwrap())
        .unwrap();
    let out = arena.tensor(vec![4, 8], DType::BF16).unwrap();
    let (w, _) = input(&ctx, &[8], 2, 1.0);
    assert!(
        native::rms_norm(&ctx, &f32_device, &w, EPS, &out).is_err(),
        "f32 input"
    );
    let host_bf16 = Tensor::from_bf16(vec![4, 8], &vec![bf16::ONE; 32]).unwrap();
    assert!(
        native::rms_norm(&ctx, &host_bf16, &w, EPS, &out).is_err(),
        "host input"
    );
}
