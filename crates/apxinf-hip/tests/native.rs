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
    self, Activation, AttentionGeometry, AttentionKernel, HipArena, HipDeviceBytes, ImageLayout,
    QkvRope,
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
    mean_inverse_std_eps(row, EPS)
}

fn mean_inverse_std_eps(row: &[f32], eps: f32) -> (f32, f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, 1.0 / (var + eps).sqrt())
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

/// hipBLASLt must agree with the reference wherever it is routed to.
///
/// A different library means a different tile and reduction order, so the
/// answer differs in the last bits -- that is expected and is why the
/// tolerance is the same accumulation tolerance the hipBLAS path uses. What
/// would not be acceptable is a transposed or partially written result, which
/// at these sizes lands nowhere near the reference.
///
/// Each comparison is paired with a check that hipBLASLt actually served that
/// GEMM, and that pairing is the point. Every way hipBLASLt can be unavailable
/// -- built against a ROCm without it, handle creation refused, no algorithm
/// for the shape -- ends with hipBLAS doing the work and returning the right
/// answer, so comparing against the reference alone would pass most
/// convincingly when hipBLASLt never ran at all.
///
/// The counters are process-wide and the harness runs tests concurrently, so
/// another test's GEMMs can inflate a delta but never suppress one. The
/// assertion is therefore a lower bound: it catches a fallback that happens to
/// run alone, and catches "nothing was routed" always.
#[test]
fn hipblaslt_matches_the_reference_where_it_is_available() {
    if std::env::var("APXINF_HIP_GEMM_PROVIDER").as_deref() != Ok("lt") {
        // The provider is process-wide and read once, so this test only means
        // something in a run configured for it. CI covers both arms by
        // invoking the suite twice.
        eprintln!(
            "skipping: set APXINF_HIP_GEMM_PROVIDER=lt to exercise the hipBLASLt path"
        );
        return;
    }
    let ctx = ctx();
    let arena = arena(&ctx);

    // GR00T's dominant DiT shapes, where the measurement puts hipBLASLt ahead,
    // plus one vision shape where it is behind: being slower is allowed, being
    // wrong is not.
    let shapes = [(41, 6144, 1536), (41, 1536, 1536), (21, 2048, 1536), (512, 1024, 3072)];
    let mut routed = 0usize;
    for (m, k, n) in shapes {
        let (a, ah) = input(&ctx, &[m, k], 30 + m as u64, 1.0);
        let (b, bh) = input(&ctx, &[k, n], 40 + n as u64, 0.05);
        let out = arena.tensor(vec![m, n], DType::BF16).unwrap();

        let before = native::dispatch_counts().1;
        native::gemm(&ctx, &a, &b, &out).unwrap();
        let served_by_lt = native::dispatch_counts().1 > before;

        assert_close(
            &host(&ctx, &out),
            &gemm_reference(&ah, &bh, m, k, n),
            ACCUMULATE_TOL,
            &format!("hipBLASLt gemm {m}x{k}x{n}"),
        );
        assert!(
            served_by_lt,
            "{m}x{k}x{n} fell back to hipBLAS, so the comparison above says nothing about \
             hipBLASLt for this shape"
        );
        routed += 1;
    }
    assert_eq!(routed, shapes.len());
}

/// CK's FMHA must agree with apxinf's own kernel wherever it is routed to.
///
/// The dispatch count is asserted per call and is the point of the test. CK
/// reporting that it cannot serve a geometry is a normal outcome -- a causal
/// call has no vendored instance, and a build without `kernels/ck` has none at
/// all -- and every such case lands on apxinf's kernel and returns the right
/// answer. So a comparison on its own passes most convincingly when CK never
/// ran.
///
/// The tolerance is looser than `ACCUMULATE_TOL`. CK runs P.V on the matrix
/// cores, which measurement puts at 5.6e-3 relative to an F64 reference where
/// apxinf's kernel sits at 1.7e-3 -- about 1.4 BF16 ulp against 0.4. That gap
/// was shown not to matter end to end (emulating it moves GR00T's
/// `final_action` from 0.00235 to 0.00243 against a 0.05 gate), but it is far
/// too large for a bit-level tolerance and far too small to hide a transposed
/// or unmasked result, which is what this is guarding.
#[test]
fn ck_attention_matches_the_apxinf_kernel_where_it_is_available() {
    // Skipped, not failed, when the operator turned CK off: that is a
    // configuration, and the assertion below exists to catch CK being
    // *unexpectedly* absent -- a build without `kernels/ck`, or a geometry no
    // instance covers -- not to override a deliberate choice.
    if std::env::var("APXINF_HIP_ATTENTION").as_deref() == Ok("apxinf") {
        eprintln!("skipping: APXINF_HIP_ATTENTION=apxinf disables the provider under test");
        return;
    }
    let ctx = ctx();
    let arena = arena(&ctx);
    native::reset_attention_counts();

    // GR00T's four sites plus a bottom-right-aligned causal case.
    //
    // Vision is 2x256 because Qwen3-VL masks attention to within each frame
    // and the port expresses that as the batch dimension. The causal rows are
    // the ones that matter most here: a mask applied with the wrong alignment,
    // or not applied at all, still produces a plausible-looking tensor, and
    // only a comparison catches it.
    let cases = [
        (2usize, 256usize, 256usize, 16usize, 64usize, None, "vision"),
        (1, 149, 149, 32, 64, None, "VL"),
        (1, 41, 149, 32, 48, None, "DiT"),
        (1, 149, 149, 16, 128, Some(0), "text backbone (causal, top-left)"),
        (1, 41, 149, 16, 128, Some(108), "prefill over a cache (bottom-right)"),
    ];
    let mut served = 0usize;
    for (batch, q_tokens, kv_tokens, heads, head_dim, causal, name) in cases {
        let (q, _) = input(&ctx, &[batch, q_tokens, heads, head_dim], 7 + head_dim as u64, 1.0);
        let (k, _) = input(&ctx, &[batch, kv_tokens, heads, head_dim], 11 + heads as u64, 1.0);
        let (v, _) = input(&ctx, &[batch, kv_tokens, heads, head_dim], 13 + q_tokens as u64, 1.0);
        let geometry = native::AttentionGeometry {
            batch,
            q_tokens,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            kv_tokens,
            causal_query_start: causal,
        };
        let dims = vec![batch, q_tokens, heads, head_dim];

        let reference = arena.tensor(dims.clone(), DType::BF16).unwrap();
        let before = native::attention_counts();
        let out = arena.tensor(dims, DType::BF16).unwrap();
        native::attention(&ctx, &q, &k, &v, &geometry, &out).unwrap();
        let after = native::attention_counts();
        if after.1 == before.1 {
            eprintln!("  {name}: CK did not serve this geometry; skipping the comparison");
            continue;
        }
        served += 1;

        // apxinf's kernel on the same inputs, for the comparison.
        native::attention_apxinf_for_test(&ctx, &q, &k, &v, &geometry, &reference).unwrap();
        assert_close(
            &host(&ctx, &out),
            &host(&ctx, &reference),
            2.0e-2,
            &format!("CK attention {name} {batch}x{q_tokens}x{heads}x{head_dim}"),
        );
    }
    assert_eq!(
        served,
        cases.len(),
        "CK served {served} of {} geometries. Every one of them has a vendored instance: \
         the three non-causal ones and both causal alignments. A shortfall means an \
         instance went missing from kernels/ck/generated, and the shapes it covered are \
         silently back on the slow kernel",
        cases.len()
    );
}

/// A packed q/k/v projection must feed attention exactly the values the split
/// tensors do, bit for bit. The fused DiT path skips the split kernel, so
/// this equality is its whole numerical contract. Both calls share a geometry
/// and therefore a provider, so the comparison isolates the strided reads.
#[test]
fn packed_qkv_attention_matches_split_tensors_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // DiT self-attention's shape, a batched case that exercises the derived q
    // and kv batch strides as well as the token strides, and a causal case:
    // a mask applied to a strided read is the error a plain-value comparison
    // would catch last.
    let cases = [
        (1usize, 41usize, 32usize, 48usize, None, "DiT self"),
        (2, 37, 8, 64, None, "batched"),
        (1, 41, 32, 48, Some(0), "DiT self, causal"),
    ];
    for (batch, tokens, heads, head_dim, causal, name) in cases {
        let width = heads * head_dim;
        let rows = batch * tokens;
        let (packed, packed_host) = input(&ctx, &[rows, 3 * width], 7 + head_dim as u64, 1.0);
        // The same values, as the three separate tensors a split would have
        // produced.
        fn section_of(packed: &[f32], first: usize, rows: usize, width: usize) -> Vec<bf16> {
            (0..rows)
                .flat_map(|row| {
                    (0..width).map(move |col| {
                        bf16::from_f32(packed[row * 3 * width + first * width + col])
                    })
                })
                .collect()
        }
        let to_device =
            |v: Vec<bf16>| ctx.to_device(&Tensor::from_bf16(vec![rows, width], &v).unwrap()).unwrap();
        let q = to_device(section_of(&packed_host, 0, rows, width));
        let k = to_device(section_of(&packed_host, 1, rows, width));
        let v = to_device(section_of(&packed_host, 2, rows, width));

        let geometry = AttentionGeometry {
            batch,
            q_tokens: tokens,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            kv_tokens: tokens,
            causal_query_start: causal,
        };
        let dims = vec![batch, tokens, heads, head_dim];
        let out_packed = arena.tensor(dims.clone(), DType::BF16).unwrap();
        let out_split = arena.tensor(dims, DType::BF16).unwrap();
        native::attention_qkv_packed(&ctx, &packed, width, width, &geometry, &out_packed).unwrap();
        native::attention(&ctx, &q, &k, &v, &geometry, &out_split).unwrap();
        assert_eq!(
            host(&ctx, &out_packed),
            host(&ctx, &out_split),
            "{name}: reading the packed sections in place changed the result"
        );
    }
}

/// The fused vision split + axial rotation must produce exactly the values
/// the three-kernel sequence produces: the kernel rounds the biased
/// projection to BF16 before rotating, the rounding `split_qkv_bias`
/// performs when it writes q, k and v out. Bit equality is the fused path's
/// whole numerical contract -- the vision tower switched to it wholesale.
#[test]
fn split_qkv_axial_rope_matches_split_then_rotate_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // The tower's own shape, and a small case with a different head count, an
    // odd token count and no bias, so the null-bias branch is covered too.
    let cases = [
        (512usize, 16usize, 64usize, true, "vision tower"),
        (7, 5, 12, false, "small, no bias"),
    ];
    for (tokens, heads, head_dim, with_bias, name) in cases {
        let width = heads * head_dim;
        let (qkv, _) = input(&ctx, &[tokens, 3 * width], 41 + head_dim as u64, 1.0);
        let bias = if with_bias {
            Some(input(&ctx, &[3 * width], 43, 0.5).0)
        } else {
            None
        };
        // (h, w) ids in the spirit of compute_vision_pos_ids; any distinct
        // pairs serve, since both arms read the same table.
        let ids: Vec<u32> = (0..tokens)
            .flat_map(|t| [(t % 23) as u32, (t / 23 % 19) as u32])
            .collect();
        let positions = HipDeviceBytes::new(&ctx, ids.len() * std::mem::size_of::<u32>()).unwrap();
        positions.upload_u32(&ids).unwrap();

        let head_shape = || vec![tokens, heads, head_dim];
        let (q0, k0, v0) = (
            arena.tensor(head_shape(), DType::BF16).unwrap(),
            arena.tensor(head_shape(), DType::BF16).unwrap(),
            arena.tensor(head_shape(), DType::BF16).unwrap(),
        );
        // The sequence the tower ran before the fusion, rotating q and k in
        // place exactly as vision.rs called it.
        native::split_qkv_bias(&ctx, &qkv, bias.as_ref(), &q0, &k0, &v0).unwrap();
        native::apply_vision_2d(&ctx, &q0, &positions, heads, head_dim, 10_000.0, &q0).unwrap();
        native::apply_vision_2d(&ctx, &k0, &positions, heads, head_dim, 10_000.0, &k0).unwrap();

        // The fused kernel, into fresh buffers so a stray write cannot pass.
        let (q1, k1, v1) = (
            arena.tensor(head_shape(), DType::BF16).unwrap(),
            arena.tensor(head_shape(), DType::BF16).unwrap(),
            arena.tensor(head_shape(), DType::BF16).unwrap(),
        );
        native::split_qkv_axial_rope(
            &ctx, &qkv, bias.as_ref(), &positions, heads, head_dim, 10_000.0, &q1, &k1, &v1,
        )
        .unwrap();

        for (split, fused, what) in [(&q0, &q1, "q"), (&k0, &k1, "k"), (&v0, &v1, "v")] {
            assert_eq!(
                host(&ctx, split),
                host(&ctx, fused),
                "{name}: fused {what} differs from split + rotate"
            );
        }
    }
}

/// The fused text QK RMSNorm + mRoPE must produce the values the two-kernel
/// sequence produces: the fused kernel rounds the normalized head to BF16
/// before rotating, the rounding `rms_norm`'s store performed. The two arms
/// now differ only by reduction partitioning (the warp-per-head fused kernel
/// and the vectorized `rms_norm` each sum a row in a different lane order),
/// which moves individual outputs by at most a BF16 ulp -- the same tolerance
/// every other norm kernel's fused/unfused pair is held to. Bit equality was
/// the contract while the fusion only reordered a shared sequence; the
/// vectorization campaigns deliberately traded that for 1-ulp-level drift.
#[test]
fn rms_norm_mrope_qk_matches_rms_norm_then_apply_mrope_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // The text backbone's shape, and a small case with a different head
    // layout and an odd token count.
    let cases = [
        (156usize, 16usize, 8usize, 128usize, [8usize, 4, 4], "text backbone"),
        (11, 3, 2, 12, [1, 2, 1], "small"),
    ];
    for (tokens, q_heads, kv_heads, head_dim, sections, name) in cases {
        let (q, _) = input(&ctx, &[tokens * q_heads, head_dim], 61 + head_dim as u64, 1.0);
        let (k, _) = input(&ctx, &[tokens * kv_heads, head_dim], 63 + head_dim as u64, 1.0);
        let (qw, _) = input(&ctx, &[head_dim], 65, 0.5);
        let (kw, _) = input(&ctx, &[head_dim], 67, 0.5);
        // (t, h, w) ids per token; any distinct triplets serve, since both
        // arms read the same table.
        let ids: Vec<u32> = (0..tokens)
            .flat_map(|t| [(t % 250) as u32, (t % 120) as u32, (t / 7) as u32])
            .collect();
        let positions = HipDeviceBytes::new(&ctx, ids.len() * std::mem::size_of::<u32>()).unwrap();
        positions.upload_u32(&ids).unwrap();

        // The fused kernel.
        let (qo, ko) = (
            arena.tensor(vec![tokens, q_heads, head_dim], DType::BF16).unwrap(),
            arena.tensor(vec![tokens, kv_heads, head_dim], DType::BF16).unwrap(),
        );
        native::rms_norm_apply_mrope_qk(
            &ctx, &q, &qw, &k, &kw, &positions, q_heads, kv_heads, head_dim, EPS, 10_000.0,
            sections, &qo, &ko,
        )
        .unwrap();

        // The sequence the backbone ran before the fusion.
        let (qn, kn) = (
            arena.tensor(vec![tokens * q_heads, head_dim], DType::BF16).unwrap(),
            arena.tensor(vec![tokens * kv_heads, head_dim], DType::BF16).unwrap(),
        );
        native::rms_norm(&ctx, &q, &qw, EPS, &qn).unwrap();
        native::rms_norm(&ctx, &k, &kw, EPS, &kn).unwrap();
        let (qr, kr) = (
            arena.tensor(vec![tokens, q_heads, head_dim], DType::BF16).unwrap(),
            arena.tensor(vec![tokens, kv_heads, head_dim], DType::BF16).unwrap(),
        );
        native::apply_mrope(
            &ctx,
            &qn.reshape(vec![tokens, q_heads, head_dim]).unwrap(),
            &positions,
            q_heads,
            head_dim,
            10_000.0,
            sections,
            &qr,
        )
        .unwrap();
        native::apply_mrope(
            &ctx,
            &kn.reshape(vec![tokens, kv_heads, head_dim]).unwrap(),
            &positions,
            kv_heads,
            head_dim,
            10_000.0,
            sections,
            &kr,
        )
        .unwrap();

        for (split, fused, what) in [(&qr, &qo, "q"), (&kr, &ko, "k")] {
            assert_close(
                &host(&ctx, fused),
                &host(&ctx, split),
                BF16_TOL,
                &format!("{name}: fused {what} differs from rms_norm + apply_mrope"),
            );
        }
    }
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

    // cols % 8 != 0 takes the scalar fallback of the vectorized
    // `bias_activation`; keep it covered so the two paths cannot drift.
    let (rows_f, cols_f) = (17, 333);
    let (xf, xfh) = input(&ctx, &[rows_f, cols_f], 21, 4.0);
    let (biasf, biasfh) = input(&ctx, &[cols_f], 22, 0.5);
    let outf = arena.tensor(vec![rows_f, cols_f], DType::BF16).unwrap();
    native::bias_activation(&ctx, &xf, Some(&biasf), Activation::GeluTanh, &outf).unwrap();
    let want_f: Vec<f32> = (0..rows_f * cols_f).map(|i| gelu_tanh(xfh[i] + biasfh[i % cols_f])).collect();
    assert_close(
        &host(&ctx, &outf),
        &want_f,
        BF16_TOL,
        "bias_activation scalar fallback (cols % 8 != 0)",
    );

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
    native::embedding(&ctx, &table, &bytes, native::EmbeddingScale::SqrtWidth, &out)
        .unwrap();
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
        causal_query_start,
    } = *g;
    let mut out = vec![0.0f32; q.len()];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for b in 0..batch {
        for i in 0..q_tokens {
            for h in 0..q_heads {
                let kv = h / (q_heads / kv_heads);
                let qb = ((b * q_tokens + i) * q_heads + h) * head_dim;
                let key = |j: usize| ((b * rows + j) * kv_heads + kv) * head_dim;
                let visible = match causal_query_start {
                    None => kv_tokens,
                    Some(start) => kv_tokens.min(start + i + 1),
                };
                let scores: Vec<f32> = (0..visible)
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
                    out[qb + d] = (0..visible)
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
                causal_query_start: None,
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
                causal_query_start: None,
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
                causal_query_start: None,
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
                causal_query_start: None,
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
        let want = attention_reference(&qh, &kh, &vh, g, *rows);
        // Every kernel that can run this shape, against the same F32 reference.
        let mut kernels = vec![AttentionKernel::Reference, AttentionKernel::Ck];
        if g.head_dim == 256 {
            kernels.push(AttentionKernel::CkSplitKv { splits: 4 });
        }
        for kernel in kernels {
            let out = arena
                .tensor(
                    vec![g.batch, g.q_tokens, g.q_heads, g.head_dim],
                    DType::BF16,
                )
                .unwrap();
            let elements = native::attention_scratch_elements(g, kernel);
            let scratch = (elements > 0).then(|| arena.tensor(vec![elements], DType::F32).unwrap());
            native::attention(&ctx, &q, &k, &v, g, kernel, scratch.as_ref(), &out).unwrap();
            assert_close(
                &host(&ctx, &out),
                &want,
                ACCUMULATE_TOL,
                &format!("attention case {index} {kernel:?} {g:?}"),
            );
        }
    }
    let g = AttentionGeometry {
        batch: 1,
        q_tokens: 10,
        q_heads: 8,
        kv_heads: 1,
        head_dim: 256,
        kv_tokens: 541,
        causal_query_start: None,
    };
    let (q, _) = input(&ctx, &[1, 10, 8, 256], 1, 1.0);
    let (k, _) = input(&ctx, &[540, 1, 256], 2, 1.0);
    let out = arena.tensor(vec![1, 10, 8, 256], DType::BF16).unwrap();
    assert!(
        native::attention(&ctx, &q, &k, &k, &g, AttentionKernel::Ck, None, &out).is_err(),
        "more keys than the cache holds"
    );
    let split = AttentionKernel::CkSplitKv { splits: 4 };
    let g = AttentionGeometry {
        kv_tokens: 540,
        ..g
    };
    assert!(
        native::attention(&ctx, &q, &k, &k, &g, split, None, &out).is_err(),
        "split-KV without scratch"
    );
}

/// Forcing CK onto a geometry the vendored instances cannot resolve (head dim
/// 40 exists in no instance) must fall back to the reference kernel at run
/// time and still produce the right answer — the behaviour a non-gfx115 device
/// relies on for every call.
#[test]
fn ck_without_an_instance_falls_back_to_the_reference_kernel() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let g = AttentionGeometry {
        batch: 1,
        q_tokens: 12,
        q_heads: 4,
        kv_heads: 4,
        head_dim: 40,
        kv_tokens: 64,
    };
    let (q, qh) = input(&ctx, &[1, 12, 4, 40], 90, 2.0);
    let (k, kh) = input(&ctx, &[64, 4, 40], 91, 2.0);
    let (v, vh) = input(&ctx, &[64, 4, 40], 92, 1.0);
    let out = arena.tensor(vec![1, 12, 4, 40], DType::BF16).unwrap();
    native::attention(&ctx, &q, &k, &v, &g, AttentionKernel::Ck, None, &out).unwrap();
    let want = attention_reference(&qh, &kh, &vh, &g, 64);
    assert_close(&host(&ctx, &out), &want, ACCUMULATE_TOL, "ck fallback");
}

#[test]
fn default_attention_kernels_for_pi05() {
    let g = |batch, q_tokens, q_heads, kv_heads, head_dim, kv_tokens| AttentionGeometry {
        batch,
        q_tokens,
        q_heads,
        kv_heads,
        head_dim,
        kv_tokens,
    };
    // Vision and prefix: CK flash-attention forward.
    assert_eq!(
        native::attention_kernel("gfx1151", &g(2, 256, 16, 16, 72, 256)),
        AttentionKernel::Ck
    );
    assert_eq!(
        native::attention_kernel("gfx1151", &g(1, 522, 8, 1, 256, 522)),
        AttentionKernel::Ck
    );
    // Action expert: 10 queries over the prefix plus its own rows, split-KV.
    assert_eq!(
        native::attention_kernel("gfx1151", &g(1, 10, 8, 1, 256, 532)),
        AttentionKernel::CkSplitKv { splits: 4 }
    );
    // Head dims CK's vendored instances cannot run fall back to apxinf's kernel.
    assert_eq!(
        native::attention_kernel("gfx1151", &g(1, 10, 8, 1, 512, 64)),
        AttentionKernel::Reference
    );
    assert_eq!(
        native::attention_kernel("gfx1151", &g(1, 10, 8, 1, 60, 64)),
        AttentionKernel::Reference
    );
    // The vendored CK dispatchers gate on the device name: forward is gfx115*,
    // split-KV gfx11*. Everything else must get the kernel that runs anywhere.
    assert_eq!(
        native::attention_kernel("gfx942", &g(1, 522, 8, 1, 256, 522)),
        AttentionKernel::Reference
    );
    assert_eq!(
        native::attention_kernel("gfx942", &g(1, 10, 8, 1, 256, 532)),
        AttentionKernel::Reference
    );
    assert_eq!(
        native::attention_kernel("gfx1100", &g(1, 522, 8, 1, 256, 522)),
        AttentionKernel::Reference
    );
    assert_eq!(
        native::attention_kernel("gfx1100", &g(1, 10, 8, 1, 256, 532)),
        AttentionKernel::CkSplitKv { splits: 4 }
    );
    assert_eq!(
        native::attention_kernel("gfx1201", &g(2, 256, 16, 16, 72, 256)),
        AttentionKernel::Reference
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

// ── Qwen3-VL mRoPE ───────────────────────────────────────────────────

/// Which position axis frequency pair `pair` reads under Qwen3-VL's
/// interleaved mRoPE layout: T (0), H (1) or W (2).
fn mrope_axis(pair: usize, section_h: usize, section_w: usize) -> usize {
    match pair % 3 {
        1 if pair < section_h * 3 => 1,
        2 if pair < section_w * 3 => 2,
        _ => 0,
    }
}

#[test]
fn apply_mrope_rotates_half_split_pairs_on_the_interleaved_axes() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T N1.7's text stack: the fixture's 149 tokens, Qwen3-VL 2B's 16 Q /
    // 8 KV heads at head_dim 128, mrope_section [24, 20, 20], theta 5e6.
    let (tokens, head_dim, theta) = (149usize, 128usize, 5_000_000.0f32);
    let sections = [24usize, 20, 20];
    let half = head_dim / 2;

    // (t, h, w) per token: a text-then-image pattern, so the three axes
    // genuinely differ and a wrong axis cannot pass by coincidence.
    let mut position_ids = Vec::with_capacity(tokens * 3);
    for token in 0..tokens {
        if token < 40 {
            position_ids.extend([token as u32, token as u32, token as u32]);
        } else {
            let patch = token - 40;
            position_ids.extend([40u32, 40 + (patch / 16) as u32, 40 + (patch % 16) as u32]);
        }
    }
    let positions = HipDeviceBytes::new(&ctx, position_ids.len() * 4).unwrap();
    positions.upload_u32(&position_ids).unwrap();

    for heads in [16usize, 8] {
        let (x, xh) = input(&ctx, &[tokens, heads, head_dim], 40 + heads as u64, 2.0);
        let out = arena
            .tensor(vec![tokens, heads, head_dim], DType::BF16)
            .unwrap();
        native::apply_mrope(&ctx, &x, &positions, heads, head_dim, theta, sections, &out).unwrap();

        let mut want = vec![0.0f32; tokens * heads * head_dim];
        for token in 0..tokens {
            for head in 0..heads {
                let base = (token * heads + head) * head_dim;
                for pair in 0..half {
                    let axis = mrope_axis(pair, sections[1], sections[2]);
                    let position = position_ids[token * 3 + axis] as f32;
                    let frequency = theta.powf(-2.0 * pair as f32 / head_dim as f32);
                    let (sine, cosine) = (position * frequency).sin_cos();
                    let (low, high) = (xh[base + pair], xh[base + half + pair]);
                    want[base + pair] = low * cosine - high * sine;
                    want[base + half + pair] = high * cosine + low * sine;
                }
            }
        }
        assert_close(
            &host(&ctx, &out),
            &want,
            BF16_TOL,
            &format!("mrope heads={heads}"),
        );
    }
}

#[test]
fn apply_mrope_is_correct_in_place_and_rejects_a_broken_contract() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let (tokens, heads, head_dim, theta) = (6usize, 2usize, 16usize, 10_000.0f32);
    let sections = [2usize, 3, 3];
    let half = head_dim / 2;
    let ids: Vec<u32> = (0..tokens * 3).map(|i| (i % 7) as u32).collect();
    let positions = HipDeviceBytes::new(&ctx, ids.len() * 4).unwrap();
    positions.upload_u32(&ids).unwrap();

    // In place: a thread reads both halves of its pair before storing either,
    // so aliasing input and output is part of the contract.
    let (x, xh) = input(&ctx, &[tokens, heads, head_dim], 44, 1.0);
    native::apply_mrope(&ctx, &x, &positions, heads, head_dim, theta, sections, &x).unwrap();
    let mut want = vec![0.0f32; xh.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half {
                let axis = mrope_axis(pair, sections[1], sections[2]);
                let position = ids[token * 3 + axis] as f32;
                let frequency = theta.powf(-2.0 * pair as f32 / head_dim as f32);
                let (sine, cosine) = (position * frequency).sin_cos();
                let (low, high) = (xh[base + pair], xh[base + half + pair]);
                want[base + pair] = low * cosine - high * sine;
                want[base + half + pair] = high * cosine + low * sine;
            }
        }
    }
    assert_close(&host(&ctx, &x), &want, BF16_TOL, "mrope in place");

    let out = arena
        .tensor(vec![tokens, heads, head_dim], DType::BF16)
        .unwrap();
    // H + W may not claim more than the half-split pairs they share.
    assert!(
        native::apply_mrope(&ctx, &x, &positions, heads, head_dim, theta, [2, 5, 5], &out).is_err(),
        "sections h+w over head_dim/2"
    );
    // Too few positions for the token count.
    let short = HipDeviceBytes::new(&ctx, (tokens * 3 - 1) * 4).unwrap();
    assert!(
        native::apply_mrope(&ctx, &x, &short, heads, head_dim, theta, sections, &out).is_err(),
        "truncated positions"
    );
    // A rank-2 input is not [tokens, heads, head_dim].
    let flat = arena.tensor(vec![tokens, heads * head_dim], DType::BF16).unwrap();
    assert!(
        native::apply_mrope(&ctx, &flat, &positions, heads, head_dim, theta, sections, &out)
            .is_err(),
        "rank 2 input"
    );
}

#[test]
fn apply_vision_2d_splits_the_half_pairs_between_height_and_width() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T's vision tower: the fixture's 512 patches over two 16x16 grids,
    // Qwen3-VL 2B's 16 heads at head_dim 64, theta 10000.
    let (tokens, heads, head_dim, theta) = (512usize, 16usize, 64usize, 10_000.0f32);
    let (half, quarter) = (head_dim / 2, head_dim / 4);

    // Two 16x16 tiles after the spatial merge: (h, w) walks each grid, so the
    // two axes differ for almost every token.
    let mut position_ids = Vec::with_capacity(tokens * 2);
    for token in 0..tokens {
        let patch = token % 256;
        position_ids.extend([(patch / 16) as u32, (patch % 16) as u32]);
    }
    let positions = HipDeviceBytes::new(&ctx, position_ids.len() * 4).unwrap();
    positions.upload_u32(&position_ids).unwrap();

    let (x, xh) = input(&ctx, &[tokens, heads, head_dim], 50, 2.0);
    let out = arena
        .tensor(vec![tokens, heads, head_dim], DType::BF16)
        .unwrap();
    native::apply_vision_2d(&ctx, &x, &positions, heads, head_dim, theta, &out).unwrap();

    let mut want = vec![0.0f32; tokens * heads * head_dim];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half {
                let (axis, within) = if pair < quarter {
                    (0, pair)
                } else {
                    (1, pair - quarter)
                };
                let position = position_ids[token * 2 + axis] as f32;
                let frequency = theta.powf(-2.0 * within as f32 / half as f32);
                let (sine, cosine) = (position * frequency).sin_cos();
                let (low, high) = (xh[base + pair], xh[base + half + pair]);
                want[base + pair] = low * cosine - high * sine;
                want[base + half + pair] = low * sine + high * cosine;
            }
        }
    }
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "vision 2d rope");

    // In place must agree with the out-of-place result.
    native::apply_vision_2d(&ctx, &x, &positions, heads, head_dim, theta, &x).unwrap();
    assert_close(&host(&ctx, &x), &want, BF16_TOL, "vision 2d rope in place");

    // head_dim must split evenly between the two axes.
    let odd = arena.tensor(vec![tokens, heads, 6], DType::BF16).unwrap();
    assert!(
        native::apply_vision_2d(&ctx, &odd, &positions, heads, 6, theta, &odd).is_err(),
        "head_dim not a multiple of 4"
    );
    // Too few positions for the token count.
    let short = HipDeviceBytes::new(&ctx, (tokens * 2 - 1) * 4).unwrap();
    assert!(
        native::apply_vision_2d(&ctx, &x, &short, heads, head_dim, theta, &out).is_err(),
        "truncated positions"
    );
}

#[test]
fn split_qkv_bias_then_vision_2d_rounds_like_the_fused_cuda_kernel() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // cuda-new fuses the vision QKV split and the axial rotation into one
    // kernel that rounds the biased projection to BF16 before rotating. The
    // composed HIP pair must land on the same values: `split_qkv_bias` writes
    // that same rounded projection, and the rope then reads it back.
    let (tokens, heads, head_dim, theta) = (64usize, 16usize, 64usize, 10_000.0f32);
    let (width, half, quarter) = (heads * head_dim, head_dim / 2, head_dim / 4);
    let (qkv, qh) = input(&ctx, &[tokens, 3 * width], 51, 2.0);
    let (bias, bh) = input(&ctx, &[3 * width], 52, 0.5);
    let ids: Vec<u32> = (0..tokens * 2).map(|i| (i % 23) as u32).collect();
    let positions = HipDeviceBytes::new(&ctx, ids.len() * 4).unwrap();
    positions.upload_u32(&ids).unwrap();

    let make = || arena.tensor(vec![tokens, heads, head_dim], DType::BF16).unwrap();
    let (q, k, v) = (make(), make(), make());
    native::split_qkv_bias(&ctx, &qkv, Some(&bias), &q, &k, &v).unwrap();
    native::apply_vision_2d(&ctx, &q, &positions, heads, head_dim, theta, &q).unwrap();
    native::apply_vision_2d(&ctx, &k, &positions, heads, head_dim, theta, &k).unwrap();

    // Reference: bias add, round to BF16, then rotate -- the fused order.
    let mut want_q = vec![0.0f32; tokens * width];
    let mut want_k = vec![0.0f32; tokens * width];
    for token in 0..tokens {
        for head in 0..heads {
            for pair in 0..half {
                let (axis, within) = if pair < quarter {
                    (0, pair)
                } else {
                    (1, pair - quarter)
                };
                let position = ids[token * 2 + axis] as f32;
                let frequency = theta.powf(-2.0 * within as f32 / half as f32);
                let (sine, cosine) = (position * frequency).sin_cos();
                for (projection, want) in [(0usize, &mut want_q), (1usize, &mut want_k)] {
                    let column = projection * width + head * head_dim;
                    let row = token * 3 * width;
                    let low = r(qh[row + column + pair] + bh[column + pair]);
                    let high = r(qh[row + column + half + pair] + bh[column + half + pair]);
                    let base = (token * heads + head) * head_dim;
                    want[base + pair] = low * cosine - high * sine;
                    want[base + half + pair] = low * sine + high * cosine;
                }
            }
        }
    }
    assert_close(&host(&ctx, &q), &want_q, BF16_TOL, "composed vision q");
    assert_close(&host(&ctx, &k), &want_k, BF16_TOL, "composed vision k");
}

#[test]
fn causal_attention_masks_each_query_to_its_own_prefix() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T's text backbone prefill: the fixture's 149 tokens, Qwen3-VL 2B's
    // 16 Q heads over 8 KV heads (GQA 2:1) at head_dim 128. A full prefill has
    // query_start 0, so row i sees keys [0, i].
    let cases = [
        (149usize, 149usize, 0usize, 16usize, 8usize, 128usize),
        // A prefill appended to a cache that already holds 100 rows: the 49
        // new queries see their own prefix plus everything cached.
        (49, 149, 100, 16, 8, 128),
        // A key count that is not a multiple of the 128-key tile, and MQA.
        (37, 37, 0, 8, 1, 64),
    ];
    for (q_tokens, kv_tokens, start, q_heads, kv_heads, head_dim) in cases {
        let geometry = AttentionGeometry {
            batch: 1,
            q_tokens,
            q_heads,
            kv_heads,
            head_dim,
            kv_tokens,
            causal_query_start: Some(start),
        };
        let (q, qh) = input(&ctx, &[1, q_tokens, q_heads, head_dim], 60, 1.0);
        let (k, kh) = input(&ctx, &[1, kv_tokens, kv_heads, head_dim], 61, 1.0);
        let (v, vh) = input(&ctx, &[1, kv_tokens, kv_heads, head_dim], 62, 1.0);
        let out = arena
            .tensor(vec![1, q_tokens, q_heads, head_dim], DType::BF16)
            .unwrap();
        native::attention(&ctx, &q, &k, &v, &geometry, &out).unwrap();
        let want = attention_reference(&qh, &kh, &vh, &geometry, kv_tokens);
        assert_close(
            &host(&ctx, &out),
            &want,
            ACCUMULATE_TOL,
            &format!("causal attention q={q_tokens} kv={kv_tokens} start={start}"),
        );
    }
}

#[test]
fn causal_attention_differs_from_the_unmasked_result() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // A mask that is silently ignored would still pass a reference that also
    // ignores it, so pin the two apart: the first query row sees exactly one
    // key under the mask, and all of them without it.
    let (tokens, heads, head_dim) = (64usize, 4usize, 64usize);
    let (q, _) = input(&ctx, &[1, tokens, heads, head_dim], 63, 1.0);
    let (k, _) = input(&ctx, &[1, tokens, heads, head_dim], 64, 1.0);
    let (v, vh) = input(&ctx, &[1, tokens, heads, head_dim], 65, 1.0);
    let shape = vec![1, tokens, heads, head_dim];
    let masked = arena.tensor(shape.clone(), DType::BF16).unwrap();
    let dense = arena.tensor(shape, DType::BF16).unwrap();
    let base = AttentionGeometry {
        batch: 1,
        q_tokens: tokens,
        q_heads: heads,
        kv_heads: heads,
        head_dim,
        kv_tokens: tokens,
        causal_query_start: None,
    };
    native::attention(&ctx, &q, &k, &v, &base, &dense).unwrap();
    native::attention(
        &ctx,
        &q,
        &k,
        &v,
        &AttentionGeometry {
            causal_query_start: Some(0),
            ..base
        },
        &masked,
    )
    .unwrap();
    let (got_masked, got_dense) = (host(&ctx, &masked), host(&ctx, &dense));
    // Row 0 attends key 0 alone, so it must equal V's first row exactly.
    assert_close(
        &got_masked[..heads * head_dim],
        &vh[..heads * head_dim],
        BF16_TOL,
        "causal row 0 is V row 0",
    );
    let difference = got_masked
        .iter()
        .zip(&got_dense)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        difference > 1e-2,
        "the mask changed nothing: max|causal - dense| = {difference:.3e}"
    );
}

#[test]
fn ada_layer_norm_conditions_layer_statistics_with_scale_then_shift() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T's DiT: 41 rows (40 action tokens + 1 state) at inner dim 1536,
    // and the output head's eps 1e-6.
    for (rows, cols, eps) in [(41usize, 1536usize, 1e-6f32), (40, 1024, 1e-5)] {
        let (x, xh) = input(&ctx, &[rows, cols], 70, 2.0);
        let (style, sh) = input(&ctx, &[2 * cols], 71, 0.5);
        let out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
        native::ada_layer_norm(&ctx, &x, &style, eps, &out).unwrap();

        let mut want = Vec::with_capacity(rows * cols);
        for row in xh.chunks(cols) {
            let (mean, inverse) = mean_inverse_std_eps(row, eps);
            want.extend((0..cols).map(|c| {
                (row[c] - mean) * inverse * (1.0 + sh[c]) + sh[cols + c]
            }));
        }
        assert_close(
            &host(&ctx, &out),
            &want,
            BF16_TOL,
            &format!("ada_layer_norm {rows}x{cols}"),
        );
    }

    // The modulation must cover both segments.
    let (x, _) = input(&ctx, &[4, 64], 72, 1.0);
    let (short, _) = input(&ctx, &[64], 73, 1.0);
    let out = arena.tensor(vec![4, 64], DType::BF16).unwrap();
    assert!(
        native::ada_layer_norm(&ctx, &x, &short, 1e-5, &out).is_err(),
        "style shorter than 2 * cols"
    );
}

#[test]
fn silu_mul_and_relu_match_the_reference() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T's text MLP keeps gate and up apart: [149, 6144] each.
    let (rows, inner) = (149usize, 6144usize);
    let (gate, gh) = input(&ctx, &[rows, inner], 74, 2.0);
    let (up, uh) = input(&ctx, &[rows, inner], 75, 2.0);
    let out = arena.tensor(vec![rows, inner], DType::BF16).unwrap();
    native::silu_mul(&ctx, &gate, &up, &out).unwrap();
    let want: Vec<f32> = gh
        .iter()
        .zip(&uh)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    assert_close(&host(&ctx, &out), &want, BF16_TOL, "silu_mul");

    // Mismatched operands are rejected rather than read out of bounds.
    let (narrow, _) = input(&ctx, &[rows, inner - 1], 76, 1.0);
    assert!(
        native::silu_mul(&ctx, &gate, &narrow, &out).is_err(),
        "silu_mul shape mismatch"
    );

    // ReLU joins the activation set the DiT action decoder needs.
    let (x, xh) = input(&ctx, &[32, 1024], 77, 2.0);
    let (bias, bh) = input(&ctx, &[1024], 78, 0.5);
    let relu = arena.tensor(vec![32, 1024], DType::BF16).unwrap();
    native::bias_activation(&ctx, &x, Some(&bias), native::Activation::Relu, &relu).unwrap();
    let want: Vec<f32> = (0..xh.len())
        .map(|i| (xh[i] + bh[i % 1024]).max(0.0))
        .collect();
    assert_close(&host(&ctx, &relu), &want, BF16_TOL, "bias + relu");
}

#[test]
fn the_two_gelus_are_distinguishable_and_each_matches_its_own_reference() {
    // Qwen3-VL asks for both: the vision MLP is configured `gelu_pytorch_tanh`
    // while the patch merger is Torch's default `nn.GELU()`. Serving one where
    // the other was meant is a silent accuracy loss, so the test pins both
    // against their own formula and then proves they really do differ in BF16.
    let ctx = ctx();
    let arena = arena(&ctx);
    let (rows, cols) = (64usize, 512usize);
    let (x, xh) = input(&ctx, &[rows, cols], 101, 3.0);

    let tanh_out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    native::bias_activation(&ctx, &x, None, native::Activation::GeluTanh, &tanh_out).unwrap();
    let want_tanh: Vec<f32> = xh
        .iter()
        .map(|&v| {
            let inner = 0.797_884_56_f32 * (v + 0.044_715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    assert_close(&host(&ctx, &tanh_out), &want_tanh, BF16_TOL, "gelu_tanh");

    let erf_out = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
    native::bias_activation(&ctx, &x, None, native::Activation::GeluErf, &erf_out).unwrap();
    let want_erf: Vec<f32> = xh
        .iter()
        .map(|&v| 0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2)))
        .collect();
    assert_close(&host(&ctx, &erf_out), &want_erf, BF16_TOL, "gelu_erf");

    let (tanh_host, erf_host) = (host(&ctx, &tanh_out), host(&ctx, &erf_out));
    let differing = tanh_host
        .iter()
        .zip(&erf_host)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > rows * cols / 100,
        "the two GELUs rounded to the same BF16 for all but {differing} of {} elements, so this \
         test could not catch a swap",
        rows * cols
    );
}

/// Abramowitz & Stegun 7.1.26, good to ~1.5e-7 -- an independent formula, not
/// the kernel's own.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y as f32
}

#[test]
fn embedding_scale_is_the_callers_choice() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T reads the Qwen3-VL table unscaled; pi05 scales by sqrt(width).
    // The two differ by that factor, so a silent default would be a real bug.
    let (vocab, width) = (300usize, 2048usize);
    let (table, th) = input(&ctx, &[vocab, width], 79, 1.0);
    let ids: Vec<u32> = vec![0, 299, 7, 150, 1_000_000];
    let bytes = HipDeviceBytes::new(&ctx, ids.len() * 4).unwrap();
    bytes.upload_u32(&ids).unwrap();
    let plain = arena.tensor(vec![ids.len(), width], DType::BF16).unwrap();
    native::embedding(&ctx, &table, &bytes, native::EmbeddingScale::None, &plain).unwrap();
    let table_host = &th;
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            let row = id as usize;
            (0..width).map(move |c| {
                if row < vocab {
                    table_host[row * width + c]
                } else {
                    0.0
                }
            })
        })
        .collect();
    assert_close(&host(&ctx, &plain), &want, BF16_TOL, "embedding unscaled");

    // And the scaled contract still holds, so pi05 is unaffected.
    let scaled = arena.tensor(vec![ids.len(), width], DType::BF16).unwrap();
    native::embedding(
        &ctx,
        &table,
        &bytes,
        native::EmbeddingScale::SqrtWidth,
        &scaled,
    )
    .unwrap();
    let factor = (width as f32).sqrt();
    let want_scaled: Vec<f32> = want.iter().map(|v| v * factor).collect();
    assert_close(
        &host(&ctx, &scaled),
        &want_scaled,
        BF16_TOL,
        "embedding scaled",
    );
}

#[test]
fn row_and_column_movement_matches_the_reference() {
    let ctx = ctx();
    let arena = arena(&ctx);
    let cols = 1536usize;

    // contiguous_rows: GR00T drops the DiT's leading state row and keeps the
    // 40 action rows.
    let (decoded, dh) = input(&ctx, &[41, cols], 80, 1.0);
    let sliced = arena.tensor(vec![40, cols], DType::BF16).unwrap();
    native::copy_row_range(&ctx, &decoded, 1, 40, &sliced, 0).unwrap();
    assert_close(
        &host(&ctx, &sliced),
        &dh[cols..],
        0.0,
        "row slice is bit exact",
    );

    // gather_rows: the backbone's image / non-image row split.
    let (features, fh) = input(&ctx, &[149, cols], 81, 1.0);
    let picked: Vec<u32> = (0..149u32).filter(|i| i % 3 == 1).collect();
    let indices = HipDeviceBytes::new(&ctx, picked.len() * 4).unwrap();
    indices.upload_u32(&picked).unwrap();
    let gathered = arena
        .tensor(vec![picked.len(), cols], DType::BF16)
        .unwrap();
    native::gather_rows(&ctx, &features, &indices, &gathered).unwrap();
    let want: Vec<f32> = picked
        .iter()
        .flat_map(|&row| fh[row as usize * cols..(row as usize + 1) * cols].to_vec())
        .collect();
    assert_close(&host(&ctx, &gathered), &want, 0.0, "gather is bit exact");

    // scatter_rows, replacing then accumulating into the same rows.
    let destination = arena.tensor(vec![149, cols], DType::BF16).unwrap();
    native::copy_rows(&ctx, &features, &destination, 0).unwrap();
    native::scatter_rows(&ctx, &destination, &indices, &gathered, false).unwrap();
    assert_close(
        &host(&ctx, &destination),
        &fh,
        0.0,
        "scatter of the gathered rows restores the source",
    );
    native::scatter_rows(&ctx, &destination, &indices, &gathered, true).unwrap();
    let mut want_added = fh.clone();
    for &row in &picked {
        let base = row as usize * cols;
        for c in 0..cols {
            want_added[base + c] = r(want_added[base + c] + fh[base + c]);
        }
    }
    assert_close(
        &host(&ctx, &destination),
        &want_added,
        BF16_TOL,
        "accumulating scatter",
    );

    // copy_columns: the action encoder appends the step's time embedding.
    let (action, ah) = input(&ctx, &[40, 1024], 82, 1.0);
    let (time, th) = input(&ctx, &[40, 512], 83, 1.0);
    let joined = arena.tensor(vec![40, 1536], DType::BF16).unwrap();
    native::copy_columns(&ctx, &action, &joined, 0).unwrap();
    native::copy_columns(&ctx, &time, &joined, 1024).unwrap();
    let mut want_joined = Vec::with_capacity(40 * 1536);
    for row in 0..40 {
        want_joined.extend_from_slice(&ah[row * 1024..(row + 1) * 1024]);
        want_joined.extend_from_slice(&th[row * 512..(row + 1) * 512]);
    }
    assert_close(
        &host(&ctx, &joined),
        &want_joined,
        0.0,
        "column concatenation is bit exact",
    );

    // Out-of-range ranges and windows are refused, not clamped.
    assert!(
        native::copy_row_range(&ctx, &decoded, 2, 40, &sliced, 0).is_err(),
        "source range past the end"
    );
    assert!(
        native::copy_columns(&ctx, &time, &joined, 1025).is_err(),
        "column window past the end"
    );
    let few = HipDeviceBytes::new(&ctx, (picked.len() - 1) * 4).unwrap();
    assert!(
        native::gather_rows(&ctx, &features, &few, &gathered).is_err(),
        "truncated indices"
    );
}

/// The fused bias/residual/adaptive-LayerNorm must equal the pair it replaces,
/// to the bit.
///
/// Not "close": the fused kernel exists to remove a launch, not to compute
/// anything differently, and it is written to round `hidden` to BF16 and read
/// it back exactly where the separate path does. A difference here means that
/// boundary moved, which is a change of model that an end-to-end cosine of
/// 0.9999 would absorb without reporting.
#[test]
fn the_fused_adaptive_layer_norm_matches_the_separate_pair_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);
    // GR00T's DiT shape, and a second with an odd row count so the reduction
    // is exercised off a round grid.
    for (rows, cols) in [(41usize, 1536usize), (7, 1024)] {
        let (projection, _) = input(&ctx, &[rows, cols], 3 + rows as u64, 1.0);
        let (bias, _) = input(&ctx, &[cols], 5, 0.5);
        let (residual, _) = input(&ctx, &[rows, cols], 7 + cols as u64, 1.0);
        let (style, _) = input(&ctx, &[2 * cols], 11, 0.25);

        let hidden_pair = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
        let normalized_pair = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
        native::bias_residual(&ctx, &projection, Some(&bias), &residual, &hidden_pair).unwrap();
        native::ada_layer_norm(&ctx, &hidden_pair, &style, 1e-5, &normalized_pair).unwrap();

        let hidden_fused = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
        let normalized_fused = arena.tensor(vec![rows, cols], DType::BF16).unwrap();
        native::bias_residual_ada_layer_norm(
            &ctx,
            &projection,
            Some(&bias),
            &residual,
            &style,
            1e-5,
            &hidden_fused,
            &normalized_fused,
        )
        .unwrap();

        assert_eq!(
            host(&ctx, &hidden_fused),
            host(&ctx, &hidden_pair),
            "fused hidden differs at {rows}x{cols}"
        );
        assert_eq!(
            host(&ctx, &normalized_fused),
            host(&ctx, &normalized_pair),
            "fused normalized differs at {rows}x{cols}"
        );
    }

    // And without a bias, which the DiT's final projection uses.
    let (projection, _) = input(&ctx, &[41, 1536], 13, 1.0);
    let (residual, _) = input(&ctx, &[41, 1536], 17, 1.0);
    let (style, _) = input(&ctx, &[2 * 1536], 19, 0.25);
    let (h1, n1) = (
        arena.tensor(vec![41, 1536], DType::BF16).unwrap(),
        arena.tensor(vec![41, 1536], DType::BF16).unwrap(),
    );
    let (h2, n2) = (
        arena.tensor(vec![41, 1536], DType::BF16).unwrap(),
        arena.tensor(vec![41, 1536], DType::BF16).unwrap(),
    );
    native::bias_residual(&ctx, &projection, None, &residual, &h1).unwrap();
    native::ada_layer_norm(&ctx, &h1, &style, 1e-5, &n1).unwrap();
    native::bias_residual_ada_layer_norm(
        &ctx, &projection, None, &residual, &style, 1e-5, &h2, &n2,
    )
    .unwrap();
    assert_eq!(host(&ctx, &h2), host(&ctx, &h1), "fused hidden differs without bias");
    assert_eq!(host(&ctx, &n2), host(&ctx, &n1), "fused normalized differs without bias");
}

// ── HIP graphs ───────────────────────────────────────────────────────

/// A replayed graph must produce exactly what the eager sequence produced.
///
/// Bit-exact, not close: a graph replays the same kernels against the same
/// addresses, so any difference means it recorded something other than what
/// ran -- a buffer that moved, a launch parameter that was re-derived. There
/// is no honest source of drift here, which makes an exact comparison the
/// right one and a tolerance a way to hide the failure.
#[test]
fn a_replayed_graph_reproduces_the_eager_result_exactly() {
    let ctx = ctx();
    let arena = arena(&ctx);

    // A sequence with a reduction, an elementwise pass and a GEMM, so the
    // graph holds more than one kind of node.
    let (a, _) = input(&ctx, &[64, 256], 5, 1.0);
    let (w, _) = input(&ctx, &[256, 128], 7, 0.05);
    let (bias, _) = input(&ctx, &[128], 11, 1.0);
    let (norm_weight, _) = input(&ctx, &[256], 13, 1.0);
    let run = |out: &apxinf_core::Tensor| -> apxinf_core::Result<()> {
        let normed = arena.tensor(vec![64, 256], DType::BF16)?;
        native::rms_norm(&ctx, &a, &norm_weight, 1e-6, &normed)?;
        native::gemm_bias_activation(
            &ctx,
            &normed,
            &w,
            Some(&bias),
            native::Activation::GeluTanh,
            out,
        )
    };

    arena.reset();
    let eager = arena.tensor(vec![64, 128], DType::BF16).unwrap();
    run(&eager).unwrap();
    ctx.synchronize().unwrap();
    let expected = host(&ctx, &eager);

    // Capture against an arena reset to the same base, so the addresses the
    // graph records are the addresses the replay uses -- the property a
    // shape-keyed plan gives the model.
    arena.reset();
    let captured = arena.tensor(vec![64, 128], DType::BF16).unwrap();
    // SAFETY: `run` allocates only from the arena (no device allocation), copies
    // nothing from host memory, and does not synchronize or download. The arena
    // was reset to the same base before capture and is reset again before each
    // replay below.
    let (graph, ()) = unsafe { apxinf_hip::graph::HipGraph::capture(&ctx, || run(&captured)) }
        .expect("capture");
    assert!(
        graph.nodes() >= 2,
        "the graph holds {} nodes; the sequence issues a norm and a GEMM, so a count \
         below two means the capture missed work",
        graph.nodes()
    );

    // Clear the output so a replay that records nothing cannot pass by leaving
    // the captured run's result behind.
    let zeros = apxinf_core::Tensor::zeros(vec![64, 128], DType::BF16);
    native::upload(&ctx, &zeros, &captured).unwrap();
    ctx.synchronize().unwrap();
    assert_ne!(host(&ctx, &captured), expected, "the output was not cleared");

    graph.launch(&ctx).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        host(&ctx, &captured),
        expected,
        "the replayed graph differs from the eager sequence"
    );

    // And again, to catch a graph that is only correct once.
    native::upload(&ctx, &zeros, &captured).unwrap();
    graph.launch(&ctx).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(host(&ctx, &captured), expected, "the second replay differs");
}

/// An empty capture must be refused.
///
/// It instantiates and replays without error, leaving whatever was in the
/// output buffer. On the inference after a real one that is the previous
/// action -- correct-looking, and wrong.
#[test]
fn an_empty_capture_is_refused() {
    let ctx = ctx();
    // SAFETY: the closure issues nothing at all, which is the case under test.
    let result = unsafe { apxinf_hip::graph::HipGraph::capture(&ctx, || Ok(())) };
    assert!(result.is_err(), "a capture that recorded no work was accepted");
}
