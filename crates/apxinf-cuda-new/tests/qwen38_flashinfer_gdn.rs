//! The vendored FlashInfer GDN prefill against our own chunked scan.
//!
//! `gdn_chunk_scan` is already checked against the official reference tensors
//! (cosine 0.999996) and against single-token decode state (cosine 1.000000),
//! so it is the reference here. What this test establishes is that the
//! FlashInfer kernel, wired through our ABI with our layouts, computes the
//! same recurrence -- before it is put in front of the model.
//!
//! The two paths do not take identical inputs, and the differences are the
//! point:
//!
//! * ours is BF16 and L2-normalizes q/k internally, then scales q by
//!   `k_dim**-0.5`; FlashInfer is FP16, wants q/k normalized by the caller,
//!   and takes the scale as an argument.
//! * ours needs `seq` to be a multiple of 64; FlashInfer does not.
//!
//! So the host normalizes once and feeds each path what it expects. The
//! remaining gap is BF16 against FP16 over the same maths, which is what the
//! tolerances below allow for.
//!
//! ```text
//! bash crates/apxinf-cuda-new/test-new.sh \
//!   test -p apxinf-cuda --test qwen38_flashinfer_gdn -- --include-ignored --nocapture
//! ```

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

const HEAD_DIM: usize = 128;
const Q_HEADS: usize = 16;
const V_HEADS: usize = 48;
const CHUNK: usize = 64;

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes.max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes.max(1)]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_bf16(tensor: &Tensor, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 2];
    CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|r| f32::from_bits((u16::from_le_bytes([r[0], r[1]]) as u32) << 16))
        .collect()
}

fn read_f16(tensor: &Tensor, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 2];
    CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|r| half::f16::from_bits(u16::from_le_bytes([r[0], r[1]])).to_f32())
        .collect()
}

fn read_f32(tensor: &Tensor, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|r| f32::from_le_bytes([r[0], r[1], r[2], r[3]]))
        .collect()
}

/// Cosine and relative L2 in f64, the pair the acceptance bars are written in.
fn cosine_rel_l2(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut dot, mut ng, mut nw, mut err) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in got.iter().zip(want.iter()) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        ng += a * a;
        nw += b * b;
        err += (a - b) * (a - b);
    }
    let cosine = if ng > 0.0 && nw > 0.0 { dot / (ng.sqrt() * nw.sqrt()) } else { 0.0 };
    let rel = if nw > 0.0 { (err / nw).sqrt() } else { 0.0 };
    (cosine, rel)
}

/// Deterministic inputs, with q and k already L2-normalized per head.
///
/// Normalizing on the host means both kernels see the same vectors: ours will
/// normalize again, which is the identity on a unit vector, and FlashInfer
/// requires it done by the caller.
#[allow(clippy::type_complexity)]
fn make_inputs(
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; seq * Q_HEADS * HEAD_DIM];
    let mut k = vec![0.0f32; seq * Q_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; seq * V_HEADS * HEAD_DIM];
    for (index, value) in q.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0013).sin();
    }
    for (index, value) in k.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0017).cos();
    }
    for (index, value) in v.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0007).sin() * 0.8;
    }
    // Keep the unnormalized copies: our scan normalizes internally and the
    // FlashInfer path normalizes on the device, so each gets the form it
    // expects from identical numbers.
    let q_raw = q.clone();
    let k_raw = k.clone();
    for row in 0..seq * Q_HEADS {
        for buffer in [&mut q, &mut k] {
            let slice = &mut buffer[row * HEAD_DIM..(row + 1) * HEAD_DIM];
            let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in slice.iter_mut() {
                *x /= norm;
            }
        }
    }
    // g is the natural-log decay and must stay negative; beta is the gate.
    let g: Vec<f32> = (0..seq * V_HEADS)
        .map(|i| -0.01 - 0.05 * (((i as f32) * 0.011).sin().abs()))
        .collect();
    let beta: Vec<f32> = (0..seq * V_HEADS)
        .map(|i| 0.2 + 0.6 * (((i as f32) * 0.019).cos().abs()))
        .collect();
    (q, k, v, g, beta, q_raw, k_raw)
}

#[test]
#[ignore = "requires a GPU"]
fn flashinfer_gdn_matches_our_chunk_scan() {
    let ctx = CudaContext::new(0).unwrap();
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    for &seq in &[CHUNK, 2 * CHUNK, 512, 1024, 2048] {
        let (q, k, v, g, beta, q_raw, k_raw) = make_inputs(seq);

        // --- our scan: BF16, q/k interleaved is not needed here, and it
        // normalizes and scales internally.
        let q_bf = upload(&ctx, &bf16_bytes(&q), vec![seq, Q_HEADS, HEAD_DIM], DType::BF16);
        let k_bf = upload(&ctx, &bf16_bytes(&k), vec![seq, Q_HEADS, HEAD_DIM], DType::BF16);
        let v_bf = upload(&ctx, &bf16_bytes(&v), vec![seq, V_HEADS, HEAD_DIM], DType::BF16);
        let g_ours = upload(&ctx, &f32_bytes(&g), vec![seq, V_HEADS], DType::F32);
        let beta_ours = upload(&ctx, &f32_bytes(&beta), vec![seq, V_HEADS], DType::F32);
        let out_ours = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::BF16);
        let state_ours = zeros(&ctx, vec![V_HEADS, HEAD_DIM, HEAD_DIM], DType::F32);
        ops::gdn_chunk_scan(
            &ctx, &q_bf, &k_bf, &v_bf, &g_ours, &beta_ours, &out_ours, &state_ours,
            V_HEADS, Q_HEADS, CHUNK,
        )
        .unwrap();

        // --- FlashInfer, reached the way the model reaches it.
        //
        // The conversion runs on the device through gdn_prepare_flashinfer,
        // not on the host. An earlier version of this test normalized and
        // narrowed here instead, which exercised the vendored kernel while
        // leaving our own preparation pass entirely uncovered -- and that is
        // exactly where the bug was: a warp reduction where the block needed a
        // block reduction, so the L2 norm summed 32 of 128 elements. This
        // comparison passed at every length while the model emitted garbage.
        //
        // So hand it what the model hands it: one interleaved BF16 row of
        // q | k | v, unnormalized, and the natural-log decay.
        let row_width = 2 * Q_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM;
        let mut interleaved = vec![0.0f32; seq * row_width];
        for token in 0..seq {
            let row = token * row_width;
            for head in 0..Q_HEADS {
                for d in 0..HEAD_DIM {
                    let source = (token * Q_HEADS + head) * HEAD_DIM + d;
                    interleaved[row + head * HEAD_DIM + d] = q_raw[source];
                    interleaved[row + (Q_HEADS + head) * HEAD_DIM + d] = k_raw[source];
                }
            }
            for index in 0..V_HEADS * HEAD_DIM {
                interleaved[row + 2 * Q_HEADS * HEAD_DIM + index] =
                    v[token * V_HEADS * HEAD_DIM + index];
            }
        }
        let fused = upload(
            &ctx,
            &bf16_bytes(&interleaved),
            vec![seq, row_width],
            DType::BF16,
        );
        let q_h = zeros(&ctx, vec![seq, Q_HEADS, HEAD_DIM], DType::F16);
        let k_h = zeros(&ctx, vec![seq, Q_HEADS, HEAD_DIM], DType::F16);
        let v_h = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::F16);
        let g_fi = zeros(&ctx, vec![seq, V_HEADS], DType::F32);
        let g_log = upload(&ctx, &f32_bytes(&g), vec![seq, V_HEADS], DType::F32);
        ops::gdn_prepare_flashinfer(
            &ctx, &fused, &q_h, &k_h, &v_h, &g_log, &g_fi, seq, row_width,
            Q_HEADS, V_HEADS, HEAD_DIM, 1e-6,
        )
        .unwrap();
        let beta_fi = upload(&ctx, &f32_bytes(&beta), vec![seq, V_HEADS], DType::F32);
        let out_fi = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::F16);
        let state_fi = zeros(&ctx, vec![V_HEADS, HEAD_DIM, HEAD_DIM], DType::F32);
        let cu = upload(
            &ctx,
            &[0i32, seq as i32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            vec![2],
            DType::I32,
        );
        let workspace_bytes = ops::flashinfer_gdn_workspace_bytes(V_HEADS, 1);
        assert!(workspace_bytes > 0, "workspace query returned 0");
        let workspace = zeros(&ctx, vec![workspace_bytes / 4], DType::F32);

        ops::flashinfer_gdn_prefill(
            &ctx, &q_h, &k_h, &v_h, &out_fi, &g_fi, &beta_fi, &cu, &state_fi,
            &workspace, seq, Q_HEADS, V_HEADS, 1, scale,
        )
        .unwrap();
        ctx.synchronize().unwrap();

        let ours_out = read_bf16(&out_ours, seq * V_HEADS * HEAD_DIM);
        let fi_out = read_f16(&out_fi, seq * V_HEADS * HEAD_DIM);
        let (out_cos, out_rel) = cosine_rel_l2(&fi_out, &ours_out);

        let ours_state = read_f32(&state_ours, V_HEADS * HEAD_DIM * HEAD_DIM);
        let fi_state = read_f32(&state_fi, V_HEADS * HEAD_DIM * HEAD_DIM);
        let (state_cos, state_rel) = cosine_rel_l2(&fi_state, &ours_state);

        println!(
            "seq={seq:4}  out: cosine {out_cos:.6} relL2 {out_rel:.5}   \\
state: cosine {state_cos:.6} relL2 {state_rel:.5}"
        );

        // Magnitudes, because a relative L2 says nothing when the
        // reference is near zero. At seq=64 it reported an absurd ratio
        // beside a perfectly healthy cosine, which is the signature of a
        // tiny denominator rather than a wrong answer.
        let rms = |values: &[f32]| -> f64 {
            (values.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()
                / values.len() as f64)
                .sqrt()
        };
        let peak = |values: &[f32]| -> f64 {
            values.iter().fold(0.0f64, |acc, x| acc.max((*x as f64).abs()))
        };
        println!(
            "           state rms ours={:.3e} fi={:.3e}  peak ours={:.3e} fi={:.3e}",
            rms(&ours_state), rms(&fi_state), peak(&ours_state), peak(&fi_state)
        );

        // BF16 carries 8 mantissa bits against FP16's 10, over the same
        // recurrence, so the two disagree at roughly BF16's resolution. The
        // bar is cosine, which is what the acceptance document binds
        // quantized paths to; relative L2 is reported but allowed to be
        // looser because it charges the dtype gap twice.
        assert!(
            out_cos > 0.999,
            "seq={seq}: FlashInfer output diverged from our scan, cosine {out_cos}"
        );
        assert!(
            state_cos > 0.999,
            "seq={seq}: FlashInfer state diverged from our scan, cosine {state_cos}"
        );
        // Cosine alone is blind to a scale error: a state that is the right
        // direction times 1e28 still scores 0.9999. At seq=64 that is exactly
        // what happens, so the magnitudes are asserted too.
        let ours_rms = rms(&ours_state);
        let fi_rms = rms(&fi_state);
        let ratio = if ours_rms > 0.0 { fi_rms / ours_rms } else { f64::INFINITY };
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "seq={seq}: FlashInfer state magnitude is {ratio:.3e}x ours \
(ours rms {ours_rms:.3e}, theirs {fi_rms:.3e}) -- same direction, wrong scale"
        );
    }
}
