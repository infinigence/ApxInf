//! Model-neutral attention contracts and workspace orchestration.

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    optional_ptr, require_address, require_buffers, require_finite, unsupported_dtype,
};
use super::elementwise::{bias_f16, concat_rows_f16};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::ffi;
use crate::workspace::{may_prepare_native_resources, output_buffer};
use crate::CudaKVCache;

pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

fn tensor_slice(
    tensor: &Tensor,
    byte_offset: usize,
    len: usize,
    device_id: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(device_id) {
        return Err(Error::DeviceMismatch {
            expected: Device::Cuda(device_id),
            got: tensor.device(),
        });
    }
    CudaBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(byte_offset, len))
        .map_err(Error::Cuda)
}

fn buffer_slice(buffer: &CudaBuffer, byte_offset: usize, len: usize) -> Result<CudaBuffer> {
    buffer.view(byte_offset, len).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_scores(
    ctx: &CudaContext,
    dtype: DType,
    query: &Tensor,
    query_offset: usize,
    key_cache: &CudaBuffer,
    key_offset: usize,
    scores: &CudaBuffer,
    scores_offset: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let query = tensor_slice(
        query,
        query_offset * element_bytes,
        gqa_ratio * head_dim * element_bytes,
        ctx.device_id(),
    )?;
    let key = buffer_slice(
        key_cache,
        key_offset * element_bytes,
        kv_len * head_dim * element_bytes,
    )?;
    let output = buffer_slice(
        scores,
        scores_offset * element_bytes,
        gqa_ratio * kv_len * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            gqa_ratio,
            kv_len,
            head_dim,
            1.0,
            &query,
            head_dim as i32,
            &key,
            head_dim as i32,
            0.0,
            &output,
            kv_len as i32,
        )
        .map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
fn gqa_values(
    ctx: &CudaContext,
    dtype: DType,
    attention: &Tensor,
    attention_offset: usize,
    value_cache: &CudaBuffer,
    value_offset: usize,
    output: &CudaBuffer,
    output_offset: usize,
    gqa_ratio: usize,
    kv_len: usize,
    head_dim: usize,
) -> Result<()> {
    let element_bytes = dtype.size_in_bytes();
    let attention = tensor_slice(
        attention,
        attention_offset * element_bytes,
        gqa_ratio * kv_len * element_bytes,
        ctx.device_id(),
    )?;
    let value = buffer_slice(
        value_cache,
        value_offset * element_bytes,
        kv_len * head_dim * element_bytes,
    )?;
    let output = buffer_slice(
        output,
        output_offset * element_bytes,
        gqa_ratio * head_dim * element_bytes,
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype,
            CublasTranspose::None,
            CublasTranspose::None,
            gqa_ratio,
            head_dim,
            kv_len,
            1.0,
            &attention,
            kv_len as i32,
            &value,
            head_dim as i32,
            0.0,
            &output,
            head_dim as i32,
        )
        .map_err(Error::Cuda)
}

// TEMP-DIAG (implement_r3 / synthesis_r3): cross-crate probe-line sink for the
// composed-causal attention P-invariant probe below; the qwen_drive model drains it
// into its diag_digest right after prefill_done so the lines land inside the receipt
// window; revert in the acceptance-bound revision.
static ATTN_DIAG_LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// TEMP-DIAG (implement_r3 / synthesis_r3): drain the attention probe lines; revert in
/// the acceptance-bound revision.
pub fn take_attn_diag_lines() -> Vec<String> {
    ATTN_DIAG_LINES
        .lock()
        .map(|mut lines| std::mem::take(&mut *lines))
        .unwrap_or_default()
}

// TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 2 support): per-prompt-row segment
// classes published by the model before prefill (1=prefix-text, 2=image, 3=vision-marker,
// 4=tail-text) so the attn_mix probe can compute token-id-driven segment masses; revert in
// the acceptance-bound revision.
static ATTN_SEG_MAP: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());

/// TEMP-DIAG (implement_r5): publish the segment map; revert in the acceptance-bound revision.
pub fn set_attn_seg_map(map: Vec<u8>) {
    if let Ok(mut guard) = ATTN_SEG_MAP.lock() {
        *guard = map;
    }
}

/// Composed causal GQA prefill for the hdim256 text stack: chunked fp32 QK^T
/// GEMMs, the base-compiled fused causal mask+softmax kernel, and fp32 PV GEMMs
/// (the route-3 floor generalized to the causal hdim256 surface).
pub(crate) fn composed_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
    causal: bool,
) -> Result<Tensor> {
    // FIX (implement_r9): the vendored FA2 hdim256 causal instantiation is measured
    // pathological on sm_89 (r8: text layer-3 prefill consumed the remaining ~225s of
    // the 300s job). Compose measured-healthy cuBLAS GEMMs with the fused causal
    // softmax instead: scores and P stay fp32 end-to-end (strictly closer to exact
    // than FA2's bf16-P MMA rounding), and one RNE cast restores the packed bf16
    // [Q,heads,256] output contract. Causality is exact at chunk boundaries: chunk
    // [t0, t0+c_len) attends keys [0, base+t0+c_len) via kv_offset=base+t0 and the
    // kernel writes masked cells exact 0.0f, so the PV GEMM adds exact zeros. GQA
    // group g=h/group matches FA2's kv-head indexing. Revert/replace in the
    // acceptance-bound revision per the prevailing marker policy.
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    let heads = q_shape[1];
    let kv_heads = k_shape[1];
    let head_dim = q_shape[2];
    let group = heads / kv_heads;
    let base = if causal { key_tokens - q_shape[0] } else { 0 };
    let alpha = (head_dim as f32).sqrt().recip(); // FIX (implement_r9): softmax scale bound to the true head dim 256
                                                  // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 2): value-level attn_mix probe
                                                  // for the composed-causal path, absorbing the r3 P-invariant lines. Fires on prefill
                                                  // full-attention calls only (q_rows>1; Rust && short-circuits so single-row decode calls
                                                  // never increment the counter): fetch counts 0..7 are the 8 prefill full layers; fire at
                                                  // count 0 (text layer 3) and count 7 (text layer 31). Probe rows {3, 5000, q_rows-16,
                                                  // q_rows-13, q_rows-1} = {3, 5000, 10441, 10444, 10456} at the measured 10457-row prefill,
                                                  // head 0 (+ head 1 at q_rows-13 for the GQA pairing leg, layer 3 only). Host math on the
                                                  // already-resident fp32 buffers (qf32/kf32/vf32/out_f32), no new kernels; one-time ~86MB
                                                  // K/V readbacks per fired layer; revert in the acceptance-bound revision.
    static ATTN_PROBE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let attn_probe = causal
        && q_shape[0] > 1
        && matches!(
            ATTN_PROBE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            0 | 7
        );
    let probe_rows: [usize; 5] = [
        3,
        5000,
        q_shape[0].saturating_sub(16),
        q_shape[0].saturating_sub(13),
        q_shape[0].saturating_sub(1),
    ];
    let mut probe_store: Vec<(usize, usize, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    let mut probe_gqa: Option<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
    let mut probe_tail_store: Vec<(usize, usize, Vec<f32>)> = Vec::new();
    const C_Q: usize = 1024; // FIX (implement_r9): query-chunk rows; bounds the fp32 scores transient
    let qf32_buf = CudaBuffer::alloc(q_shape[0] * heads * head_dim * 4, ctx.device_id())
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    let qf32_t = qf32_buf
        .as_tensor(q.shape().clone(), DType::F32)
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    super::linear_attention::cast_bf16_to_f32(ctx, q, &qf32_t)?; // FIX (implement_r9)
    let kf32_buf = CudaBuffer::alloc(k_shape[0] * kv_heads * head_dim * 4, ctx.device_id())
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    let kf32_t = kf32_buf
        .as_tensor(k.shape().clone(), DType::F32)
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    super::linear_attention::cast_bf16_to_f32(ctx, k, &kf32_t)?; // FIX (implement_r9)
    let vf32_buf = CudaBuffer::alloc(k_shape[0] * kv_heads * head_dim * 4, ctx.device_id())
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    let vf32_t = vf32_buf
        .as_tensor(v.shape().clone(), DType::F32)
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    super::linear_attention::cast_bf16_to_f32(ctx, v, &vf32_t)?; // FIX (implement_r9)
    let out_f32_buf = CudaBuffer::alloc(q_shape[0] * heads * head_dim * 4, ctx.device_id())
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    let out_f32_t = out_f32_buf
        .as_tensor(q.shape().clone(), DType::F32)
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    let scores = CudaBuffer::alloc(
        heads * C_Q.min(q_shape[0]) * key_tokens * 4,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?; // FIX (implement_r10): fp32 [C,heads,kend] token-major slab reused per chunk; min() is byte-identical at the r9 prefill geometry (q=10457>C_Q=1024) and shrinks the decode Q=1 slab from ~685MB to heads*kv*4 (~669KB); beta=0 GEMMs write every read cell
    for t0 in (0..q_shape[0]).step_by(C_Q) {
        let c_len = C_Q.min(q_shape[0] - t0);
        let kend = if causal {
            base + t0 + c_len
        } else {
            key_tokens
        };
        // An offset beyond all keys makes the same softmax kernel unmasked.
        let kv_offset = if causal { base + t0 } else { key_tokens } as u32;
        for h in 0..heads {
            let g = h / group; // FIX (implement_r9): kv head for q head h
            let q_head = buffer_slice(
                &qf32_buf,
                ((t0 * heads + h) * head_dim) * 4,
                ((c_len - 1) * heads * head_dim + head_dim) * 4,
            )?; // FIX (implement_r9)
            let k_head = buffer_slice(
                &kf32_buf,
                (g * head_dim) * 4,
                ((kend - 1) * kv_heads * head_dim + head_dim) * 4,
            )?; // FIX (implement_r9)
            let scores_head = buffer_slice(
                &scores,
                (h * kend) * 4,
                (((c_len - 1) * heads + 1) * kend) * 4,
            )?; // FIX (implement_r9): ldc=heads*kend writes the token-major [C,heads,kend] layout
            ctx.cublas()
                .gemm_ex(
                    DType::F32,
                    CublasTranspose::None,
                    CublasTranspose::Transpose,
                    c_len,
                    kend,
                    head_dim,
                    alpha,
                    &q_head,
                    (heads * head_dim) as i32,
                    &k_head,
                    (kv_heads * head_dim) as i32,
                    0.0,
                    &scores_head,
                    (heads * kend) as i32,
                )
                .map_err(Error::Cuda)?; // FIX (implement_r9): fp32 Q*K^T; alpha folds the softmax scale
        }
        // FIX (implement_final_r20): Option A budget repair -- in-place tiled causal fp32
        // softmax over the scores slab (adapters/custom_kernels.cu
        // row_softmax_causal_f32_kernel) replaces the thread-per-element
        // attention_softmax_f32_kernel and softmax_causal's fresh ~3.84GB/layer
        // alloc/memset/free churn; masked cells are written exact 0.0f so the PV GEMM
        // adds exact zeros. In-place is safe: rows are block-exclusive, every index is
        // owned by exactly one thread, and block_max/block_sum synchronize between the
        // read and write passes. The PV loop below reads probs_head from `scores`.
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_row_softmax_causal_f32(
                scores.ptr() as *const std::ffi::c_void,
                scores.ptr(),
                kend as u32,
                (c_len * heads) as u32,
                kv_offset,
                heads as u32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        for h in 0..heads {
            let g = h / group; // FIX (implement_r9)
            let probs_head = buffer_slice(
                &scores,
                (h * kend) * 4,
                (((c_len - 1) * heads + 1) * kend) * 4,
            )?; // FIX (implement_final_r20): in-place probs live in the scores slab
            let v_head = buffer_slice(
                &vf32_buf,
                (g * head_dim) * 4,
                ((kend - 1) * kv_heads * head_dim + head_dim) * 4,
            )?; // FIX (implement_r9)
            let out_head = buffer_slice(
                &out_f32_buf,
                ((t0 * heads + h) * head_dim) * 4,
                ((c_len - 1) * heads * head_dim + head_dim) * 4,
            )?; // FIX (implement_r9)
            ctx.cublas()
                .gemm_ex(
                    DType::F32,
                    CublasTranspose::None,
                    CublasTranspose::None,
                    c_len,
                    head_dim,
                    kend,
                    1.0,
                    &probs_head,
                    (heads * kend) as i32,
                    &v_head,
                    (kv_heads * head_dim) as i32,
                    0.0,
                    &out_head,
                    (heads * head_dim) as i32,
                )
                .map_err(Error::Cuda)?; // FIX (implement_r9): fp32 P*V into the packed [Q,heads,256] fp32 stage
        }
        if attn_probe {
            let read_row = |buf: &CudaBuffer, r: usize, h: usize| -> Result<Vec<f32>> {
                // TEMP-DIAG (implement_r5): packed [rows, heads, head_dim] fp32 row readback.
                let bytes = crate::transfers::copy_device_to_host(
                    ctx.device_id(),
                    buf.ptr() as usize + ((r * heads + h) * head_dim) * 4,
                    head_dim * 4,
                )
                .map_err(Error::Cuda)?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            };
            for &row in probe_rows.iter() {
                if row >= t0 && row < t0 + c_len {
                    // TEMP-DIAG (implement_r5): capture the probed row's P row (head 0; element
                    // offset ((row-t0)*heads + head0)*kend in the token-major slab), q row and
                    // out row to host Vecs; the P row is per-chunk-live only.
                    let p_bytes = crate::transfers::copy_device_to_host(
                        ctx.device_id(),
                        scores.ptr() as usize + ((row - t0) * heads) * kend * 4,
                        kend * 4,
                    )
                    .map_err(Error::Cuda)?;
                    let p_row: Vec<f32> = p_bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    let q_row = read_row(&qf32_buf, row, 0)?;
                    let out_row = read_row(&out_f32_buf, row, 0)?;
                    probe_store.push((row, kend, p_row, q_row, out_row));
                    if row == q_shape[0].saturating_sub(13)
                        && ATTN_PROBE.load(std::sync::atomic::Ordering::Relaxed) == 1
                    {
                        // TEMP-DIAG (implement_r5): head-1 GQA dual-fit leg at row 10444,
                        // layer 3 only (counter reads 1 during the first firing).
                        let p1_bytes = crate::transfers::copy_device_to_host(
                            ctx.device_id(),
                            scores.ptr() as usize + ((row - t0) * heads + 1) * kend * 4,
                            kend * 4,
                        )
                        .map_err(Error::Cuda)?;
                        let p1_row: Vec<f32> = p1_bytes
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        let q1_row = read_row(&qf32_buf, row, 1)?;
                        let out1_row = read_row(&out_f32_buf, row, 1)?;
                        probe_gqa = Some((row, p1_row, q1_row, out1_row));
                    }
                }
            }
            for row in q_shape[0].saturating_sub(15)..q_shape[0] {
                if row >= t0 && row < t0 + c_len {
                    for h in 0..heads {
                        // TEMP-DIAG (implement_r5): capture tail-row attention outputs at every
                        // head for the query-independence (constant-bias) leg.
                        let out_row = read_row(&out_f32_buf, row, h)?;
                        probe_tail_store.push((row, h, out_row));
                    }
                }
            }
        }
    }

    // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 2): host-side value reconstruction
    // on the captured rows plus the already-resident fp32 K/V buffers (~86MB one-time readbacks
    // per fired layer); emits 8 lines at layer 3 (5 per-row + gqa + kv + tail) and 7 at layer 31.
    if attn_probe {
        let layer_tag = if ATTN_PROBE.load(std::sync::atomic::Ordering::Relaxed) == 1 {
            3
        } else {
            31
        };
        let kv_len = k_shape[0];
        let read_kv = |buf: &CudaBuffer| -> Result<Vec<f32>> {
            let bytes = crate::transfers::copy_device_to_host(
                ctx.device_id(),
                buf.ptr() as usize,
                kv_len * kv_heads * head_dim * 4,
            )
            .map_err(Error::Cuda)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        let k_host = read_kv(&kf32_buf)?;
        let v_host = read_kv(&vf32_buf)?;
        let seg_map: Vec<u8> = ATTN_SEG_MAP
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let seg_of = |j: usize| -> u8 { seg_map.get(j).copied().unwrap_or(0) };
        // Row j of kv head g in the packed [kv_len, kv_heads, head_dim] fp32 host copies.
        let k_slice =
            |j: usize, g: usize| -> (usize, usize) { ((j * kv_heads + g) * head_dim, head_dim) };
        if let Ok(mut lines) = ATTN_DIAG_LINES.lock() {
            for (row, kend_r, p_row, q_row, out_row) in &probe_store {
                let valid = (*row + 1).min(*kend_r);
                let mut p_max = 0.0f32;
                let mut viol = 0usize;
                for (j, &p) in p_row.iter().enumerate() {
                    if j < valid && p > p_max {
                        p_max = p;
                    }
                    if j > *row && p != 0.0 {
                        viol += 1;
                    }
                }
                let mut masses = [0.0f64; 5];
                let mut entropy = 0.0f64;
                let mut p_sum = 0.0f64;
                for (j, &p) in p_row.iter().enumerate().take(valid) {
                    let pf = p as f64;
                    p_sum += pf;
                    let cls = seg_of(j).min(4) as usize;
                    masses[cls] += pf;
                    if p > 0.0 {
                        entropy -= pf * pf.ln();
                    }
                }
                let keff = entropy.exp();
                let out_l2: f64 = out_row
                    .iter()
                    .map(|&v| (v as f64) * (v as f64))
                    .sum::<f64>()
                    .sqrt();
                // M5: affine score reconstruction s = ln(p/p_max) vs alpha*(q.K_j) at kv head 0.
                let mut s_pred: Vec<f32> = Vec::with_capacity(valid);
                let mut s_pred_max = f32::NEG_INFINITY;
                for j in 0..valid {
                    let (ko, kw) = k_slice(j, 0);
                    let mut dot = 0.0f32;
                    for c in 0..head_dim {
                        dot += q_row[c] * k_host[ko + c.min(kw - 1)];
                    }
                    let s = alpha * dot;
                    s_pred.push(s);
                    if s > s_pred_max {
                        s_pred_max = s;
                    }
                }
                let mut m5res = 0.0f64;
                for j in 0..valid {
                    let p = p_row[j];
                    if p > 0.0 && p_max > 0.0 {
                        let s_meas = (p / p_max).ln();
                        let s_pred_n = s_pred[j] - s_pred_max;
                        let d = (s_meas - s_pred_n).abs() as f64;
                        if d > m5res {
                            m5res = d;
                        }
                    }
                }
                let mut p_top: Vec<usize> = (0..valid).collect();
                p_top.sort_by(|&a, &b| {
                    p_row[b]
                        .partial_cmp(&p_row[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                p_top.truncate(8);
                let mut s_top: Vec<usize> = (0..valid).collect();
                s_top.sort_by(|&a, &b| {
                    s_pred[b]
                        .partial_cmp(&s_pred[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                s_top.truncate(8);
                let m5agr = p_top == s_top;
                // M6: PV reconstruction and uniform-mean distance at kv head 0.
                let mut pv = vec![0.0f64; head_dim];
                let mut mu = vec![0.0f64; head_dim];
                for j in 0..valid {
                    let pj = p_row[j] as f64;
                    let (vo, _vw) = k_slice(j, 0);
                    for c in 0..head_dim {
                        pv[c] += pj * v_host[vo + c] as f64;
                        mu[c] += v_host[vo + c] as f64;
                    }
                }
                let inv = 1.0 / valid.max(1) as f64;
                let mut pv_err = 0.0f64;
                let mut mu_dist = 0.0f64;
                for c in 0..head_dim {
                    let d = pv[c] - out_row[c] as f64;
                    pv_err += d * d;
                    let dm = out_row[c] as f64 - mu[c] * inv;
                    mu_dist += dm * dm;
                }
                let denom = out_l2.max(1e-12);
                lines.push(format!(
                    "[qwen_drive] attn_mix k={} row={} valid={} m_pre={:.6} m_img={:.6} m_mark={:.6} m_tail={:.6} keff={:.1} pmax={:.6} viol={} outl2={:.4} m5res={:.4} m5agr={} m6pv={:.4} m6mu={:.4}",
                    layer_tag,
                    row,
                    valid,
                    masses[1] / p_sum.max(1e-12),
                    masses[2] / p_sum.max(1e-12),
                    masses[3] / p_sum.max(1e-12),
                    masses[4] / p_sum.max(1e-12),
                    keff,
                    p_max,
                    viol,
                    out_l2,
                    m5res,
                    m5agr,
                    pv_err.sqrt() / denom,
                    mu_dist.sqrt() / denom
                ));
            }
            // GQA dual-fit leg (head 1 against kv heads 0 and 1), emitted at layer 3 only.
            if let Some((row, p1_row, q1_row, _out1_row)) = &probe_gqa {
                let valid = (*row + 1).min(p1_row.len());
                let mut p1_max = 0.0f32;
                for &p in p1_row.iter().take(valid) {
                    if p > p1_max {
                        p1_max = p;
                    }
                }
                let mut res = [0.0f64; 2];
                for g in 0..2usize {
                    let mut s_max = f32::NEG_INFINITY;
                    let mut s_vec: Vec<f32> = Vec::with_capacity(valid);
                    for j in 0..valid {
                        let (ko, kw) = k_slice(j, g);
                        let mut dot = 0.0f32;
                        for c in 0..head_dim {
                            dot += q1_row[c] * k_host[ko + c.min(kw - 1)];
                        }
                        let s = alpha * dot;
                        s_vec.push(s);
                        if s > s_max {
                            s_max = s;
                        }
                    }
                    for j in 0..valid {
                        let p = p1_row[j];
                        if p > 0.0 && p1_max > 0.0 {
                            let d = ((p / p1_max).ln() - (s_vec[j] - s_max)).abs();
                            if d as f64 > res[g] {
                                res[g] = d as f64;
                            }
                        }
                    }
                }
                lines.push(format!(
                    "[qwen_drive] attn_mix_gqa k=3 row={} head=1 res_g0={:.4} res_g1={:.4}",
                    row, res[0], res[1]
                ));
            }
            // M7: per-segment mean cosine to the segment mean (kv head 0), image vs prefix, K and V.
            let seg_cos = |src: &[f32], cls: u8| -> f64 {
                let mut mean_v = vec![0.0f64; head_dim];
                let mut n = 0usize;
                for j in 0..kv_len {
                    if seg_of(j) == cls {
                        let (o, _w) = k_slice(j, 0);
                        for c in 0..head_dim {
                            mean_v[c] += src[o + c] as f64;
                        }
                        n += 1;
                    }
                }
                if n == 0 {
                    return f64::NAN;
                }
                for c in 0..head_dim {
                    mean_v[c] /= n as f64;
                }
                let mean_norm: f64 = mean_v.iter().map(|&v| v * v).sum::<f64>().sqrt();
                let mut acc = 0.0f64;
                for j in 0..kv_len {
                    if seg_of(j) == cls {
                        let (o, _w) = k_slice(j, 0);
                        let mut dot = 0.0f64;
                        let mut nj = 0.0f64;
                        for c in 0..head_dim {
                            dot += src[o + c] as f64 * mean_v[c];
                            nj += (src[o + c] as f64) * (src[o + c] as f64);
                        }
                        acc += dot / (nj.sqrt() * mean_norm).max(1e-12);
                    }
                }
                acc / n as f64
            };
            lines.push(format!(
                "[qwen_drive] attn_mix_kv k={} k_img_cos={:.4} k_pre_cos={:.4} v_img_cos={:.4} v_pre_cos={:.4}",
                layer_tag,
                seg_cos(&k_host, 2),
                seg_cos(&k_host, 1),
                seg_cos(&v_host, 2),
                seg_cos(&v_host, 1)
            ));
            // M8: max pairwise cosine between tail-row attention outputs, per head then max.
            let mut maxcos = 0.0f64;
            for h in 0..heads {
                let rows_h: Vec<&Vec<f32>> = probe_tail_store
                    .iter()
                    .filter(|(_, hh, _)| *hh == h)
                    .map(|(_, _, v)| v)
                    .collect();
                for i in 0..rows_h.len() {
                    for j in (i + 1)..rows_h.len() {
                        let mut dot = 0.0f64;
                        let mut na = 0.0f64;
                        let mut nb = 0.0f64;
                        for c in 0..head_dim {
                            let a = rows_h[i][c] as f64;
                            let b = rows_h[j][c] as f64;
                            dot += a * b;
                            na += a * a;
                            nb += b * b;
                        }
                        let cs = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
                        if cs > maxcos {
                            maxcos = cs;
                        }
                    }
                }
            }
            lines.push(format!(
                "[qwen_drive] attn_mix_tail k={} maxcos={:.4}",
                layer_tag, maxcos
            ));
        }
    }
    let out_bf16 = output_buffer(ctx, q.size_in_bytes())?; // FIX (implement_r9)
    let out_bf16_t = out_bf16
        .as_tensor(q.shape().clone(), DType::BF16)
        .map_err(Error::Cuda)?; // FIX (implement_r9)
    super::linear_attention::cast_f32_to_bf16(ctx, &out_f32_t, &out_bf16_t)?; // FIX (implement_r9)
    Ok(out_bf16_t)
}

/// GQA scaled-dot-product attention over an existing CUDA KV cache.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(
    ctx: &CudaContext,
    query: &Tensor,
    kv: &dyn KvCache,
    layer_idx: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    max_seq_len: usize,
    kv_offset: u32,
) -> Result<Tensor> {
    if n_kv_heads == 0 || n_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(Error::Other(format!(
            "attention requires non-zero divisible head counts, got {n_heads}/{n_kv_heads}"
        )));
    }
    let query_dims = query.shape().dims();
    if query.device() != Device::Cuda(ctx.device_id())
        || query_dims.len() != 3
        || query_dims[1] != n_heads
        || query_dims[2] != head_dim
    {
        return Err(Error::Other(format!(
            "attention query must be CUDA{} [seq,{n_heads},{head_dim}], got {} {:?}",
            ctx.device_id(),
            query.device(),
            query_dims
        )));
    }
    if kv_len == 0 || kv_len > max_seq_len {
        return Err(Error::Other(format!(
            "attention kv_len {kv_len} is outside 1..={max_seq_len}"
        )));
    }
    let cache = kv
        .as_any()
        .downcast_ref::<CudaKVCache>()
        .ok_or_else(|| Error::Other("expected CudaKVCache".into()))?;

    let seq_len = query_dims[0];
    let gqa_ratio = n_heads / n_kv_heads;
    let dtype = query.dtype();
    let element_bytes = dtype.size_in_bytes();
    let scores = CudaBuffer::alloc(seq_len * n_heads * kv_len * element_bytes, ctx.device_id())
        .map_err(Error::Cuda)?;
    let key_cache = cache.k_buffer(layer_idx);

    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_scores(
                ctx,
                dtype,
                query,
                (sequence * n_heads + kv_head * gqa_ratio) * head_dim,
                key_cache,
                kv_head * max_seq_len * head_dim,
                &scores,
                (sequence * n_heads + kv_head * gqa_ratio) * kv_len,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        }
    }

    let scores = scores.into_tensor(Shape::new(vec![seq_len * n_heads, kv_len]), dtype);
    let scores = super::elementwise::scale(ctx, &scores, 1.0 / (head_dim as f32).sqrt())?;
    let attention = softmax_causal(ctx, &scores, kv_offset, n_heads as u32)?;

    let output = CudaBuffer::alloc(
        seq_len * n_heads * head_dim * element_bytes,
        ctx.device_id(),
    )
    .map_err(Error::Cuda)?;
    let value_cache = cache.v_buffer(layer_idx);
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            gqa_values(
                ctx,
                dtype,
                &attention,
                (sequence * n_heads + kv_head * gqa_ratio) * kv_len,
                value_cache,
                kv_head * max_seq_len * head_dim,
                &output,
                (sequence * n_heads + kv_head * gqa_ratio) * head_dim,
                gqa_ratio,
                kv_len,
                head_dim,
            )?;
        }
    }

    Ok(output.into_tensor(Shape::new(vec![seq_len, n_heads * head_dim]), dtype))
}

/// F32 attention softmax into caller-owned storage using a device position.
pub fn softmax_f32_into(
    ctx: &CudaContext,
    scores: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    heads: usize,
    position: CudaDeviceAddress,
) -> Result<()> {
    let bytes = checked_bytes(DType::F32, &[heads, cols], "attention softmax")?;
    require_buffers(
        ctx,
        "attention softmax",
        &[("scores", scores, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "attention softmax", "position", position, 4)?;
    check_cuda(unsafe {
        ffi::apxinf_attention_softmax_decode_f32(
            scores.ptr(),
            output.ptr(),
            cols as u32,
            heads as u32,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

/// Decode-time BF16 flash attention into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn flash_bf16_into(
    ctx: &CudaContext,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    output: &CudaBuffer,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bucket_kv_len: usize,
    max_seq_len: usize,
    scale: f32,
    position: CudaDeviceAddress,
) -> Result<()> {
    require_finite("flash attention", &[scale])?;
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 || bucket_kv_len > max_seq_len {
        return Err(Error::Other(
            "flash attention received invalid head or sequence dimensions".into(),
        ));
    }
    let query_size = checked_bytes(DType::BF16, &[heads, head_dim], "flash attention")?;
    let cache_size = checked_bytes(
        DType::BF16,
        &[kv_heads, max_seq_len, head_dim],
        "flash attention",
    )?;
    require_buffers(
        ctx,
        "flash attention",
        &[
            ("query", query, query_size),
            ("key cache", key_cache, cache_size),
            ("value cache", value_cache, cache_size),
            ("output", output, query_size),
        ],
    )?;
    require_address(ctx, "flash attention", "position", position, 4)?;
    check_cuda(unsafe {
        ffi::apxinf_flash_attn_decode_bf16(
            query.ptr(),
            key_cache.ptr(),
            value_cache.ptr(),
            output.ptr(),
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
            bucket_kv_len as u32,
            max_seq_len as u32,
            scale,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

/// Softmax on CUDA. Dispatches on dtype.
pub fn softmax(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_softmax_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_softmax_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("softmax", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// Non-causal full attention for the vision tower. Q/K/V each
/// `[seq, n_heads, head_dim]` bf16; returns `[seq, n_heads * head_dim]`.
/// head_dim must be 64 (Qwen3-VL-2B vision).
pub fn vision(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Err(Error::Other("vision_sdpa: only BF16 supported".into()));
    }
    if head_dim != 64 {
        return Err(Error::Other("vision_sdpa: head_dim must be 64".into()));
    }
    let device_id = ctx.device_id();
    let out_bytes = seq_len * n_heads * head_dim * DType::BF16.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    unsafe {
        let res = ffi::apxinf_vision_sdpa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            out_buf.ptr(),
            seq_len as u32,
            n_heads as u32,
            head_dim as u32,
            scale,
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![seq_len, n_heads * head_dim]),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// Causal attention mask on CUDA. Dispatches on dtype.
pub fn causal_mask(ctx: &CudaContext, input: &Tensor, kv_offset: u32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_causal_mask_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_causal_mask_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("causal_mask", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// Fused causal mask + softmax. Dispatches on dtype.
pub fn softmax_causal(
    ctx: &CudaContext,
    input: &Tensor,
    kv_offset: u32,
    n_heads: u32,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap();

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_attention_softmax_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_attention_softmax_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                kv_offset,
                n_heads,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("attention_softmax", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}
pub fn split_qkv_bias_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV split")?;
    let projection_width = heads * head_dim;
    if qkv.dtype() != DType::BF16
        || width != 3 * projection_width
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [width])
    {
        return Err(Error::Other(
            "static inference BF16 vision QKV shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, projection_width)?;
    let k = bf16_output(ctx, tokens, projection_width)?;
    let v = bf16_output(ctx, tokens, projection_width)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_split_bias_bf16(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            q.ptr(),
            k.ptr(),
            v.ptr(),
            tokens as i32,
            projection_width as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), q),
        k: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), k),
        v: make_gpu_tensor(shape, DType::BF16, ctx.device_id(), v),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn split_gqa_qkv_mrope_cache_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &CudaBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    cache_tokens: usize,
    caches: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "GQA QKV mRoPE")?;
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let expected_width = q_width + 2 * kv_width;
    if !matches!(qkv.dtype(), DType::BF16 | DType::F16)
        || width != expected_width
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 2 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || sections[1] + sections[2] > head_dim / 2
        || position_ids.len() < tokens * 3 * std::mem::size_of::<u32>()
        || bias.is_some_and(|value| {
            value.dtype() != DType::BF16 || value.shape().dims() != [expected_width]
        })
    {
        return Err(Error::Other(
            "static inference BF16 GQA QKV mRoPE shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, q_width)?;
    let owned_k = caches
        .is_none()
        .then(|| bf16_output(ctx, cache_tokens * kv_heads, head_dim))
        .transpose()?;
    let owned_v = caches
        .is_none()
        .then(|| bf16_output(ctx, cache_tokens * kv_heads, head_dim))
        .transpose()?;
    let (k_ptr, v_ptr, cache_offset) = if let Some((k, v, offset)) = caches {
        let expected_shape = [cache_tokens, kv_heads, head_dim];
        if k.dtype() != DType::BF16
            || v.dtype() != DType::BF16
            || k.shape().dims() != expected_shape
            || v.shape().dims() != expected_shape
            || offset + tokens > cache_tokens
        {
            return Err(Error::Other(
                "static inference BF16 GQA QKV cache shape mismatch".into(),
            ));
        }
        (gpu_ptr(k)?, gpu_ptr(v)?, offset)
    } else {
        if tokens > cache_tokens {
            return Err(Error::Other(
                "static inference BF16 GQA QKV cache is too short".into(),
            ));
        }
        (
            owned_k.as_ref().unwrap().ptr(),
            owned_v.as_ref().unwrap().ptr(),
            0,
        )
    };
    let status = unsafe {
        let launch = if qkv.dtype() == DType::F16 {
            ffi::apxinf_static_gqa_qkv_mrope_cache_f16
        } else {
            ffi::apxinf_static_gqa_qkv_mrope_cache_bf16
        };
        launch(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            position_ids.ptr().cast(),
            q.ptr(),
            k_ptr,
            v_ptr,
            tokens as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            theta,
            sections[1] as i32,
            sections[2] as i32,
            cache_offset as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cuda(status).map_err(Error::Cuda)?;
    let q = make_gpu_tensor(
        Shape::new(vec![tokens, q_heads, head_dim]),
        DType::BF16,
        ctx.device_id(),
        q,
    );
    if let (Some(k), Some(v)) = (owned_k, owned_v) {
        Ok(QkvTensors {
            q,
            k: make_gpu_tensor(
                Shape::new(vec![cache_tokens, kv_heads, head_dim]),
                DType::BF16,
                ctx.device_id(),
                k,
            ),
            v: make_gpu_tensor(
                Shape::new(vec![cache_tokens, kv_heads, head_dim]),
                DType::BF16,
                ctx.device_id(),
                v,
            ),
        })
    } else {
        Ok(QkvTensors {
            q,
            k: caches.unwrap().0.clone(),
            v: caches.unwrap().1.clone(),
        })
    }
}

pub fn split_vision_qkv_rope_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &CudaBuffer,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV RoPE")?;
    let projection_width = heads * head_dim;
    let expected_width = 3 * projection_width;
    if !matches!(qkv.dtype(), DType::BF16 | DType::F16)
        || width != expected_width
        || heads == 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 4 != 0
        || !theta.is_finite()
        || theta <= 0.0
        || position_ids.len() < tokens * 2 * std::mem::size_of::<u32>()
        || bias.is_some_and(|value| {
            value.dtype() != DType::BF16 || value.shape().dims() != [expected_width]
        })
    {
        return Err(Error::Other(
            "static inference BF16 vision QKV RoPE shape mismatch".into(),
        ));
    }
    let q = bf16_output(ctx, tokens, projection_width)?;
    let k = bf16_output(ctx, tokens, projection_width)?;
    let v = bf16_output(ctx, tokens, projection_width)?;
    let status = unsafe {
        let launch = if qkv.dtype() == DType::F16 {
            ffi::apxinf_static_vision_qkv_rope_f16
        } else {
            ffi::apxinf_static_vision_qkv_rope_bf16
        };
        launch(
            gpu_ptr(qkv)?,
            optional_ptr(bias)?,
            position_ids.ptr().cast(),
            q.ptr(),
            k.ptr(),
            v.ptr(),
            tokens as i32,
            heads as i32,
            head_dim as i32,
            theta,
            ctx.stream().handle(),
        )
    };
    ffi::check_cuda(status).map_err(Error::Cuda)?;
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), q),
        k: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), k),
        v: make_gpu_tensor(shape, DType::BF16, ctx.device_id(), v),
    })
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
fn fa2_attention(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batches: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = batches
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .ok_or_else(|| Error::Other("static inference BF16 FA2 LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Error::Other("static inference BF16 FA2 LSE byte size overflow".into())
            })?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            batches as i32,
            query_tokens as i32,
            key_tokens as i32,
            query_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fa2_attention_causal(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let softmax_lse = output_buffer(ctx, query_heads * query_tokens * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_bf16_causal(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            1,
            query_tokens as i32,
            key_tokens as i32,
            query_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            (head_dim as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
fn fa2_splitkv_enabled(
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> bool {
    if std::env::var_os("APXINF_DISABLE_FA2_SPLITKV").is_some() {
        return false;
    }
    query_tokens <= 64
        && key_tokens > query_tokens
        && query_heads > kv_heads
        && matches!(head_dim, 128 | 256)
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fa2_attention_splitkv(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batches: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Tensor> {
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = batches
        .checked_mul(query_heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .ok_or_else(|| Error::Other("static inference BF16 split-KV LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Other("static inference BF16 split-KV LSE overflow".into()))?,
    )?;
    let block_n = if head_dim <= 64 {
        256
    } else if head_dim <= 128 {
        128
    } else {
        64
    };
    let max_splits = key_tokens.div_ceil(block_n).min(128);
    let softmax_lse_accum = output_buffer(
        ctx,
        max_splits
            .checked_mul(lse_elements)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                Error::Other("static inference BF16 split-KV LSE accum overflow".into())
            })?,
    )?;
    let o_accum_elements = max_splits
        .checked_mul(batches)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(query_heads))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::Other("static inference BF16 split-KV O accum overflow".into()))?;
    let o_accum = output_buffer(
        ctx,
        o_accum_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                Error::Other("static inference BF16 split-KV O accum byte overflow".into())
            })?,
    )?;
    unsafe {
        let status = if causal {
            ffi::apxinf_static_fa2_bf16_causal_splitkv(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                softmax_lse_accum.ptr(),
                o_accum.ptr(),
                batches as i32,
                query_tokens as i32,
                key_tokens as i32,
                query_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                (head_dim as f32).sqrt().recip(),
                ctx.caps().multiprocessor_count as i32,
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_static_fa2_bf16_splitkv(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                softmax_lse_accum.ptr(),
                o_accum.ptr(),
                batches as i32,
                query_tokens as i32,
                key_tokens as i32,
                query_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                (head_dim as f32).sqrt().recip(),
                ctx.caps().multiprocessor_count as i32,
                ctx.stream().handle(),
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(apxinf_cutlass_fmha)]
fn cublas_mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let status = unsafe {
        ffi::apxinf_static_cublas_mqa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() < 2
        || v.shape() != k.shape()
        || k_shape[k_shape.len() - 1] != q_shape[2]
        || key_tokens == 0
        || key_tokens > k.numel() / q_shape[2]
    {
        return Err(Error::Other(
            "static inference BF16 MQA shape mismatch".into(),
        ));
    }
    #[cfg(apxinf_fa2_sm80)]
    {
        if fa2_splitkv_enabled(q_shape[0], key_tokens, q_shape[1], 1, q_shape[2]) {
            return fa2_attention_splitkv(
                ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], 1, q_shape[2], false,
            );
        }
        return fa2_attention(
            ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], 1, q_shape[2],
        );
    }
    #[cfg(apxinf_cutlass_fmha)]
    {
        return cublas_mqa_bf16(ctx, q, k, v, key_tokens);
    }
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mqa_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(all(not(apxinf_fa2_sm80), not(apxinf_cutlass_fmha)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn causal_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape() != k.shape()
        || k_shape[0] < key_tokens
        || k_shape[2] != q_shape[2]
        || k_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || key_tokens < q_shape[0]
    {
        return Err(Error::Other(
            "static inference causal BF16 GQA shape mismatch".into(),
        ));
    }
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        // Use the head256 specialization for complete causal prefill and
        // single-token decode. Unequal multi-token prefixes use composed GQA.
        if q_shape[2] == 256 {
            #[cfg(apxinf_fa2_sm80)]
            if q_shape[0] == key_tokens && q_shape[0] > 1 {
                return fa2_attention_causal(ctx,q,k,v,q_shape[0],key_tokens,q_shape[1],k_shape[1],q_shape[2]);
            }
            #[cfg(apxinf_fa2_sm80)]
            if q_shape[0] == 1 {
                return fa2_attention_splitkv(ctx,q,k,v,1,1,key_tokens,q_shape[1],k_shape[1],q_shape[2],false);
            }
            return composed_gqa_bf16(ctx, q, k, v, key_tokens, true);
        }
        if fa2_splitkv_enabled(q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2]) {
            return fa2_attention_splitkv(
                ctx, q, k, v, 1, q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2], true,
            );
        }
        return fa2_attention_causal(
            ctx, q, k, v, q_shape[0], key_tokens, q_shape[1], k_shape[1], q_shape[2],
        );
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Err(Error::Other(
        "causal BF16 GQA requires the FA2 backend".into(),
    ))
}

pub fn mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || shape[2] > 256
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other(
            "static inference BF16 MHA shape mismatch".into(),
        ));
    }
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        return fa2_attention(
            ctx,
            q,
            k,
            v,
            shape[0] / tokens_per_batch,
            tokens_per_batch,
            tokens_per_batch,
            shape[1],
            shape[1],
            shape[2],
        );
    }
    #[cfg(all(apxinf_cutlass_fmha, not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))))]
    if tokens_per_batch == 256 && shape[1] == 16 && shape[2] == 72 {
        let output = output_buffer(ctx, q.size_in_bytes())?;
        unsafe {
            let batches = shape[0] / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_bf16(
                    gpu_ptr(q)?,
                    gpu_ptr(k)?,
                    gpu_ptr(v)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    tokens_per_batch as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "CUTLASS BF16 FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_bf16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                tokens_per_batch as i32,
                shape[1] as i32,
                shape[1] as i32,
                shape[2] as i32,
                ctx.stream().handle(),
            );
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "CUTLASS BF16 FMHA execution failed with status {status}"
                )));
            }
            return Ok(make_gpu_tensor(
                q.shape().clone(),
                DType::BF16,
                ctx.device_id(),
                output,
            ));
        }
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mha_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            tokens_per_batch as i32,
            (shape[0] / tokens_per_batch) as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn segmented_mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offsets: &crate::buffer::CudaBuffer,
    host_offsets: &[u32],
    segments: usize,
    max_tokens: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || segments == 0
        || host_offsets.len() != segments + 1
        || max_tokens == 0
        || shape[2] > 256
    {
        return Err(Error::Other(
            "segmented BF16 MHA requires matching [tokens,heads,head_dim] tensors".into(),
        ));
    }
    super::contracts::require_buffers(
        ctx,
        "segmented BF16 MHA",
        &[(
            "offsets",
            offsets,
            (segments + 1) * std::mem::size_of::<u32>(),
        )],
    )?;
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    {
        // TEMP-DIAG (implement_r4): per-call gate + clock; revert in the acceptance-bound revision.
        static MHA_DIAG_CALL: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let mha_diag_call = MHA_DIAG_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mha_diag = mha_diag_call < 2; // blocks 0 and 1 are the receipt-implicated calls
        let mha_t0 = std::time::Instant::now();
        // FIX (implement_r8): route-3 composed gemm_ex + tiled-softmax per-segment attention
        // replaces the r6/r7 pad route. The vendored FA2 fwd family is measured pathological
        // on sm_89 across hdim96 IsEvenK=false (r4 ~19.3s/launch), hdim96 IsEvenK=true
        // (r6 ~19.2s) and hdim128 IsEvenK=true (r7 ~28.2s), so the head_dim==64 vision arm
        // runs on measured-healthy cuBLAS GEMMs plus the tiled row-softmax kernel; scores
        // stay fp32 end-to-end and P is rounded to bf16 for the PV MMA exactly as FA2 does.
        // Revert/replace in the acceptance-bound revision per the prevailing marker policy.
        let orig_head_dim = shape[2]; // FIX (implement_r6)
        #[cfg(apxinf_fa2_sm80)]
        if orig_head_dim == 64 {
            if host_offsets.first() != Some(&0)
                || host_offsets.last().map(|&v| v as usize) != Some(shape[0])
                || host_offsets
                    .windows(2)
                    .any(|v| v[1] < v[0] || (v[1] - v[0]) as usize > max_tokens)
                || [q, k, v]
                    .iter()
                    .any(|t| t.device() != Device::Cuda(ctx.device_id()))
            {
                return Err(Error::Other("segmented MHA offsets/device mismatch".into()));
            }
            let bytes = checked_bytes(DType::BF16, shape, "segmented head64 MHA")?;
            let output = output_buffer(ctx, bytes)?;
            let row_bytes = checked_bytes(DType::BF16, &[shape[1], 64], "head64 row")?;
            for bounds in host_offsets.windows(2) {
                let start = bounds[0] as usize;
                let tokens = (bounds[1] - bounds[0]) as usize;
                if tokens == 0 {
                    continue;
                }
                let piece = |tensor: &Tensor| -> Result<Tensor> {
                    CudaBuffer::from_tensor(tensor)
                        .map_err(Error::Cuda)?
                        .view(start * row_bytes, tokens * row_bytes)
                        .map_err(Error::Cuda)?
                        .as_tensor(Shape::new(vec![tokens, shape[1], 64]), DType::BF16)
                        .map_err(Error::Cuda)
                };
                let (sq, sk, sv) = (piece(q)?, piece(k)?, piece(v)?);
                let segment = fa2_attention_splitkv(
                    ctx, &sq, &sk, &sv, 1, tokens, tokens, shape[1], shape[1], 64, false,
                )?;
                crate::transfers::copy_tensor_2d_to_buffer(
                    ctx,
                    &segment,
                    &output,
                    start * row_bytes,
                    row_bytes,
                    row_bytes,
                    row_bytes,
                    tokens,
                )?;
            }
            return Ok(make_gpu_tensor(
                q.shape().clone(),
                DType::BF16,
                ctx.device_id(),
                output,
            ));
        }
        #[cfg(not(apxinf_fa2_sm80))]
        if orig_head_dim == 64 {
            // FIX (implement_r8): composed per-segment attention at the true dim
            let qf32_buf =
                CudaBuffer::alloc(shape[0] * shape[1] * orig_head_dim * 4, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8)
            let qf32_t = qf32_buf
                .as_tensor(q.shape().clone(), DType::F32)
                .map_err(Error::Cuda)?; // FIX (implement_r8)
            super::linear_attention::cast_bf16_to_f32(ctx, q, &qf32_t)?; // FIX (implement_r8)
            let kf32_buf =
                CudaBuffer::alloc(shape[0] * shape[1] * orig_head_dim * 4, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8)
            let kf32_t = kf32_buf
                .as_tensor(k.shape().clone(), DType::F32)
                .map_err(Error::Cuda)?; // FIX (implement_r8)
            super::linear_attention::cast_bf16_to_f32(ctx, k, &kf32_t)?; // FIX (implement_r8)
            let output = output_buffer(ctx, q.size_in_bytes())?; // FIX (implement_r8): packed [T,16,64]; out_bytes=85,327,872 signs the composed route
                                                                 // TEMP-DIAG (implement_r4): entry-alloc bracket; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!(
                    "[qwen_drive] mha_alloc_ok call={} out_bytes={} lse_bytes={} ms={}",
                    mha_diag_call,
                    q.size_in_bytes(),
                    0,
                    mha_t0.elapsed().as_millis()
                );
            }
            for (seg_idx, bounds) in host_offsets.windows(2).enumerate() {
                let start = bounds[0] as usize;
                let tokens = (bounds[1] - bounds[0]) as usize;
                if tokens == 0 {
                    continue;
                } // FIX (implement_r8): degenerate-segment guard
                  // TEMP-DIAG (implement_r4): pre-launch segment marker; revert in the acceptance-bound revision.
                if mha_diag {
                    eprintln!(
                        "[qwen_drive] mha_seg call={} i={} start={} tokens={} ms={}",
                        mha_diag_call,
                        seg_idx,
                        start,
                        tokens,
                        mha_t0.elapsed().as_millis()
                    );
                }
                let scores = CudaBuffer::alloc(shape[1] * tokens * tokens * 4, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8): fp32 [heads,tokens,tokens]
                let probs = CudaBuffer::alloc(shape[1] * tokens * tokens * 2, ctx.device_id())
                    .map_err(Error::Cuda)?; // FIX (implement_r8): bf16 [heads,tokens,tokens]
                for head in 0..shape[1] {
                    let q_head = buffer_slice(
                        &qf32_buf,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 4,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 4,
                    )?; // FIX (implement_r8)
                    let k_head = buffer_slice(
                        &kf32_buf,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 4,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 4,
                    )?; // FIX (implement_r8)
                    let scores_head =
                        buffer_slice(&scores, head * tokens * tokens * 4, tokens * tokens * 4)?; // FIX (implement_r8)
                    ctx.cublas()
                        .gemm_ex(
                            DType::F32,
                            CublasTranspose::None,
                            CublasTranspose::Transpose,
                            tokens,
                            tokens,
                            orig_head_dim,
                            (orig_head_dim as f32).sqrt().recip(),
                            &q_head,
                            (shape[1] * orig_head_dim) as i32,
                            &k_head,
                            (shape[1] * orig_head_dim) as i32,
                            0.0,
                            &scores_head,
                            tokens as i32,
                        )
                        .map_err(Error::Cuda)?; // FIX (implement_r8): Q*K^T; alpha folds the softmax scale bound to the true dim 64
                }
                unsafe {
                    ffi::check_cuda(ffi::apxinf_static_row_softmax_f32_bf16(
                        scores.ptr() as *const std::ffi::c_void,
                        probs.ptr(),
                        tokens as u32,
                        (shape[1] * tokens) as u32,
                        ctx.stream().handle(),
                    ))
                    .map_err(Error::Cuda)?; // FIX (implement_r8): tiled block-per-row softmax, fp32 in / bf16 out
                }
                for head in 0..shape[1] {
                    let probs_head =
                        buffer_slice(&probs, head * tokens * tokens * 2, tokens * tokens * 2)?; // FIX (implement_r8)
                    let v_head = tensor_slice(
                        v,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 2,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 2,
                        ctx.device_id(),
                    )?; // FIX (implement_r8)
                    let out_head = buffer_slice(
                        &output,
                        (start * shape[1] * orig_head_dim + head * orig_head_dim) * 2,
                        ((tokens - 1) * shape[1] * orig_head_dim + orig_head_dim) * 2,
                    )?; // FIX (implement_r8)
                    ctx.cublas()
                        .gemm_ex(
                            DType::BF16,
                            CublasTranspose::None,
                            CublasTranspose::None,
                            tokens,
                            orig_head_dim,
                            tokens,
                            1.0,
                            &probs_head,
                            tokens as i32,
                            &v_head,
                            (shape[1] * orig_head_dim) as i32,
                            0.0,
                            &out_head,
                            (shape[1] * orig_head_dim) as i32,
                        )
                        .map_err(Error::Cuda)?; // FIX (implement_r8): P*V straight into the packed [T,16,64] output
                }
                // TEMP-DIAG (implement_r4): post-launch retirement probe; converts an async fault into a loud Err; revert in the acceptance-bound revision.
                if mha_diag {
                    ctx.synchronize().map_err(Error::Cuda)?;
                    eprintln!(
                        "[qwen_drive] mha_seg_ok call={} i={} ms={}",
                        mha_diag_call,
                        seg_idx,
                        mha_t0.elapsed().as_millis()
                    );
                }
            }
            // TEMP-DIAG (implement_r4): separates "12 launches retired" from "epilogue cudaFree wedged"; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!(
                    "[qwen_drive] mha_loop_done call={} ms={}",
                    mha_diag_call,
                    mha_t0.elapsed().as_millis()
                );
            }
            let result = make_gpu_tensor(q.shape().clone(), DType::BF16, ctx.device_id(), output);
            return Ok(result);
        }
        let output = output_buffer(ctx, q.size_in_bytes())?;
        let softmax_lse = output_buffer(ctx, shape[0] * shape[1] * std::mem::size_of::<f32>())?;
        // TEMP-DIAG (implement_r4): entry-alloc bracket; revert in the acceptance-bound revision.
        if mha_diag {
            eprintln!(
                "[qwen_drive] mha_alloc_ok call={} out_bytes={} lse_bytes={} ms={}",
                mha_diag_call,
                q.size_in_bytes(),
                shape[0] * shape[1] * std::mem::size_of::<f32>(),
                mha_t0.elapsed().as_millis()
            );
        }
        let row_bytes = shape[1] * shape[2] * DType::BF16.size_in_bytes();
        let lse_row_bytes = shape[1] * std::mem::size_of::<f32>();
        for (seg_idx, bounds) in host_offsets.windows(2).enumerate() {
            let start = bounds[0] as usize;
            let tokens = (bounds[1] - bounds[0]) as usize;
            // TEMP-DIAG (implement_r4): pre-launch segment marker; revert in the acceptance-bound revision.
            if mha_diag {
                eprintln!(
                    "[qwen_drive] mha_seg call={} i={} start={} tokens={} ms={}",
                    mha_diag_call,
                    seg_idx,
                    start,
                    tokens,
                    mha_t0.elapsed().as_millis()
                );
            }
            unsafe {
                ffi::check_cuda(ffi::apxinf_static_fa2_bf16(
                    gpu_ptr(q)?.cast::<u8>().add(start * row_bytes).cast(),
                    gpu_ptr(k)?.cast::<u8>().add(start * row_bytes).cast(),
                    gpu_ptr(v)?.cast::<u8>().add(start * row_bytes).cast(),
                    output.ptr().cast::<u8>().add(start * row_bytes).cast(),
                    softmax_lse
                        .ptr()
                        .cast::<u8>()
                        .add(start * lse_row_bytes)
                        .cast(),
                    1,
                    tokens as i32,
                    tokens as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    (shape[2] as f32).sqrt().recip(),
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
            // TEMP-DIAG (implement_r4): post-launch retirement probe; converts an async fault into a loud Err; revert in the acceptance-bound revision.
            if mha_diag {
                ctx.synchronize().map_err(Error::Cuda)?;
                eprintln!(
                    "[qwen_drive] mha_seg_ok call={} i={} ms={}",
                    mha_diag_call,
                    seg_idx,
                    mha_t0.elapsed().as_millis()
                );
            }
        }
        // TEMP-DIAG (implement_r4): separates "12 launches retired" from "epilogue cudaFree wedged"; revert in the acceptance-bound revision.
        if mha_diag {
            eprintln!(
                "[qwen_drive] mha_loop_done call={} ms={}",
                mha_diag_call,
                mha_t0.elapsed().as_millis()
            );
        }
        let result = make_gpu_tensor(q.shape().clone(), DType::BF16, ctx.device_id(), output);
        return Ok(result);
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_segmented_mha_bf16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            offsets.ptr(),
            output.ptr(),
            segments as i32,
            max_tokens as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100)))]
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
pub(crate) fn cublas_mqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let status = unsafe {
        ffi::apxinf_static_cublas_mqa_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            q_shape[2] as i32,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

#[cfg(apxinf_fa2_f16_sm100)]
pub(crate) fn fa2_mqa_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let output = output_buffer(ctx, q.size_in_bytes())?;
    let lse_elements = q_shape[0]
        .checked_mul(q_shape[1])
        .ok_or_else(|| Error::Other("FA2 FP16 LSE size overflow".into()))?;
    let softmax_lse = output_buffer(
        ctx,
        lse_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Other("FA2 FP16 LSE byte size overflow".into()))?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_fa2_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            softmax_lse.ptr(),
            1,
            q_shape[0] as i32,
            key_tokens as i32,
            q_shape[1] as i32,
            1,
            q_shape[2] as i32,
            (q_shape[2] as f32).sqrt().recip(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

/// Exact MQA for the static inference action expert. Prefix and suffix
/// K/V are `[tokens, head_dim]`; Q/output are `[suffix, heads, head_dim]`.
pub fn mqa_prefix_suffix_f16(
    ctx: &CudaContext,
    q: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    suffix_k: &Tensor,
    suffix_v: &Tensor,
) -> Result<Tensor> {
    for tensor in [q, prefix_k, prefix_v, suffix_k, suffix_v] {
        if tensor.dtype() != DType::F16 {
            return Err(Error::DTypeMismatch {
                expected: DType::F16,
                got: tensor.dtype(),
            });
        }
    }
    let q_shape = q.shape().dims();
    let prefix_shape = prefix_k.shape().dims();
    let suffix_shape = suffix_k.shape().dims();
    if q_shape.len() != 3
        || prefix_shape.len() != 2
        || suffix_shape.len() != 2
        || prefix_v.shape().dims() != prefix_shape
        || suffix_v.shape().dims() != suffix_shape
        || q_shape[0] != suffix_shape[0]
        || q_shape[2] != prefix_shape[1]
        || q_shape[2] != suffix_shape[1]
    {
        return Err(Error::Other(format!(
            "static inference MQA shape mismatch: q={q_shape:?}, prefix={prefix_shape:?}, suffix={suffix_shape:?}"
        )));
    }
    let combined_k = concat_rows_f16(ctx, prefix_k, suffix_k)?;
    let combined_v = concat_rows_f16(ctx, prefix_v, suffix_v)?;
    cublas_mqa_f16(
        ctx,
        q,
        &combined_k,
        &combined_v,
        prefix_shape[0] + suffix_shape[0],
    )
}

/// MQA over an already contiguous K/V cache. This avoids rebuilding
/// `[prefix, suffix]` buffers at every action-expert layer and flow step.
pub fn mqa_cached_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    for tensor in [q, k_cache, v_cache] {
        if tensor.dtype() != DType::F16 {
            return Err(Error::DTypeMismatch {
                expected: DType::F16,
                got: tensor.dtype(),
            });
        }
    }
    let q_shape = q.shape().dims();
    let cache_shape = k_cache.shape().dims();
    if q_shape.len() != 3
        || cache_shape.len() != 2
        || v_cache.shape().dims() != cache_shape
        || cache_shape[0] != key_tokens
        || q_shape[2] != cache_shape[1]
    {
        return Err(Error::Other(format!(
            "static inference cached MQA shape mismatch: q={q_shape:?}, cache={cache_shape:?}, key_tokens={key_tokens}"
        )));
    }

    cublas_mqa_f16(ctx, q, k_cache, v_cache, key_tokens)
}

/// Split a biased dense QKV projection into `[tokens, heads, head_dim]`
/// tensors without RoPE. This is the SigLIP attention layout.
pub fn split_qkv_bias_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let (tokens, width) = matrix_shape(qkv, "vision QKV split")?;
    let projection_width = heads * head_dim;
    if qkv.dtype() != DType::F16
        || width != 3 * projection_width
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [width])
    {
        return Err(Error::Other(format!(
            "static inference vision QKV expected FP16 [tokens,{}], got {:?}",
            3 * projection_width,
            qkv.shape().dims()
        )));
    }
    let q_buffer = f16_output(ctx, tokens, projection_width)?;
    let k_buffer = f16_output(ctx, tokens, projection_width)?;
    let v_buffer = f16_output(ctx, tokens, projection_width)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qkv_split_bias_f16(
            gpu_ptr(qkv)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            tokens as i32,
            projection_width as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    Ok(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::F16, ctx.device_id(), q_buffer),
        k: make_gpu_tensor(shape.clone(), DType::F16, ctx.device_id(), k_buffer),
        v: make_gpu_tensor(shape, DType::F16, ctx.device_id(), v_buffer),
    })
}

/// Apply QKV bias without splitting the projection, then let SM100-family
/// FMHA consume Q/K/V through row-strided views of `[tokens, 3 * heads * dim]`.
/// This avoids materializing three layout copies for SigLIP attention.
pub fn mha_packed_qkv_bias_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    tokens_per_batch: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (tokens, width) = matrix_shape(qkv, "packed vision QKV")?;
    let projection_width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("packed vision QKV width overflow".into()))?;
    let packed_width = projection_width
        .checked_mul(3)
        .ok_or_else(|| Error::Other("packed vision QKV width overflow".into()))?;
    if qkv.dtype() != DType::F16
        || width != packed_width
        || tokens_per_batch == 0
        || tokens % tokens_per_batch != 0
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [width])
    {
        return Err(Error::Other(format!(
            "packed vision QKV expects FP16 [tokens,{}], got {:?}",
            packed_width,
            qkv.shape().dims()
        )));
    }

    #[cfg(apxinf_cutlass_fmha)]
    if tokens_per_batch == 256 && heads == 16 && head_dim == 72 {
        let biased = bias
            .map(|value| bias_f16(ctx, qkv, Some(value)))
            .transpose()?;
        let packed = biased.as_ref().unwrap_or(qkv);
        let output = f16_output(ctx, tokens, projection_width)?;
        unsafe {
            let batches = tokens / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_packed_qkv_f16(
                    gpu_ptr(packed)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    heads as i32,
                    head_dim as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "packed CUTLASS FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_packed_qkv_f16(
                gpu_ptr(packed)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                heads as i32,
                head_dim as i32,
                ctx.stream().handle(),
            );
            if status == 0 {
                return Ok(make_gpu_tensor(
                    Shape::new(vec![tokens, heads, head_dim]),
                    DType::F16,
                    ctx.device_id(),
                    output,
                ));
            }
        }
    }

    let split = split_qkv_bias_f16(ctx, qkv, bias, heads, head_dim)?;
    mha_f16(ctx, &split.q, &split.k, &split.v, tokens_per_batch)
}

pub fn mha_f16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = q.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || shape.len() != 3
        || k.shape() != q.shape()
        || v.shape() != q.shape()
        || shape[2] > 256
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other(
            "static inference MHA expects matching FP16 [tokens,heads,head_dim] tensors".into(),
        ));
    }
    let output = output_buffer(ctx, q.size_in_bytes())?;
    #[cfg(apxinf_cutlass_fmha)]
    if tokens_per_batch == 256 && shape[1] == 16 && shape[2] == 72 {
        unsafe {
            let batches = shape[0] / tokens_per_batch;
            if may_prepare_native_resources() {
                let status = ffi::apxinf_static_prepare_cutlass_mha_f16(
                    gpu_ptr(q)?,
                    gpu_ptr(k)?,
                    gpu_ptr(v)?,
                    output.ptr(),
                    batches as i32,
                    tokens_per_batch as i32,
                    tokens_per_batch as i32,
                    shape[1] as i32,
                    shape[1] as i32,
                    shape[2] as i32,
                    ctx.stream().handle(),
                );
                if status != 0 {
                    return Err(Error::Cuda(format!(
                        "CUTLASS FMHA resource preparation failed with status {status}"
                    )));
                }
            }
            let status = ffi::apxinf_static_cutlass_mha_f16(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                batches as i32,
                tokens_per_batch as i32,
                shape[1] as i32,
                shape[1] as i32,
                shape[2] as i32,
                ctx.stream().handle(),
            );
            if status == 0 {
                return Ok(make_gpu_tensor(
                    Shape::new(vec![tokens, heads, head_dim]),
                    DType::F16,
                    ctx.device_id(),
                    output,
                ));
            }
        }
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_mha_flash_f16(
            gpu_ptr(q)?,
            gpu_ptr(k)?,
            gpu_ptr(v)?,
            output.ptr(),
            tokens_per_batch as i32,
            (shape[0] / tokens_per_batch) as i32,
            shape[1] as i32,
            shape[2] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        q.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

pub fn mqa_f16(ctx: &CudaContext, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape().dims() != k_shape
        || k_shape[0] != q_shape[0]
        || k_shape[1] != 1
        || k_shape[2] != q_shape[2]
    {
        return Err(Error::Other(
            "static inference self MQA expects Q [T,H,D], K/V [T,1,D] FP16".into(),
        ));
    }
    #[cfg(apxinf_fa2_f16_sm100)]
    {
        return fa2_mqa_f16(ctx, q, k, v, k_shape[0]);
    }
    #[cfg(not(apxinf_fa2_f16_sm100))]
    cublas_mqa_f16(ctx, q, k, v, k_shape[0])
}

pub fn mqa_f16_e4m3_522(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output_scale: f32,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if q.dtype() != DType::F16
        || k.dtype() != DType::F16
        || v.dtype() != DType::F16
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape().dims() != k_shape
        || q_shape != [522, 8, 256]
        || k_shape != [522, 1, 256]
        || v.shape().dims() != k_shape
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "FA2 direct E4M3 requires FP16 Q [522,8,256], K/V [522,1,256] and finite positive scale; got q={q_shape:?}, k={k_shape:?}, v={:?}, scale={output_scale}",
            v.shape().dims()
        )));
    }
    #[cfg(apxinf_fa2_direct_e4m3_sm100)]
    {
        let output = output_buffer(ctx, q.numel())?;
        let lse_elements = q_shape[0]
            .checked_mul(q_shape[1])
            .ok_or_else(|| Error::Other("FA2 direct E4M3 LSE size overflow".into()))?;
        let softmax_lse = output_buffer(
            ctx,
            lse_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Other("FA2 direct E4M3 LSE byte size overflow".into()))?,
        )?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_fa2_f16_direct_e4m3_522(
                gpu_ptr(q)?,
                gpu_ptr(k)?,
                gpu_ptr(v)?,
                output.ptr(),
                softmax_lse.ptr(),
                1,
                522,
                522,
                8,
                1,
                256,
                output_scale,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(make_gpu_tensor(
            q.shape().clone(),
            DType::F8E4M3,
            ctx.device_id(),
            output,
        ));
    }
    #[cfg(not(apxinf_fa2_direct_e4m3_sm100))]
    Err(Error::Other(
        "FA2 direct E4M3 requires an SM100-family FA2 build".into(),
    ))
}
