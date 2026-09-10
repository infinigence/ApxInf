use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::activation::{
    bias_gelu_bf16, bias_relu_bf16, bias_silu_bf16, gelu_tanh, silu, silu_mul_bf16,
};
use crate::kernels::attention::{
    causal_gqa_prefill_bf16, causal_mask, noncausal, softmax, softmax_causal, split_qkv_bias_bf16,
    vision,
};
use crate::kernels::cache::append;
use crate::kernels::elementwise::{
    add, add_bias, bias_bf16, bias_qkv_in_place_bf16, contiguous_rows, gather_rows_bf16,
    gather_rows_bf16_prepared, mul, prepare_row_indices, scale, scatter_rows_bf16,
    scatter_rows_bf16_prepared,
};
use crate::kernels::embedding::lookup;
use crate::kernels::fused::{
    bias_residual_bf16, bias_residual_layer_bf16, bias_then_residual_bf16,
};
use crate::kernels::norm::{adaptive_layer, layer, rms};
use crate::kernels::rope::{
    apply, apply_batched, apply_mrope, apply_mrope_precomputed, apply_vision_2d,
    apply_vision_2d_pair, prepare_mrope_cos_sin, prepare_vision_rope_cos_sin,
    split_qkv_bias_apply_vision_2d, split_qkv_bias_apply_vision_2d_precomputed,
};

fn gpu_ptr(tensor: &Tensor) -> Result<*mut std::ffi::c_void> {
    Ok(CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?.ptr())
}

fn make_gpu_tensor(shape: Shape, dtype: DType, _device: usize, buffer: CudaBuffer) -> Tensor {
    buffer.into_tensor(shape, dtype)
}
use crate::test_util::{
    assert_bf16_close_elementwise, assert_bf16_close_reduction, download_bf16_as_fp32,
    upload_fp32_as_bf16,
};

fn silu_ref(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

fn report_error_metrics(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut absolute_sum = 0.0f64;
    let mut nan_count = 0usize;
    let mut infinity_count = 0usize;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if actual.is_nan() {
            nan_count += 1;
        }
        if actual.is_infinite() {
            infinity_count += 1;
        }
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1e-6);
        max_abs = max_abs.max(absolute);
        max_relative = max_relative.max(relative);
        absolute_sum += f64::from(absolute);
    }
    let mean_absolute = absolute_sum / actual.len() as f64;
    eprintln!(
        "{name}: max_abs={max_abs:.8e} max_relative={max_relative:.8e} mean_abs={mean_absolute:.8e} nan={nan_count} inf={infinity_count}"
    );
}

#[test]
fn silu_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // A mix of magnitudes and signs so we exercise the tails of exp/sigmoid.
    let input: Vec<f32> = (-32..32).map(|i| (i as f32) * 0.25).collect();
    let expected: Vec<f32> = input.iter().map(|&x| silu_ref(x)).collect();

    let bf_in = upload_fp32_as_bf16(&ctx, &input, vec![input.len()]).unwrap();
    let bf_out = silu(&ctx, &bf_in).unwrap();
    let actual = download_bf16_as_fp32(&bf_out).unwrap();

    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn silu_mul_separate_packed4_bf16_matches_scalar_kernel() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Exercise several grid-stride iterations as well as non-trivial BF16
    // rounding in both the SiLU intermediate and the final product.
    let count = 4096usize + 12;
    let gate: Vec<f32> = (0..count)
        .map(|i| (((i as f32) * 0.037).sin() * 9.0) - 0.25)
        .collect();
    let up: Vec<f32> = (0..count)
        .map(|i| (((i as f32) * 0.019).cos() * 3.5) + 0.125)
        .collect();
    let gate = upload_fp32_as_bf16(&ctx, &gate, vec![count]).unwrap();
    let up = upload_fp32_as_bf16(&ctx, &up, vec![count]).unwrap();

    // SAFETY: this CUDA-only test is the sole test that mutates this
    // candidate-specific selector, and restores it before returning.
    unsafe { std::env::remove_var("APXINF_SILU_MUL_SEPARATE_BF16_PACKED4") };
    let scalar = silu_mul_bf16(&ctx, &gate, &up).unwrap();
    unsafe { std::env::set_var("APXINF_SILU_MUL_SEPARATE_BF16_PACKED4", "1") };
    let packed4 = silu_mul_bf16(&ctx, &gate, &up).unwrap();
    unsafe { std::env::remove_var("APXINF_SILU_MUL_SEPARATE_BF16_PACKED4") };

    let scalar = download_bf16_as_fp32(&scalar).unwrap();
    let packed4 = download_bf16_as_fp32(&packed4).unwrap();
    assert_eq!(packed4, scalar, "packed4 SiLU-mul must be bit-identical");
}

// ── Elementwise: add ──────────────────────────────────────────────

#[test]
fn add_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 6.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.05 + 3.2).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let ta = upload_fp32_as_bf16(&ctx, &a, vec![n]).unwrap();
    let tb = upload_fp32_as_bf16(&ctx, &b, vec![n]).unwrap();
    let out = add(&ctx, &ta, &tb).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Elementwise: mul ──────────────────────────────────────────────

#[test]
fn mul_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 64;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

    let ta = upload_fp32_as_bf16(&ctx, &a, vec![n]).unwrap();
    let tb = upload_fp32_as_bf16(&ctx, &b, vec![n]).unwrap();
    let out = mul(&ctx, &ta, &tb).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Elementwise: scale ────────────────────────────────────────────

#[test]
fn scale_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 100;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let factor = 0.25f32;
    let expected: Vec<f32> = input.iter().map(|x| x * factor).collect();

    let t = upload_fp32_as_bf16(&ctx, &input, vec![n]).unwrap();
    let out = scale(&ctx, &t, factor).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Reduction: rms_norm ───────────────────────────────────────────

#[test]
fn rms_norm_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (2usize, 64usize);
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let eps = 1e-5f32;

    // Reference computation
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let row = &input[off..off + cols];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for i in 0..cols {
            expected[off + i] = row[i] * inv_rms * weight[i];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_w = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let out = rms(&ctx, &t_in, &t_w, eps).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Reduction: softmax ────────────────────────────────────────────

#[test]
fn softmax_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (3usize, 32usize);
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();

    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let row = &input[off..off + cols];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max_v).exp()).sum();
        for i in 0..cols {
            expected[off + i] = (row[i] - max_v).exp() / sum;
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = softmax(&ctx, &t_in).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── RoPE (batched, half-split) ────────────────────────────────────

#[test]
fn rope_batched_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 2usize, 8usize);
    let theta = 10000.0f32;
    let pos_offset = 3u32;

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.1).sin() * 2.0)
        .collect();

    // fp32 reference (half-split): pair (i, i + head_dim/2)
    let mut expected = vec![0.0f32; input.len()];
    let half = head_dim / 2;
    for s in 0..seq_len {
        let pos = pos_offset as usize + s;
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..half {
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + pair];
                let x1 = input[base + half + pair];
                expected[base + pair] = x0 * c - x1 * sn;
                expected[base + half + pair] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let out = apply_batched(&ctx, &t_in, n_heads, head_dim, theta, pos_offset).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── RoPE (interleaved pairs) ──────────────────────────────────────

#[test]
fn rope_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (1usize, 2usize, 8usize);
    let theta = 10000.0f32;
    let pos_offset = 5u32;

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.2).cos())
        .collect();

    // fp32 reference for the interleaved (2i, 2i+1) variant
    let mut expected = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        let pos = pos_offset as usize + s;
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..head_dim / 2 {
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + 2 * pair];
                let x1 = input[base + 2 * pair + 1];
                expected[base + 2 * pair] = x0 * c - x1 * sn;
                expected[base + 2 * pair + 1] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let out = apply(&ctx, &t_in, n_heads, head_dim, theta, pos_offset).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Embedding lookup ──────────────────────────────────────────────

#[test]
fn embedding_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (vocab, embed_dim) = (16usize, 8usize);
    let seq = [3u32, 0u32, 15u32];
    let table: Vec<f32> = (0..vocab * embed_dim)
        .map(|i| (i as f32) * 0.01 - 1.0)
        .collect();

    let mut expected = Vec::with_capacity(seq.len() * embed_dim);
    for &tid in &seq {
        let off = tid as usize * embed_dim;
        expected.extend_from_slice(&table[off..off + embed_dim]);
    }

    // Upload table as bf16 and ids as raw u32 buffer.
    let t_table = upload_fp32_as_bf16(&ctx, &table, vec![vocab, embed_dim]).unwrap();
    let ids_bytes: Vec<u8> = seq.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let ids_buf = crate::buffer::CudaBuffer::alloc(ids_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    ids_buf
        .copy_from_host(&ids_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = lookup(&ctx, &t_table, &ids_buf, seq.len()).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn embedding_bf16_lookup_can_be_captured() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (vocab, embed_dim) = (16usize, 8usize);
    let seq = [3u32, 0u32, 15u32];
    let table = (0..vocab * embed_dim)
        .map(|i| (i as f32) * 0.01 - 1.0)
        .collect::<Vec<_>>();
    let t_table = upload_fp32_as_bf16(&ctx, &table, vec![vocab, embed_dim]).unwrap();
    let ids_bytes = seq
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let ids = CudaBuffer::alloc(ids_bytes.len(), 0).unwrap();
    ids.copy_from_host(&ids_bytes).unwrap();
    let workspace = crate::workspace::GraphWorkspace::new(4096, 0).unwrap();

    let eager = crate::workspace::prepare_with_workspace(&workspace, || {
        lookup(&ctx, &t_table, &ids, seq.len())
    })
    .unwrap();
    ctx.synchronize().unwrap();
    drop(eager);

    crate::graph::begin(&ctx, crate::graph::CaptureMode::ThreadLocal).unwrap();
    let captured =
        crate::workspace::with_workspace(&workspace, || lookup(&ctx, &t_table, &ids, seq.len()))
            .unwrap();
    let graph = crate::graph::end(&ctx).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    let actual = download_bf16_as_fp32(&captured).unwrap();
    let mut expected = Vec::with_capacity(seq.len() * embed_dim);
    for &token in &seq {
        let offset = token as usize * embed_dim;
        expected.extend_from_slice(&table[offset..offset + embed_dim]);
    }
    assert_bf16_close_elementwise(&actual, &expected);
}

// ── Causal mask ───────────────────────────────────────────────────

#[test]
fn causal_mask_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (4usize, 6usize);
    let kv_offset = 0u32;
    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
    // Expected: below the diagonal + kv_offset stays, above becomes -inf.
    let mut expected = input.clone();
    for r in 0..rows {
        for c in 0..cols {
            if c > r + kv_offset as usize {
                expected[r * cols + c] = f32::NEG_INFINITY;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = causal_mask(&ctx, &t_in, kv_offset).unwrap();
    let got = download_bf16_as_fp32(&out).unwrap();
    // Special-case -inf comparison (any tolerance fails for infinities).
    for i in 0..rows * cols {
        if expected[i].is_infinite() {
            assert!(
                got[i].is_infinite() && got[i].is_sign_negative(),
                "expected -inf at {i}, got {}",
                got[i]
            );
        } else {
            assert!(
                (got[i] - expected[i]).abs() <= 1e-3 + 1e-2 * expected[i].abs(),
                "idx {i}: got {}, expected {}",
                got[i],
                expected[i]
            );
        }
    }
}

// ── Attention softmax (fused causal + softmax) ────────────────────

#[test]
fn attention_softmax_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, kv_len) = (2usize, 3usize, 5usize);
    let rows = seq_len * n_heads;
    let cols = kv_len;
    let kv_offset = 0u32;
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) % 7.0) * 0.3 - 1.0)
        .collect();

    // Reference: for each row, seq_pos = row / n_heads; valid_cols = min(seq_pos + kv_offset + 1, cols).
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let seq_pos = r / n_heads;
        let valid = (seq_pos + kv_offset as usize + 1).min(cols);
        let row = &input[r * cols..r * cols + cols];
        let max_v = row[..valid]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row[..valid].iter().map(|x| (x - max_v).exp()).sum();
        for c in 0..cols {
            if c < valid {
                expected[r * cols + c] = (row[c] - max_v).exp() / sum;
            } else {
                expected[r * cols + c] = 0.0;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = softmax_causal(&ctx, &t_in, kv_offset, n_heads as u32).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── KV cache append ───────────────────────────────────────────────

#[test]
fn kv_cache_append_bf16_writes_correct_slot() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_kv_heads, head_dim, max_seq_len) = (2usize, 4usize, 16usize);
    let seq_len = 3usize; // current cache position (append starts here)
    let append_len = 2usize;

    // Fresh zero cache, one layer.
    let cache_bytes = n_kv_heads * max_seq_len * head_dim * 2;
    let cache_buf = crate::buffer::CudaBuffer::alloc_zeros(cache_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();

    // New data layout: [append_len, n_kv_heads, head_dim]
    let new_data: Vec<f32> = (0..append_len * n_kv_heads * head_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let new_t =
        upload_fp32_as_bf16(&ctx, &new_data, vec![append_len, n_kv_heads, head_dim]).unwrap();

    append(
        &ctx,
        &cache_buf,
        &new_t,
        n_kv_heads,
        head_dim,
        max_seq_len,
        seq_len,
        append_len,
    )
    .unwrap();

    // Read the cache back and validate the written slot.
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize()).unwrap();
    }
    let mut cache_host = vec![0u8; cache_bytes];
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            cache_host.as_mut_ptr() as *mut std::ffi::c_void,
            cache_buf.ptr() as *const std::ffi::c_void,
            cache_bytes,
            crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))
        .unwrap();
    }

    // Interpret as bf16 → fp32 host slice.
    let cache_bf: Vec<half::bf16> = cache_host
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect();
    // For each (s, h, d), cache[h * max_seq_len * head_dim + (seq_len+s)*head_dim + d]
    // should equal new_data[s*n_kv_heads*head_dim + h*head_dim + d].
    for s in 0..append_len {
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let cache_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
                let src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
                let got = cache_bf[cache_idx].to_f32();
                let want = new_data[src_idx];
                assert!(
                    (got - want).abs() < 1e-2,
                    "cache[{cache_idx}] got {got}, want {want}"
                );
            }
        }
    }
}

// ── Decode-pos kernel variants (rope_decode, attn_softmax_decode, kv_cache_append_decode) ──

#[test]
fn rope_decode_bf16_matches_rope_bf16() {
    // The decode kernel reads pos from a device buffer, seq_len=1 implicitly.
    // Correctness: match the batched form at seq_len=1.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, head_dim) = (2usize, 8usize);
    let theta = 10000.0f32;
    let pos = 4u32;

    let input: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.1).collect();

    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![1, n_heads, head_dim]).unwrap();
    let expected_out = apply_batched(&ctx, &t_ref, n_heads, head_dim, theta, pos).unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Run decode kernel directly through FFI.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, head_dim]).unwrap();
    let out_bytes = t_in.size_in_bytes();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(out_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();
    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_rope_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            theta,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, head_dim]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn attention_softmax_decode_bf16_matches_full() {
    // Decode variant is a special case of attention_softmax with rows=n_heads.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, kv_len) = (3usize, 6usize);
    let pos = 4u32; // valid_cols = pos + 1 = 5
    let input: Vec<f32> = (0..n_heads * kv_len)
        .map(|i| ((i as f32) % 5.0) * 0.4 - 1.0)
        .collect();

    // Reference: attention_softmax with rows=n_heads, kv_offset=pos, n_heads=n_heads.
    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, kv_len]).unwrap();
    let expected_out = softmax_causal(&ctx, &t_ref, pos, n_heads as u32).unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Run decode kernel directly.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, kv_len]).unwrap();
    let out_bytes = t_in.size_in_bytes();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(out_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();
    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_attention_softmax_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            kv_len as u32,
            n_heads as u32,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, kv_len]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn kv_cache_append_decode_bf16_writes_correct_slot() {
    // Decode variant: 1 row of new data, position from device buffer.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_kv_heads, head_dim, max_seq_len) = (2usize, 4usize, 16usize);
    let pos = 5u32;

    let cache_bytes = n_kv_heads * max_seq_len * head_dim * 2;
    let cache_buf = crate::buffer::CudaBuffer::alloc_zeros(cache_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();

    // new_data shape: [n_kv_heads, head_dim] (no leading append_len)
    let new_data: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let new_t = upload_fp32_as_bf16(&ctx, &new_data, vec![n_kv_heads, head_dim]).unwrap();

    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_kv_cache_append_decode_bf16(
            cache_buf.ptr(),
            gpu_ptr(&new_t).unwrap(),
            n_kv_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize()).unwrap();
    }

    let mut cache_host = vec![0u8; cache_bytes];
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            cache_host.as_mut_ptr() as *mut std::ffi::c_void,
            cache_buf.ptr() as *const std::ffi::c_void,
            cache_bytes,
            crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))
        .unwrap();
    }

    let cache_bf: Vec<half::bf16> = cache_host
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect();
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let cache_idx = h * max_seq_len * head_dim + (pos as usize) * head_dim + d;
            let src_idx = h * head_dim + d;
            let got = cache_bf[cache_idx].to_f32();
            let want = new_data[src_idx];
            assert!(
                (got - want).abs() < 1e-2,
                "cache[{cache_idx}] got {got}, want {want}"
            );
        }
    }
}

// ── mRoPE (Qwen3-VL) ──────────────────────────────────────────────

/// Reference implementation mirroring HF `apply_interleaved_mrope`
/// (rotate_half + axis-per-pair lookup). Used as the ground truth in
/// unit tests below.
fn mrope_reference(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &[[u32; 3]],
) -> Vec<f32> {
    assert_eq!(pos_ids.len(), seq_len);
    let half = head_dim / 2;
    let mut out = vec![0.0f32; input.len()];
    let (sec_h, sec_w) = (sections[1], sections[2]);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..half {
                let axis = if pair % 3 == 1 && pair < sec_h * 3 {
                    1
                } else if pair % 3 == 2 && pair < sec_w * 3 {
                    2
                } else {
                    0
                };
                let pos = pos_ids[s][axis];
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + pair];
                let x1 = input[base + half + pair];
                out[base + pair] = x0 * c - x1 * sn;
                out[base + half + pair] = x0 * sn + x1 * c;
            }
        }
    }
    out
}

#[test]
fn rope_mrope_bf16_matches_reference_text_only() {
    // With pos_ids = (i, i, i) for every token, mRoPE degenerates to
    // 1-D RoPE with rotate_half. Verifies the axis dispatch is a no-op
    // when all axes are equal, which is the text-only case.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (3usize, 2usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();

    let pos_ids: Vec<[u32; 3]> = (0..seq_len)
        .map(|i| [i as u32, i as u32, i as u32])
        .collect();
    let expected = mrope_reference(
        &input, seq_len, n_heads, head_dim, theta, sections, &pos_ids,
    );

    // Upload input and pos_ids buffer to device.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = apply_mrope(&ctx, &t_in, n_heads, head_dim, theta, sections, &pos_buf).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn rope_mrope_bf16_matches_reference_distinct_axes() {
    // Distinct (t, h, w) per token — exercises the axis dispatch. The
    // T section (24 pairs; the leftover) is exercised by the tail
    // pair_idx >= 60 which always falls through to T regardless of
    // pair_idx % 3.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 4usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];
    let pos_ids: Vec<[u32; 3]> = vec![[7, 3, 11], [8, 4, 12]];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (((i as f32) * 0.03).cos() - 0.1) * 0.5)
        .collect();

    let expected = mrope_reference(
        &input, seq_len, n_heads, head_dim, theta, sections, &pos_ids,
    );

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = apply_mrope(&ctx, &t_in, n_heads, head_dim, theta, sections, &pos_buf).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn precomputed_mrope_bf16_matches_reference_distinct_axes() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (3usize, 16usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];
    let pos_ids: Vec<[u32; 3]> = vec![[7, 3, 11], [8, 4, 12], [9, 5, 13]];
    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (((i as f32) * 0.017).cos() - 0.2) * 0.75)
        .collect();
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|triple| triple.iter().flat_map(|&value| value.to_ne_bytes()))
        .collect();
    let pos_buf = CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let table = prepare_mrope_cos_sin(&ctx, seq_len, head_dim, theta, sections, &pos_buf).unwrap();
    let reference = apply_mrope(&ctx, &t_in, n_heads, head_dim, theta, sections, &pos_buf).unwrap();
    let out = apply_mrope_precomputed(&ctx, &t_in, n_heads, head_dim, &table).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    let expected = download_bf16_as_fp32(&reference).unwrap();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let dot = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| a * b)
        .sum::<f32>();
    let actual_l2 = actual.iter().map(|value| value * value).sum::<f32>().sqrt();
    let expected_l2 = expected
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let cosine = dot / (actual_l2 * expected_l2);
    assert!(max_abs <= 0.0078125, "precomputed mRoPE max abs={max_abs}");
    assert!(cosine >= 0.99999, "precomputed mRoPE cosine={cosine}");
}

#[test]
fn rope_mrope_decode_bf16_matches_batched_seq1() {
    // Decode kernel: seq_len=1 implicitly, pos_ids buffer is [3] u32.
    // Must match rope_mrope at seq_len=1.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, head_dim) = (4usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];
    let pos_ids = [[9u32, 5, 13]];

    let input: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.01).collect();

    // Reference via batched path (seq_len=1).
    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![1, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf_batched = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf_batched
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let expected_out = apply_mrope(
        &ctx,
        &t_ref,
        n_heads,
        head_dim,
        theta,
        sections,
        &pos_buf_batched,
    )
    .unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Decode kernel direct-FFI, [3] pos buffer.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, head_dim]).unwrap();
    let pos_bytes3: Vec<u8> = pos_ids[0].iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let pos_buf_dec = crate::buffer::CudaBuffer::alloc(pos_bytes3.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf_dec
        .copy_from_host(&pos_bytes3)
        .map_err(Error::Cuda)
        .unwrap();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(t_in.size_in_bytes(), 0)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_rope_mrope_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            theta,
            pos_buf_dec.ptr(),
            sections[1] as u32,
            sections[2] as u32,
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, head_dim]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

// ── LayerNorm / GELU-tanh / add-bias (Qwen3-VL vision) ────────────

#[test]
fn layer_norm_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (3usize, 32usize);
    let eps = 1e-6f32;

    let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.05 - 0.7).collect();
    let weight: Vec<f32> = (0..cols).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..cols).map(|i| -0.1 + (i as f32) * 0.003).collect();

    // Reference computed in fp32 (as the kernel does internally).
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let mean = input[off..off + cols].iter().sum::<f32>() / cols as f32;
        let var = input[off..off + cols]
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / cols as f32;
        let inv = (var + eps).sqrt().recip();
        for c in 0..cols {
            expected[off + c] = weight[c] * (input[off + c] - mean) * inv + bias[c];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_w = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let t_b = upload_fp32_as_bf16(&ctx, &bias, vec![cols]).unwrap();
    let out = layer(&ctx, &t_in, &t_w, &t_b, eps).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn layer_norm_bf16_matches_qwen_merger_width() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (7usize, 4096usize);
    let eps = 1e-6f32;
    let input = (0..rows * cols)
        .map(|index| (index as f32 * 0.017 - 0.4).sin())
        .collect::<Vec<_>>();
    let weight = (0..cols)
        .map(|index| 0.8 + (index as f32 * 0.009).cos() * 0.15)
        .collect::<Vec<_>>();
    let bias = (0..cols)
        .map(|index| (index as f32 * 0.011).sin() * 0.1)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let offset = row * cols;
        let mean = input[offset..offset + cols].iter().sum::<f32>() / cols as f32;
        let variance = input[offset..offset + cols]
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / cols as f32;
        let inverse_std = (variance + eps).sqrt().recip();
        for column in 0..cols {
            expected[offset + column] =
                weight[column] * (input[offset + column] - mean) * inverse_std + bias[column];
        }
    }

    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![cols]).unwrap();
    let output = layer(&ctx, &input, &weight, &bias, eps).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    report_error_metrics("layer_norm_qwen_merger", &actual, &expected);
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn adaptive_layer_norm_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Released GR00T N1.7 DiT shape: one state token plus 40 action
    // tokens, with an action embedding width of 1536.
    let (rows, cols) = (41usize, 1536usize);
    let eps = 1e-5f32;
    let input = (0..rows * cols)
        .map(|index| (index as f32 * 0.019 - 0.7).sin())
        .collect::<Vec<_>>();
    let scale = (0..cols)
        .map(|index| (index as f32 * 0.013).cos() * 0.2)
        .collect::<Vec<_>>();
    let shift = (0..cols)
        .map(|index| (index as f32 * 0.021).sin() * 0.1)
        .collect::<Vec<_>>();
    let modulation = scale.iter().chain(&shift).copied().collect::<Vec<_>>();
    let mut expected = vec![0.0f32; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let values = &input[offset..offset + cols];
        let mean = values.iter().sum::<f32>() / cols as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / cols as f32;
        let inverse_std = (variance + eps).sqrt().recip();
        for column in 0..cols {
            expected[offset + column] =
                (values[column] - mean) * inverse_std * (1.0 + scale[column]) + shift[column];
        }
    }

    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let modulation = upload_fp32_as_bf16(&ctx, &modulation, vec![2 * cols]).unwrap();
    let output = adaptive_layer(&ctx, &input, &modulation, eps).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    report_error_metrics("adaptive_layer_norm_41x1536", &actual, &expected);
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn adaptive_layer_norm_rejects_invalid_modulation() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = upload_fp32_as_bf16(&ctx, &[0.0; 16], vec![2, 8]).unwrap();
    let wrong = upload_fp32_as_bf16(&ctx, &[0.0; 8], vec![8]).unwrap();
    assert!(adaptive_layer(&ctx, &input, &wrong, 1e-5).is_err());
    let right = upload_fp32_as_bf16(&ctx, &[0.0; 16], vec![16]).unwrap();
    assert!(adaptive_layer(&ctx, &input, &right, 0.0).is_err());
}

#[test]
fn gelu_tanh_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input: Vec<f32> = (0..65).map(|i| -4.0 + (i as f32) * 0.125).collect();

    let beta = (2.0f32 / std::f32::consts::PI).sqrt();
    let alpha = 0.044715f32;
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (beta * (x + alpha * x * x * x)).tanh()))
        .collect();

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![input.len()]).unwrap();
    let out = gelu_tanh(&ctx, &t_in).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn bias_gelu_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (3usize, 65usize);
    let input = (0..rows * columns)
        .map(|index| (index as f32 * 0.071).sin() * 3.0)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| index as f32 * 0.013 - 0.4)
        .collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();
    let rounded_input = download_bf16_as_fp32(&input).unwrap();
    let rounded_bias = download_bf16_as_fp32(&bias).unwrap();
    let beta = (2.0f32 / std::f32::consts::PI).sqrt();
    let alpha = 0.044715f32;
    let expected = rounded_input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = *value + rounded_bias[index % columns];
            0.5 * value * (1.0 + (beta * (value + alpha * value * value * value)).tanh())
        })
        .collect::<Vec<_>>();
    let output = bias_gelu_bf16(&ctx, &input, Some(&bias)).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn specialized_bias_activation_bf16_matches_established_kernels() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (17usize, 128usize);
    let input = (0..rows * columns)
        .map(|index| (index as f32 * 0.037).sin() * 7.0 - 0.125)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| (index as f32 * 0.071).cos() * 0.75)
        .collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();

    for activation in 1..=3 {
        unsafe { std::env::remove_var("APXINF_BIAS_ACTIVATION_SPECIALIZED") };
        let established = match activation {
            1 => bias_gelu_bf16(&ctx, &input, Some(&bias)),
            2 => bias_silu_bf16(&ctx, &input, Some(&bias)),
            3 => bias_relu_bf16(&ctx, &input, Some(&bias)),
            _ => unreachable!(),
        }
        .unwrap();
        unsafe { std::env::set_var("APXINF_BIAS_ACTIVATION_SPECIALIZED", "1") };
        let specialized = match activation {
            1 => bias_gelu_bf16(&ctx, &input, Some(&bias)),
            2 => bias_silu_bf16(&ctx, &input, Some(&bias)),
            3 => bias_relu_bf16(&ctx, &input, Some(&bias)),
            _ => unreachable!(),
        }
        .unwrap();
        unsafe { std::env::remove_var("APXINF_BIAS_ACTIVATION_SPECIALIZED") };

        assert_eq!(
            download_bf16_as_fp32(&specialized).unwrap(),
            download_bf16_as_fp32(&established).unwrap(),
            "specialized activation {activation} must be bit-identical"
        );
    }
}

#[test]
fn bias_residual_layer_bf16_matches_rounded_hidden_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (3usize, 64usize);
    let eps = 1.0e-6f32;
    let projection = (0..rows * columns)
        .map(|index| (index as f32 * 0.037).sin())
        .collect::<Vec<_>>();
    let residual = (0..rows * columns)
        .map(|index| (index as f32 * 0.019).cos() * 0.5)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| index as f32 * 0.002 - 0.05)
        .collect::<Vec<_>>();
    let weight = (0..columns)
        .map(|index| 0.9 + index as f32 * 0.001)
        .collect::<Vec<_>>();
    let norm_bias = (0..columns)
        .map(|index| index as f32 * -0.0005)
        .collect::<Vec<_>>();
    let projection = upload_fp32_as_bf16(&ctx, &projection, vec![rows, columns]).unwrap();
    let residual = upload_fp32_as_bf16(&ctx, &residual, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &weight, vec![columns]).unwrap();
    let norm_bias = upload_fp32_as_bf16(&ctx, &norm_bias, vec![columns]).unwrap();

    let fused = bias_residual_layer_bf16(
        &ctx,
        &projection,
        Some(&bias),
        &residual,
        &weight,
        &norm_bias,
        eps,
    )
    .unwrap();
    let hidden = bias_residual_bf16(&ctx, &projection, Some(&bias), &residual).unwrap();
    let expected_hidden = download_bf16_as_fp32(&hidden).unwrap();
    let actual_hidden = download_bf16_as_fp32(&fused.hidden).unwrap();
    assert_eq!(actual_hidden, expected_hidden);

    let weight = download_bf16_as_fp32(&weight).unwrap();
    let norm_bias = download_bf16_as_fp32(&norm_bias).unwrap();
    let mut expected_norm = Vec::with_capacity(rows * columns);
    for row in expected_hidden.chunks_exact(columns) {
        let mean = row.iter().sum::<f32>() / columns as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / columns as f32;
        let inverse_std = (variance + eps).sqrt().recip();
        expected_norm.extend(row.iter().enumerate().map(|(column, value)| {
            (*value - mean) * inverse_std * weight[column] + norm_bias[column]
        }));
    }
    let actual_norm = download_bf16_as_fp32(&fused.normalized).unwrap();
    report_error_metrics("bias_residual_layer_3x64", &actual_norm, &expected_norm);
    assert_bf16_close_reduction(&actual_norm, &expected_norm);
}

#[test]
fn bias_then_residual_bf16_matches_two_kernel_contract() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (7usize, 132usize);
    let projection = (0..rows * columns)
        .map(|index| (index as f32 * 0.037).sin() * 3.0)
        .collect::<Vec<_>>();
    let residual = (0..rows * columns)
        .map(|index| (index as f32 * 0.019).cos() * 2.0)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| index as f32 * 0.002 - 0.05)
        .collect::<Vec<_>>();
    let projection = upload_fp32_as_bf16(&ctx, &projection, vec![rows, columns]).unwrap();
    let residual = upload_fp32_as_bf16(&ctx, &residual, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();

    let biased = bias_bf16(&ctx, &projection, Some(&bias)).unwrap();
    let expected = add(&ctx, &biased, &residual).unwrap();
    let actual = bias_then_residual_bf16(&ctx, &projection, Some(&bias), &residual).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&actual).unwrap(),
        download_bf16_as_fp32(&expected).unwrap()
    );
}

#[test]
fn fused_qkv_bias_bf16_matches_three_independent_bias_kernels() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (41usize, 1536usize);
    let make_values = |factor: usize, modulus: usize| {
        (0..rows * columns)
            .map(|index| ((index * factor % modulus) as f32 - modulus as f32 / 2.0) / 64.0)
            .collect::<Vec<_>>()
    };
    let make_bias = |factor: usize| {
        (0..columns)
            .map(|index| ((index * factor % 127) as f32 - 63.0) / 128.0)
            .collect::<Vec<_>>()
    };
    let query = upload_fp32_as_bf16(&ctx, &make_values(17, 257), vec![rows, columns]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &make_values(29, 251), vec![rows, columns]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &make_values(43, 241), vec![rows, columns]).unwrap();
    let query_bias = upload_fp32_as_bf16(&ctx, &make_bias(11), vec![columns]).unwrap();
    let key_bias = upload_fp32_as_bf16(&ctx, &make_bias(19), vec![columns]).unwrap();
    let value_bias = upload_fp32_as_bf16(&ctx, &make_bias(31), vec![columns]).unwrap();

    let expected_query = bias_bf16(&ctx, &query, Some(&query_bias)).unwrap();
    let expected_key = bias_bf16(&ctx, &key, Some(&key_bias)).unwrap();
    let expected_value = bias_bf16(&ctx, &value, Some(&value_bias)).unwrap();
    let (actual_query, actual_key, actual_value) =
        bias_qkv_in_place_bf16(&ctx, query, key, value, &query_bias, &key_bias, &value_bias)
            .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&actual_query).unwrap(),
        download_bf16_as_fp32(&expected_query).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual_key).unwrap(),
        download_bf16_as_fp32(&expected_key).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual_value).unwrap(),
        download_bf16_as_fp32(&expected_value).unwrap()
    );
}

#[test]
fn bias_relu_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (3usize, 12usize);
    let input = (0..rows * columns)
        .map(|index| index as f32 * 0.1 - 1.5)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| index as f32 * -0.03 + 0.2)
        .collect::<Vec<_>>();
    let expected = input
        .iter()
        .enumerate()
        .map(|(index, value)| (value + bias[index % columns]).max(0.0))
        .collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();
    let output = bias_relu_bf16(&ctx, &input, Some(&bias)).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&output).unwrap(), &expected);

    let without_bias = bias_relu_bf16(&ctx, &input, None).unwrap();
    let expected_without_bias = download_bf16_as_fp32(&input)
        .unwrap()
        .into_iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    assert_bf16_close_elementwise(
        &download_bf16_as_fp32(&without_bias).unwrap(),
        &expected_without_bias,
    );
}

#[test]
fn bias_silu_bf16_matches_rounded_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (3usize, 12usize);
    let input = (0..rows * columns)
        .map(|index| (index as f32 * 0.17).sin() * 3.0)
        .collect::<Vec<_>>();
    let bias = (0..columns)
        .map(|index| index as f32 * 0.021 - 0.1)
        .collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, columns]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![columns]).unwrap();
    let rounded_input = download_bf16_as_fp32(&input).unwrap();
    let rounded_bias = download_bf16_as_fp32(&bias).unwrap();
    let expected = rounded_input
        .iter()
        .enumerate()
        .map(|(index, value)| silu_ref(*value + rounded_bias[index % columns]))
        .collect::<Vec<_>>();
    let output = bias_silu_bf16(&ctx, &input, Some(&bias)).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn bias_bf16_identity_matches_rounded_reference_without_bias() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (3usize, 12usize);
    let input = (0..rows * columns)
        .map(|index| index as f32 * 0.11 - 1.75)
        .collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![rows, columns]).unwrap();
    let expected = download_bf16_as_fp32(&input).unwrap();
    let output = bias_bf16(&ctx, &input, None).unwrap();
    assert_eq!(download_bf16_as_fp32(&output).unwrap(), expected);
}

#[test]
fn add_bias_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (5usize, 16usize);
    let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let bias: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            expected[r * cols + c] = input[r * cols + c] + bias[c];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_b = upload_fp32_as_bf16(&ctx, &bias, vec![cols]).unwrap();
    let out = add_bias(&ctx, &t_in, &t_b).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn contiguous_rows_returns_the_requested_cuda_view() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = (0..20).map(|value| value as f32).collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![5, 4]).unwrap();
    let output = contiguous_rows(&ctx, &input, 2, 2).unwrap();
    assert_eq!(output.shape().dims(), &[2, 4]);
    assert_eq!(
        download_bf16_as_fp32(&output).unwrap(),
        vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
    assert!(contiguous_rows(&ctx, &input, 4, 2).is_err());
}

#[test]
fn gather_rows_bf16_matches_host_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let input = upload_fp32_as_bf16(&ctx, &input, vec![6, 4]).unwrap();
    let output = gather_rows_bf16(&ctx, &input, &[4, 1, 5]).unwrap();

    assert_eq!(output.shape().dims(), &[3, 4]);
    assert_eq!(
        download_bf16_as_fp32(&output).unwrap(),
        vec![16.0, 17.0, 18.0, 19.0, 4.0, 5.0, 6.0, 7.0, 20.0, 21.0, 22.0, 23.0]
    );
    assert!(gather_rows_bf16(&ctx, &input, &[6]).is_err());

    let indices = prepare_row_indices(&ctx, &[0, 2], 3).unwrap();
    assert!(gather_rows_bf16_prepared(&ctx, &input, &indices).is_err());
}

#[test]
fn scatter_rows_bf16_overwrites_and_adds_on_device() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let destination = upload_fp32_as_bf16(&ctx, &[1.0; 20], vec![5, 4]).unwrap();
    let source =
        upload_fp32_as_bf16(&ctx, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![2, 4]).unwrap();

    let overwritten = scatter_rows_bf16(&ctx, &destination, &[3, 1], &source, false).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&overwritten).unwrap(),
        vec![
            1.0, 1.0, 1.0, 1.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0,
            1.0, 1.0, 1.0,
        ]
    );

    let added = scatter_rows_bf16(&ctx, &destination, &[3, 1], &source, true).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&added).unwrap(),
        vec![
            1.0, 1.0, 1.0, 1.0, 7.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 1.0,
            1.0, 1.0, 1.0,
        ]
    );
    assert!(scatter_rows_bf16(&ctx, &destination, &[1, 1], &source, true).is_err());

    let duplicate_indices = prepare_row_indices(&ctx, &[1, 1], 5).unwrap();
    assert!(
        scatter_rows_bf16_prepared(&ctx, &destination, &duplicate_indices, &source, true,).is_err()
    );
}

// ── Vision 2D-RoPE ───────────────────────────────────────────────

#[test]
fn rope_vision_2d_bf16_matches_reference() {
    // HF vision RoPE: head_dim=64, 16 freq pairs per axis (h then w).
    // pair p < 16 uses h coord; pair p >= 16 uses w coord.
    // inv_freq[i] = 1/theta^(2i/32) for i in [0,16).  rotate_half.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 4usize, 64usize);
    let theta = 10000.0f32;
    let pos_ids: Vec<[u32; 2]> = vec![[3u32, 7], [5, 11]];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();

    // Reference.
    let half = head_dim / 2; // 32
    let mut expected = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for p in 0..half {
                let axis = if p < half / 2 { 0 } else { 1 };
                let pair_in_axis = if p < half / 2 { p } else { p - half / 2 };
                let pos = pos_ids[s][axis];
                let freq = 1.0f32 / theta.powf(2.0 * pair_in_axis as f32 / half as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + p];
                let x1 = input[base + half + p];
                expected[base + p] = x0 * c - x1 * sn;
                expected[base + half + p] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let out = apply_vision_2d(&ctx, &t_in, n_heads, head_dim, theta, &pos_buf).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn rope_vision_2d_pair_matches_individual_launches() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (257usize, 2usize, 64usize);
    let theta = 10000.0f32;
    let q: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    let k: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (i as f32 * 0.017).cos())
        .collect();
    let pos_ids: Vec<[u32; 2]> = (0..seq_len)
        .map(|i| [((i / 16) % 16) as u32, (i % 16) as u32])
        .collect();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|position| position.iter().flat_map(|value| value.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let q = upload_fp32_as_bf16(&ctx, &q, vec![seq_len, n_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![seq_len, n_heads, head_dim]).unwrap();
    let q_expected = apply_vision_2d(&ctx, &q, n_heads, head_dim, theta, &pos_buf).unwrap();
    let k_expected = apply_vision_2d(&ctx, &k, n_heads, head_dim, theta, &pos_buf).unwrap();
    let (q_actual, k_actual) =
        apply_vision_2d_pair(&ctx, &q, &k, n_heads, head_dim, theta, &pos_buf).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&q_actual).unwrap(),
        download_bf16_as_fp32(&q_expected).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&k_actual).unwrap(),
        download_bf16_as_fp32(&k_expected).unwrap()
    );
}

#[test]
fn fused_vision_qkv_split_bias_rope_matches_decomposed_path() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (257usize, 4usize, 64usize);
    let width = n_heads * head_dim;
    let theta = 10000.0f32;
    let qkv: Vec<f32> = (0..seq_len * 3 * width)
        .map(|i| (i as f32 * 0.0091 - 0.7).sin())
        .collect();
    let bias: Vec<f32> = (0..3 * width)
        .map(|i| (i as f32 * 0.017 - 0.3).cos() * 0.1)
        .collect();
    let positions: Vec<[u32; 2]> = (0..seq_len)
        .map(|i| [((i / 16) % 16) as u32, (i % 16) as u32])
        .collect();
    let position_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|position| position.iter().flat_map(|value| value.to_ne_bytes()))
        .collect();
    let position_buffer = CudaBuffer::alloc(position_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    position_buffer
        .copy_from_host(&position_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let qkv = upload_fp32_as_bf16(&ctx, &qkv, vec![seq_len, 3 * width]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![3 * width]).unwrap();
    let split = split_qkv_bias_bf16(&ctx, &qkv, Some(&bias), n_heads, head_dim).unwrap();
    let (q_expected, k_expected) = apply_vision_2d_pair(
        &ctx,
        &split.q,
        &split.k,
        n_heads,
        head_dim,
        theta,
        &position_buffer,
    )
    .unwrap();
    let actual = split_qkv_bias_apply_vision_2d(
        &ctx,
        &qkv,
        &bias,
        n_heads,
        head_dim,
        theta,
        &position_buffer,
    )
    .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&actual.q).unwrap(),
        download_bf16_as_fp32(&q_expected).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual.k).unwrap(),
        download_bf16_as_fp32(&k_expected).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual.v).unwrap(),
        download_bf16_as_fp32(&split.v).unwrap()
    );
}

#[test]
fn fused_vision_qkv_precomputed_rope_matches_established_kernel() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (257usize, 16usize, 64usize);
    let width = n_heads * head_dim;
    let theta = 10000.0f32;
    let qkv: Vec<f32> = (0..seq_len * 3 * width)
        .map(|i| (i as f32 * 0.0073 - 0.4).sin())
        .collect();
    let bias: Vec<f32> = (0..3 * width)
        .map(|i| (i as f32 * 0.013 + 0.2).cos() * 0.125)
        .collect();
    let positions: Vec<[u32; 2]> = (0..seq_len)
        .map(|i| [((i / 16) % 16) as u32, (i % 16) as u32])
        .collect();
    let position_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|position| position.iter().flat_map(|value| value.to_ne_bytes()))
        .collect();
    let position_buffer = CudaBuffer::alloc(position_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    position_buffer
        .copy_from_host(&position_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let qkv = upload_fp32_as_bf16(&ctx, &qkv, vec![seq_len, 3 * width]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![3 * width]).unwrap();
    let expected = split_qkv_bias_apply_vision_2d(
        &ctx,
        &qkv,
        &bias,
        n_heads,
        head_dim,
        theta,
        &position_buffer,
    )
    .unwrap();
    let table =
        prepare_vision_rope_cos_sin(&ctx, seq_len, head_dim, theta, &position_buffer).unwrap();
    let actual = split_qkv_bias_apply_vision_2d_precomputed(
        &ctx,
        &qkv,
        &bias,
        n_heads,
        head_dim,
        &position_buffer,
        &table,
    )
    .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&actual.q).unwrap(),
        download_bf16_as_fp32(&expected.q).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual.k).unwrap(),
        download_bf16_as_fp32(&expected.k).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&actual.v).unwrap(),
        download_bf16_as_fp32(&expected.v).unwrap()
    );
}

// ── Vision SDPA (non-causal full attention) ──────────────────────

#[test]
fn vision_sdpa_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // 257 deliberately crosses multiple 128-row FA2 tiles and exercises the
    // non-multiple padding mask used by the released 256-patch vision path.
    let (seq, n_heads, head_dim) = (257usize, 2usize, 64usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.01 - 0.3).sin())
        .collect();
    let k: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.007).tanh())
        .collect();

    // Reference: non-causal, per head.
    let mut expected = vec![0.0f32; seq * n_heads * head_dim];
    for h in 0..n_heads {
        for qi in 0..seq {
            // scores[ki] = (Q[qi,h] · K[ki,h]) * scale
            let mut scores = vec![0.0f32; seq];
            let mut mx = f32::NEG_INFINITY;
            for ki in 0..seq {
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += q[qi * n_heads * head_dim + h * head_dim + d]
                        * k[ki * n_heads * head_dim + h * head_dim + d];
                }
                s *= scale;
                scores[ki] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0;
            for ki in 0..seq {
                scores[ki] = (scores[ki] - mx).exp();
                sum += scores[ki];
            }
            for ki in 0..seq {
                scores[ki] /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0;
                for ki in 0..seq {
                    acc += scores[ki] * v[ki * n_heads * head_dim + h * head_dim + d];
                }
                expected[qi * n_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }

    let t_q = upload_fp32_as_bf16(&ctx, &q, vec![seq, n_heads, head_dim]).unwrap();
    let t_k = upload_fp32_as_bf16(&ctx, &k, vec![seq, n_heads, head_dim]).unwrap();
    let t_v = upload_fp32_as_bf16(&ctx, &v, vec![seq, n_heads, head_dim]).unwrap();
    let out = vision(&ctx, &t_q, &t_k, &t_v, seq, n_heads, head_dim).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn causal_gqa_prefill_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (sequence_len, query_heads, kv_heads, head_dim) = (17usize, 4usize, 2usize, 128usize);
    let group_size = query_heads / kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q = (0..sequence_len * query_heads * head_dim)
        .map(|index| (index as f32 * 0.007 - 0.4).sin())
        .collect::<Vec<_>>();
    let k = (0..sequence_len * kv_heads * head_dim)
        .map(|index| (index as f32 * 0.011 + 0.2).cos())
        .collect::<Vec<_>>();
    let v = (0..sequence_len * kv_heads * head_dim)
        .map(|index| (index as f32 * 0.009 - 0.1).tanh())
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; sequence_len * query_heads * head_dim];
    for query in 0..sequence_len {
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let mut scores = vec![0.0f32; query + 1];
            for key in 0..=query {
                scores[key] = (0..head_dim)
                    .map(|dimension| {
                        q[(query * query_heads + query_head) * head_dim + dimension]
                            * k[(key * kv_heads + kv_head) * head_dim + dimension]
                    })
                    .sum::<f32>()
                    * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            for score in &mut scores {
                *score /= denominator;
            }
            for dimension in 0..head_dim {
                expected[(query * query_heads + query_head) * head_dim + dimension] = (0..=query)
                    .map(|key| scores[key] * v[(key * kv_heads + kv_head) * head_dim + dimension])
                    .sum();
            }
        }
    }

    let q = upload_fp32_as_bf16(&ctx, &q, vec![sequence_len, query_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![sequence_len, kv_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v, vec![sequence_len, kv_heads, head_dim]).unwrap();
    let Some(output) = causal_gqa_prefill_bf16(&ctx, &q, &k, &v).unwrap() else {
        eprintln!("causal GQA prefill is not compiled for this CUDA target");
        return;
    };
    assert_eq!(
        output.shape().dims(),
        &[sequence_len, query_heads, head_dim]
    );
    let actual = download_bf16_as_fp32(&output).unwrap();
    report_error_metrics("causal_gqa_s17_qh4_kvh2_d128", &actual, &expected);
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn noncausal_cross_sdpa_bf16_matches_reference_for_gr00t_head_dim() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Use the released DiT query/head geometry. The shorter deterministic
    // source sequence keeps this operator test quick while still exercising
    // every one of GR00T's 32 heads and the 48-wide partial warp path.
    let (query_len, key_value_len, n_heads, head_dim) = (41usize, 19usize, 32usize, 48usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q = (0..query_len * n_heads * head_dim)
        .map(|index| (index as f32 * 0.017 - 0.4).sin())
        .collect::<Vec<_>>();
    let k = (0..key_value_len * n_heads * head_dim)
        .map(|index| (index as f32 * 0.011 + 0.2).cos())
        .collect::<Vec<_>>();
    let v = (0..key_value_len * n_heads * head_dim)
        .map(|index| (index as f32 * 0.009 - 0.1).tanh())
        .collect::<Vec<_>>();

    let mut expected = vec![0.0f32; query_len * n_heads * head_dim];
    for query in 0..query_len {
        for head in 0..n_heads {
            let mut scores = vec![0.0f32; key_value_len];
            for key in 0..key_value_len {
                scores[key] = (0..head_dim)
                    .map(|dimension| {
                        q[(query * n_heads + head) * head_dim + dimension]
                            * k[(key * n_heads + head) * head_dim + dimension]
                    })
                    .sum::<f32>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - max).exp();
                    *score
                })
                .sum::<f32>();
            for score in &mut scores {
                *score /= sum;
            }
            for dimension in 0..head_dim {
                expected[(query * n_heads + head) * head_dim + dimension] = (0..key_value_len)
                    .map(|key| scores[key] * v[(key * n_heads + head) * head_dim + dimension])
                    .sum();
            }
        }
    }

    let q = upload_fp32_as_bf16(&ctx, &q, vec![query_len, n_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![key_value_len, n_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v, vec![key_value_len, n_heads, head_dim]).unwrap();
    let output = noncausal(&ctx, &q, &k, &v, n_heads, head_dim).unwrap();
    assert_eq!(output.shape().dims(), &[query_len, n_heads * head_dim]);
    let actual = download_bf16_as_fp32(&output).unwrap();
    report_error_metrics("noncausal_sdpa_q41_kv19_h32_d48", &actual, &expected);
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn noncausal_sdpa_rejects_invalid_shapes() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let q = upload_fp32_as_bf16(&ctx, &[0.0; 2 * 48], vec![1, 2, 48]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &[0.0; 3 * 2 * 48], vec![3, 2, 48]).unwrap();
    let bad_v = upload_fp32_as_bf16(&ctx, &[0.0; 2 * 2 * 48], vec![2, 2, 48]).unwrap();
    assert!(noncausal(&ctx, &q, &k, &bad_v, 2, 48).is_err());
    assert!(noncausal(&ctx, &q, &k, &k, 2, 47).is_err());
}

// ── concat_2d (fused weight packing) ─────────────────────────────

#[test]
fn concat_2d_bf16_packs_qkv_correctly() {
    // Simulates the fused-QKV weight packing: concat(wq, wk, wv)
    // along the output axis. wq=[hidden,hidden], wk=wv=[hidden,kv_proj].
    use crate::backend::CudaBackend;
    use apxinf_core::Backend;

    let be = CudaBackend::new(0).expect("CUDA device required");
    let hidden = 64;
    let kv_proj = 32;
    let rows = hidden;

    let wq: Vec<f32> = (0..rows * hidden).map(|i| (i as f32) * 0.01).collect();
    let wk: Vec<f32> = (0..rows * kv_proj)
        .map(|i| (i as f32) * 0.02 - 1.0)
        .collect();
    let wv: Vec<f32> = (0..rows * kv_proj)
        .map(|i| (i as f32) * 0.03 + 0.5)
        .collect();

    let t_wq = upload_fp32_as_bf16(be.context(), &wq, vec![rows, hidden]).unwrap();
    let t_wk = upload_fp32_as_bf16(be.context(), &wk, vec![rows, kv_proj]).unwrap();
    let t_wv = upload_fp32_as_bf16(be.context(), &wv, vec![rows, kv_proj]).unwrap();

    let packed = be.concat_2d(&[&t_wq, &t_wk, &t_wv]).expect("concat_2d");
    let out = download_bf16_as_fp32(&packed).unwrap();
    let total_cols = hidden + 2 * kv_proj;
    assert_eq!(packed.shape().dims(), &[rows, total_cols]);

    // Build expected = wq | wk | wv concatenated row-by-row.
    let mut expected = vec![0.0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..hidden {
            expected[r * total_cols + c] = wq[r * hidden + c];
        }
        for c in 0..kv_proj {
            expected[r * total_cols + hidden + c] = wk[r * kv_proj + c];
        }
        for c in 0..kv_proj {
            expected[r * total_cols + hidden + kv_proj + c] = wv[r * kv_proj + c];
        }
    }
    assert_bf16_close_elementwise(&out, &expected);
}

#[test]
fn concat_2d_bf16_packs_gate_up_correctly() {
    // Simulates the fused Gate/Up weight packing.
    use crate::backend::CudaBackend;
    use apxinf_core::Backend;

    let be = CudaBackend::new(0).expect("CUDA device required");
    let hidden = 64;
    let inter = 128;
    let rows = hidden;

    let w_gate: Vec<f32> = (0..rows * inter).map(|i| (i as f32) * 0.01).collect();
    let w_up: Vec<f32> = (0..rows * inter).map(|i| (i as f32) * 0.02 - 0.5).collect();

    let t_gate = upload_fp32_as_bf16(be.context(), &w_gate, vec![rows, inter]).unwrap();
    let t_up = upload_fp32_as_bf16(be.context(), &w_up, vec![rows, inter]).unwrap();

    let packed = be.concat_2d(&[&t_gate, &t_up]).expect("concat_2d");
    let out = download_bf16_as_fp32(&packed).unwrap();
    let total_cols = 2 * inter;
    assert_eq!(packed.shape().dims(), &[rows, total_cols]);

    let mut expected = vec![0.0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..inter {
            expected[r * total_cols + c] = w_gate[r * inter + c];
        }
        for c in 0..inter {
            expected[r * total_cols + inter + c] = w_up[r * inter + c];
        }
    }
    assert_bf16_close_elementwise(&out, &expected);
}
