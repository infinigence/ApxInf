use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::backend::vision_group_plan;
use crate::buffer::{CudaBuffer, HostMappedBuffer};
use crate::context::CudaContext;
use crate::kernels::activation::{gelu_tanh, silu, silu_mul, silu_mul_packed_rows_exact};
use crate::kernels::attention::{
    causal_mask, grouped, grouped_indexed, sdpa_with_batched_prefill, softmax, softmax_causal,
    softmax_causal_bf16_scaled_exp_cache, softmax_causal_bf16_scaled_in_place_gqa_packed,
    softmax_causal_bf16_scaled_plain, softmax_causal_bf16_scaled_plain_gqa_packed,
    softmax_causal_with_exp_cache, softmax_causal_with_global_exp_cache,
    split_gqa_qkv_bias_bf16, vision,
};
#[cfg(apxinf_fa2_causal_sm80)]
use crate::kernels::attention::causal_fa2_gqa_prefill;
#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
use crate::kernels::attention::grouped_varlen_fa2;
use crate::kernels::cache::append;
use crate::kernels::elementwise::{add, add_bias, mul, scale};
use crate::kernels::embedding::lookup;
use crate::kernels::norm::{layer, residual_add_rms_exact_bf16_into, rms};
use crate::kernels::preprocess::{avg_pool1d_bf16, im2col1d_bf16};
use crate::kernels::qwen25_omni_attention::{
    grouped2_split_cta_write, grouped4_split_cta_write, packed_qkv_prelude_write,
    prefill_packed_qkv_prelude_write, short_w32_write, SplitCtaWorkspace,
};
use crate::kernels::qwen25_omni_fused::residual_add_rmsnorm_pack8_bf16_into;
use crate::kernels::qwen25_omni_vision::{
    bias_residual_exact as qwen25_vision_bias_residual_exact,
    gate_up_bias_silu_mul_exact as qwen25_vision_gate_up_bias_silu_mul_exact,
    grouped_qkv_bias_rope as qwen25_vision_grouped_qkv_bias_rope,
    packed_gate_up_bias_silu_mul_exact as qwen25_vision_packed_gate_up_bias_silu_mul_exact,
    packed_qkv_bias_rope as qwen25_vision_packed_qkv_bias_rope,
    qkv_bias_rope as qwen25_vision_qkv_bias_rope,
};
use crate::kernels::rope::{apply, apply_batched, apply_mrope, apply_tmrope, apply_vision_2d};
use crate::kernels::selection::argmax_bf16_into;
use crate::CudaKVCache;

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

#[test]
fn argmax_bf16_matches_lowest_index_cpu_contract() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let run = |values: &[f32]| {
        let tensor = upload_fp32_as_bf16(&ctx, values, vec![values.len()]).unwrap();
        let logits = CudaBuffer::from_tensor(&tensor).unwrap();
        let partials = CudaBuffer::alloc_zeros(
            crate::kernels::selection::ARGMAX_PARTIAL_BYTES,
            ctx.device_id(),
        )
        .unwrap();
        let output = HostMappedBuffer::alloc(4, ctx.device_id()).unwrap();
        argmax_bf16_into(
            &ctx,
            &logits,
            &partials,
            output.address(),
            values.len(),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        output.read_u32().unwrap()
    };

    assert_eq!(run(&[1.0, 5.0, 5.0, 4.0]), 1);
    assert_eq!(run(&[-0.0, 0.0]), 0);
    assert_eq!(run(&[f32::NAN, 3.0, 3.0]), 1);
    assert_eq!(run(&[f32::NAN, f32::NAN]), 0);

    let mut full_vocab = vec![-4.0; 151_936];
    full_vocab[17] = 8.0;
    full_vocab[150_000] = 8.0;
    assert_eq!(run(&full_vocab), 17);
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
fn silu_mul_separate_bf16_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (4usize, 257usize);
    let gate_values = (0..rows * cols)
        .map(|index| ((index as f32 * 0.017) - 7.0).sin() * 6.0)
        .collect::<Vec<_>>();
    let up_values = (0..rows * cols)
        .map(|index| ((index as f32 * 0.023) - 3.0).cos() * 4.0)
        .collect::<Vec<_>>();
    let gate = upload_fp32_as_bf16(&ctx, &gate_values, vec![rows, cols]).unwrap();
    let up = upload_fp32_as_bf16(&ctx, &up_values, vec![rows, cols]).unwrap();
    let activated = silu(&ctx, &gate).unwrap();
    let separate = mul(&ctx, &activated, &up).unwrap();
    let fused = silu_mul(&ctx, &gate, &up).unwrap();
    let packed_values = gate_values
        .chunks_exact(cols)
        .zip(up_values.chunks_exact(cols))
        .flat_map(|(gate, up)| gate.iter().chain(up).copied())
        .collect::<Vec<_>>();
    let packed = upload_fp32_as_bf16(&ctx, &packed_values, vec![rows, 2 * cols]).unwrap();
    let packed_fused = silu_mul_packed_rows_exact(&ctx, &packed, cols).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&fused).unwrap(),
        download_bf16_as_fp32(&separate).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&packed_fused).unwrap(),
        download_bf16_as_fp32(&separate).unwrap()
    );
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

#[test]
fn residual_add_rms_exact_bf16_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (2usize, 2048usize);
    let residual = (0..rows * cols)
        .map(|index| ((index * 17 % 257) as f32 - 128.0) / 31.0)
        .collect::<Vec<_>>();
    let delta = (0..rows * cols)
        .map(|index| ((index * 29 % 193) as f32 - 96.0) / 47.0)
        .collect::<Vec<_>>();
    let weight = (0..cols)
        .map(|index| 0.75 + (index * 11 % 127) as f32 / 256.0)
        .collect::<Vec<_>>();
    let eps = 1e-6f32;

    let baseline_residual = add(
        &ctx,
        &upload_fp32_as_bf16(&ctx, &residual, vec![rows, cols]).unwrap(),
        &upload_fp32_as_bf16(&ctx, &delta, vec![rows, cols]).unwrap(),
    )
    .unwrap();
    let weight_tensor = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let baseline_norm = rms(&ctx, &baseline_residual, &weight_tensor, eps).unwrap();

    let candidate_residual = upload_fp32_as_bf16(&ctx, &residual, vec![rows, cols]).unwrap();
    let candidate_residual_buffer = CudaBuffer::from_tensor(&candidate_residual).unwrap();
    let delta_tensor = upload_fp32_as_bf16(&ctx, &delta, vec![rows, cols]).unwrap();
    let delta_buffer = CudaBuffer::from_tensor(&delta_tensor).unwrap();
    let weight_buffer = CudaBuffer::from_tensor(&weight_tensor).unwrap();
    let output_buffer =
        CudaBuffer::alloc_zeros(rows * cols * DType::BF16.size_in_bytes(), ctx.device_id())
            .unwrap();
    residual_add_rms_exact_bf16_into(
        &ctx,
        &candidate_residual_buffer,
        &delta_buffer,
        &weight_buffer,
        &output_buffer,
        cols,
        rows,
        eps,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let candidate_norm = make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output_buffer,
    );

    assert_eq!(
        download_bf16_as_fp32(&candidate_residual).unwrap(),
        download_bf16_as_fp32(&baseline_residual).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&candidate_norm).unwrap(),
        download_bf16_as_fp32(&baseline_norm).unwrap()
    );
}

#[test]
fn qwen25_pack8_residual_rmsnorm_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let columns = 2048usize;
    let residual = (0..columns)
        .map(|index| ((index * 17 % 257) as f32 - 128.0) / 31.0)
        .collect::<Vec<_>>();
    let delta = (0..columns)
        .map(|index| ((index * 29 % 193) as f32 - 96.0) / 47.0)
        .collect::<Vec<_>>();
    let weight = (0..columns)
        .map(|index| 0.75 + (index * 11 % 127) as f32 / 256.0)
        .collect::<Vec<_>>();
    let eps = 1.0e-6f32;
    let baseline_residual = upload_fp32_as_bf16(&ctx, &residual, vec![1, columns]).unwrap();
    let candidate_residual = upload_fp32_as_bf16(&ctx, &residual, vec![1, columns]).unwrap();
    let delta = upload_fp32_as_bf16(&ctx, &delta, vec![1, columns]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &weight, vec![columns]).unwrap();
    let baseline_residual_buffer = CudaBuffer::from_tensor(&baseline_residual).unwrap();
    let candidate_residual_buffer = CudaBuffer::from_tensor(&candidate_residual).unwrap();
    let delta_buffer = CudaBuffer::from_tensor(&delta).unwrap();
    let weight_buffer = CudaBuffer::from_tensor(&weight).unwrap();
    let output_bytes = columns * DType::BF16.size_in_bytes();
    let baseline_output = CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).unwrap();
    let candidate_output = CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).unwrap();
    residual_add_rms_exact_bf16_into(
        &ctx,
        &baseline_residual_buffer,
        &delta_buffer,
        &weight_buffer,
        &baseline_output,
        columns,
        1,
        eps,
    )
    .unwrap();
    residual_add_rmsnorm_pack8_bf16_into(
        &ctx,
        &candidate_residual_buffer,
        &delta_buffer,
        &weight_buffer,
        &candidate_output,
        columns,
        1,
        eps,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let baseline_output = baseline_output.into_tensor(Shape::new(vec![1, columns]), DType::BF16);
    let candidate_output = candidate_output.into_tensor(Shape::new(vec![1, columns]), DType::BF16);
    assert_eq!(
        download_bf16_as_fp32(&candidate_residual).unwrap(),
        download_bf16_as_fp32(&baseline_residual).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&candidate_output).unwrap(),
        download_bf16_as_fp32(&baseline_output).unwrap()
    );
    assert!(residual_add_rmsnorm_pack8_bf16_into(
        &ctx,
        &candidate_residual_buffer,
        &delta_buffer,
        &weight_buffer,
        &CudaBuffer::alloc_zeros(output_bytes, ctx.device_id()).unwrap(),
        columns,
        1,
        1.0e-5,
    )
    .unwrap_err()
    .to_string()
    .contains("pack8 residual RMSNorm contract mismatch"));
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

#[test]
fn attention_softmax_exp_cache_bf16_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, cols) = (3usize, 2usize, 257usize);
    let rows = seq_len * n_heads;
    let input = (0..rows * cols)
        .map(|index| ((index as f32 * 0.019) - 4.0).sin())
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let scalar = softmax_causal_with_exp_cache(&ctx, &tensor, 4, n_heads as u32, false)
        .unwrap();
    let cached = softmax_causal_with_exp_cache(&ctx, &tensor, 4, n_heads as u32, true)
        .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&cached).unwrap(),
        download_bf16_as_fp32(&scalar).unwrap()
    );
}

#[test]
fn attention_softmax_parallel_max_is_bit_exact_at_prefill_cache_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, cols) = (3usize, 2usize, 4_096usize);
    let rows = seq_len * n_heads;
    let input = (0..rows * cols)
        .map(|index| {
            let bits = (index as u32)
                .wrapping_mul(22_695_477)
                .wrapping_add(1);
            (bits & 0xffff) as f32 / 4_096.0 - 8.0
        })
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let parallel = softmax_causal_with_exp_cache(
        &ctx,
        &tensor,
        (cols - seq_len) as u32,
        n_heads as u32,
        false,
    )
    .unwrap();
    let sequential = softmax_causal_with_exp_cache(
        &ctx,
        &tensor,
        (cols - seq_len) as u32,
        n_heads as u32,
        true,
    )
    .unwrap();
    let parallel = download_bf16_as_fp32(&parallel).unwrap();
    let sequential = download_bf16_as_fp32(&sequential).unwrap();
    assert_eq!(parallel.len(), sequential.len());
    if let Some((index, (parallel, sequential))) = parallel
        .iter()
        .zip(&sequential)
        .enumerate()
        .find(|(_, (parallel, sequential))| parallel != sequential)
    {
        panic!(
            "parallel max differed at index {index}: \
             parallel={parallel:?}, sequential={sequential:?}"
        );
    }
}

#[test]
fn attention_softmax_parallel_max_is_bit_exact_at_decode_cache_limit() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, cols) = (2usize, 11_264usize);
    let input = (0..n_heads * cols)
        .map(|index| {
            let bits = (index as u32)
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            (bits & 0xffff) as f32 / 4_096.0 - 8.0
        })
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, cols]).unwrap();
    let scalar = softmax_causal_with_exp_cache(
        &ctx,
        &tensor,
        (cols - 1) as u32,
        n_heads as u32,
        false,
    )
    .unwrap();
    let cached = softmax_causal_with_exp_cache(
        &ctx,
        &tensor,
        (cols - 1) as u32,
        n_heads as u32,
        true,
    )
    .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&cached).unwrap(),
        download_bf16_as_fp32(&scalar).unwrap()
    );
}

#[test]
fn attention_softmax_fused_scale_is_bit_exact_across_long_prefill() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (3usize, 2usize, 128usize);
    let rows = seq_len * n_heads;
    let score_scale = 1.0 / (head_dim as f32).sqrt();
    for cols in [4_097usize, 8_192, 12_288] {
        let input = (0..rows * cols)
            .map(|index| {
                let bits = (index as u32)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345);
                (bits & 0xffff) as f32 / 8_192.0 - 4.0
            })
            .collect::<Vec<_>>();
        let tensor = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
        let scaled = scale(&ctx, &tensor, score_scale).unwrap();
        let separate = softmax_causal_with_exp_cache(
            &ctx,
            &scaled,
            (cols - seq_len) as u32,
            n_heads as u32,
            false,
        )
        .unwrap();
        let fused = softmax_causal_bf16_scaled_plain(
            &ctx,
            &tensor,
            (cols - seq_len) as u32,
            n_heads as u32,
            score_scale,
        )
        .unwrap();
        let separate = download_bf16_as_fp32(&separate).unwrap();
        let fused = download_bf16_as_fp32(&fused).unwrap();
        assert_eq!(separate.len(), fused.len());
        if let Some((index, (separate, fused))) = separate
            .iter()
            .zip(&fused)
            .enumerate()
            .find(|(_, (separate, fused))| separate != fused)
        {
            panic!(
                "fused scale differed at {cols} columns, index {index}: \
                 separate={separate:?}, fused={fused:?}"
            );
        }
    }
}

#[test]
fn attention_softmax_scaled_exp_cache_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n_heads = 16usize;
    let score_scale = 1.0 / 128.0f32.sqrt();
    for seq_len in [2usize, 32, 128] {
        let cols = seq_len;
        let rows = seq_len * n_heads;
        let input = (0..rows * cols)
            .map(|index| {
                let bits = (index as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                (bits & 0xffff) as f32 / 8_192.0 - 4.0
            })
            .collect::<Vec<_>>();
        let tensor = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
        let scaled = scale(&ctx, &tensor, score_scale).unwrap();
        let baseline = softmax_causal_with_exp_cache(
            &ctx,
            &scaled,
            0,
            n_heads as u32,
            true,
        )
        .unwrap();
        let candidate = softmax_causal_bf16_scaled_exp_cache(
            &ctx,
            &tensor,
            0,
            n_heads as u32,
            score_scale,
        )
        .unwrap();
        assert_eq!(
            download_bf16_as_fp32(&candidate).unwrap(),
            download_bf16_as_fp32(&baseline).unwrap()
        );
    }
}

#[test]
fn attention_softmax_packed_gqa_rows_match_standard_layout() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, n_kv_heads, cols) = (3usize, 4usize, 2usize, 4_097usize);
    let gqa_ratio = n_heads / n_kv_heads;
    let rows = seq_len * n_heads;
    let score_scale = 1.0 / 128.0f32.sqrt();
    let row_major = (0..rows * cols)
        .map(|index| {
            let bits = (index as u32)
                .wrapping_mul(1_103_515_245)
                .wrapping_add(12_345);
            (bits & 0xffff) as f32 / 8_192.0 - 4.0
        })
        .collect::<Vec<_>>();
    let mut packed = vec![0.0f32; row_major.len()];
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            for local_head in 0..gqa_ratio {
                let source_row = sequence * n_heads + kv_head * gqa_ratio + local_head;
                let packed_row = (kv_head * seq_len + sequence) * gqa_ratio + local_head;
                packed[packed_row * cols..(packed_row + 1) * cols]
                    .copy_from_slice(&row_major[source_row * cols..(source_row + 1) * cols]);
            }
        }
    }
    let row_major_tensor =
        upload_fp32_as_bf16(&ctx, &row_major, vec![rows, cols]).unwrap();
    let packed_tensor = upload_fp32_as_bf16(&ctx, &packed, vec![rows, cols]).unwrap();
    let kv_offset = (cols - seq_len) as u32;
    let standard = softmax_causal_bf16_scaled_plain(
        &ctx,
        &row_major_tensor,
        kv_offset,
        n_heads as u32,
        score_scale,
    )
    .unwrap();
    let packed = softmax_causal_bf16_scaled_plain_gqa_packed(
        &ctx,
        &packed_tensor,
        kv_offset,
        n_heads as u32,
        gqa_ratio as u32,
        score_scale,
    )
    .unwrap();
    let standard = download_bf16_as_fp32(&standard).unwrap();
    let packed = download_bf16_as_fp32(&packed).unwrap();
    let mut unpacked = vec![0.0f32; packed.len()];
    for kv_head in 0..n_kv_heads {
        for sequence in 0..seq_len {
            for local_head in 0..gqa_ratio {
                let destination_row = sequence * n_heads + kv_head * gqa_ratio + local_head;
                let packed_row = (kv_head * seq_len + sequence) * gqa_ratio + local_head;
                unpacked[destination_row * cols..(destination_row + 1) * cols]
                    .copy_from_slice(&packed[packed_row * cols..(packed_row + 1) * cols]);
            }
        }
    }
    assert_eq!(unpacked, standard);
}

#[test]
fn attention_softmax_in_place_scale_matches_plain_long_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, gqa_ratio) = (2usize, 4usize, 2usize);
    let rows = seq_len * n_heads;
    let score_scale = 1.0 / 128.0f32.sqrt();
    for cols in [4_097usize, 8_192, 12_288] {
        let values = (0..rows * cols)
            .map(|index| {
                let bits = (index as u32)
                    .wrapping_mul(22_695_477)
                    .wrapping_add(1);
                (bits & 0xffff) as f32 / 8_192.0 - 4.0
            })
            .collect::<Vec<_>>();
        let plain_input = upload_fp32_as_bf16(&ctx, &values, vec![rows, cols]).unwrap();
        let inplace_input = upload_fp32_as_bf16(&ctx, &values, vec![rows, cols]).unwrap();
        let kv_offset = (cols - seq_len) as u32;
        let plain = softmax_causal_bf16_scaled_plain_gqa_packed(
            &ctx,
            &plain_input,
            kv_offset,
            n_heads as u32,
            gqa_ratio as u32,
            score_scale,
        )
        .unwrap();
        let inplace = softmax_causal_bf16_scaled_in_place_gqa_packed(
            &ctx,
            inplace_input,
            kv_offset,
            n_heads as u32,
            gqa_ratio as u32,
            score_scale,
        )
        .unwrap();
        assert_eq!(
            download_bf16_as_fp32(&inplace).unwrap(),
            download_bf16_as_fp32(&plain).unwrap(),
            "in-place scale differed at {cols} columns"
        );
    }
}

#[test]
fn attention_softmax_global_exp_cache_decode_is_bit_exact_at_32k() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, cols) = (2usize, 32_768usize);
    let input = (0..n_heads * cols)
        .map(|index| ((index as f32 * 0.019) - 4.0).sin())
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, cols]).unwrap();
    let scalar = softmax_causal_with_exp_cache(
        &ctx,
        &tensor,
        (cols - 1) as u32,
        n_heads as u32,
        false,
    )
    .unwrap();
    let cached = softmax_causal_with_global_exp_cache(
        &ctx,
        &tensor,
        (cols - 1) as u32,
        n_heads as u32,
    )
    .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&cached).unwrap(),
        download_bf16_as_fp32(&scalar).unwrap()
    );
}

#[test]
fn attention_softmax_global_exp_cache_decode_is_bit_exact_across_long_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n_heads = 2usize;
    for cols in [11_265usize, 12_288, 16_385, 32_767] {
        let input = (0..n_heads * cols)
            .map(|index| {
                let bits = (index as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                let unit = (bits & 0xffff) as f32 / 65_535.0;
                unit * 24.0 - 12.0
            })
            .collect::<Vec<_>>();
        let tensor = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, cols]).unwrap();
        let scalar = softmax_causal_with_exp_cache(
            &ctx,
            &tensor,
            (cols - 1) as u32,
            n_heads as u32,
            false,
        )
        .unwrap();
        let cached = softmax_causal_with_global_exp_cache(
            &ctx,
            &tensor,
            (cols - 1) as u32,
            n_heads as u32,
        )
        .unwrap();
        let cached = download_bf16_as_fp32(&cached).unwrap();
        let scalar = download_bf16_as_fp32(&scalar).unwrap();
        assert_eq!(cached.len(), scalar.len());
        if let Some((index, (cached, scalar))) = cached
            .iter()
            .zip(&scalar)
            .enumerate()
            .find(|(_, (cached, scalar))| cached != scalar)
        {
            panic!(
                "global exp cache differed at {cols} columns, index {index}: \
                 cached={cached:?}, scalar={scalar:?}"
            );
        }
    }
}

#[test]
fn attention_softmax_bf16_crosses_legacy_grid_y_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let rows = 65_536usize;
    let cols = 1usize;
    let n_heads = 16u32;
    let input = vec![0.0; rows * cols];
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let output = softmax_causal(&ctx, &tensor, 0, n_heads).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    assert_eq!(actual.len(), rows);
    assert!(actual.iter().all(|value| (*value - 1.0).abs() <= 1e-3));
}

#[test]
fn attention_softmax_exp_cache_crosses_legacy_grid_y_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let rows = 65_536usize;
    let cols = 1usize;
    let tensor = upload_fp32_as_bf16(&ctx, &vec![0.0; rows], vec![rows, cols]).unwrap();
    let output = softmax_causal_with_exp_cache(&ctx, &tensor, 0, 16, true).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    assert_eq!(actual.len(), rows);
    assert!(actual.iter().all(|value| (*value - 1.0).abs() <= 1e-3));
}

#[test]
fn gqa_batched_prefill_matches_scalar_bf16() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, n_kv_heads, head_dim, max_seq_len) =
        (4usize, 4usize, 2usize, 16usize, 8usize);
    let q_values = (0..seq_len * n_heads * head_dim)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let k_values = (0..seq_len * n_kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.047) - 1.0).cos())
        .collect::<Vec<_>>();
    let v_values = (0..seq_len * n_kv_heads * head_dim)
        .map(|index| (index as f32 * 0.013) - 0.75)
        .collect::<Vec<_>>();
    let q = upload_fp32_as_bf16(
        &ctx,
        &q_values,
        vec![seq_len, n_heads, head_dim],
    )
    .unwrap();
    let k = upload_fp32_as_bf16(
        &ctx,
        &k_values,
        vec![seq_len, n_kv_heads, head_dim],
    )
    .unwrap();
    let v = upload_fp32_as_bf16(
        &ctx,
        &v_values,
        vec![seq_len, n_kv_heads, head_dim],
    )
    .unwrap();
    let cache = CudaKVCache::new(
        ctx.device_id(),
        1,
        n_kv_heads,
        head_dim,
        max_seq_len,
    )
    .unwrap();
    cache.append(&ctx, 0, &k, &v, seq_len).unwrap();

    let scalar = sdpa_with_batched_prefill(
        &ctx,
        &q,
        &cache,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
        0,
        false,
    )
    .unwrap();
    let batched = sdpa_with_batched_prefill(
        &ctx,
        &q,
        &cache,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        max_seq_len,
        0,
        true,
    )
    .unwrap();
    let scalar = download_bf16_as_fp32(&scalar).unwrap();
    let batched = download_bf16_as_fp32(&batched).unwrap();
    assert_bf16_close_reduction(&batched, &scalar);
}

#[test]
fn gqa_flattened_long_prefill_matches_scalar_bf16() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, kv_len, n_heads, n_kv_heads, head_dim, max_seq_len) =
        (3usize, 4_097usize, 4usize, 2usize, 16usize, 4_100usize);
    let q_values = (0..seq_len * n_heads * head_dim)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let k_values = (0..kv_len * n_kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.047) - 1.0).cos())
        .collect::<Vec<_>>();
    let v_values = (0..kv_len * n_kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.013) - 0.75).sin())
        .collect::<Vec<_>>();
    let q = upload_fp32_as_bf16(&ctx, &q_values, vec![seq_len, n_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k_values, vec![kv_len, n_kv_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v_values, vec![kv_len, n_kv_heads, head_dim]).unwrap();
    let cache = CudaKVCache::new(ctx.device_id(), 1, n_kv_heads, head_dim, max_seq_len).unwrap();
    cache.append(&ctx, 0, &k, &v, kv_len).unwrap();
    let kv_offset = (kv_len - seq_len) as u32;
    let scalar = sdpa_with_batched_prefill(
        &ctx,
        &q,
        &cache,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        max_seq_len,
        kv_offset,
        false,
    )
    .unwrap();
    let flattened = sdpa_with_batched_prefill(
        &ctx,
        &q,
        &cache,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_len,
        max_seq_len,
        kv_offset,
        true,
    )
    .unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&flattened).unwrap(),
        &download_bf16_as_fp32(&scalar).unwrap(),
    );
}

#[test]
fn qwen25_grouped4_split_cta_matches_accepted_long_decode() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (query_heads, kv_heads, head_dim, kv_len, max_seq_len) =
        (16usize, 2usize, 128usize, 32_767usize, 32_768usize);
    let query_values = (0..query_heads * head_dim)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let key_values = (0..kv_len * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.047) - 1.0).cos())
        .collect::<Vec<_>>();
    let value_values = (0..kv_len * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.013) - 0.75).sin())
        .collect::<Vec<_>>();
    let query = upload_fp32_as_bf16(
        &ctx,
        &query_values,
        vec![1, query_heads, head_dim],
    )
    .unwrap();
    let key = upload_fp32_as_bf16(&ctx, &key_values, vec![kv_len, kv_heads, head_dim])
        .unwrap();
    let value = upload_fp32_as_bf16(
        &ctx,
        &value_values,
        vec![kv_len, kv_heads, head_dim],
    )
    .unwrap();
    let cache = CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    cache.append(&ctx, 0, &key, &value, kv_len).unwrap();
    let accepted = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_heads * head_dim],
        vec![1, query_heads * head_dim],
    )
    .unwrap();
    let candidate = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_heads * head_dim],
        vec![1, query_heads * head_dim],
    )
    .unwrap();
    let position = CudaBuffer::alloc(std::mem::size_of::<u32>(), ctx.device_id()).unwrap();
    position
        .copy_from_host(&u32::try_from(kv_len - 1).unwrap().to_ne_bytes())
        .unwrap();
    let workspace = SplitCtaWorkspace::new(&ctx).unwrap();
    let scale = (head_dim as f32).sqrt().recip();
    grouped2_split_cta_write(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        &accepted,
        &workspace,
        48,
        kv_len,
        max_seq_len,
        scale,
        position.address(),
    )
    .unwrap();
    grouped4_split_cta_write(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        &candidate,
        &workspace,
        64,
        kv_len,
        max_seq_len,
        scale,
        position.address(),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&candidate).unwrap(),
        &download_bf16_as_fp32(&accepted).unwrap(),
    );
}

#[test]
fn qwen25_short_w32_attention_matches_w16_reduction_gate() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (query_heads, kv_heads, head_dim, kv_len, max_seq_len) =
        (16usize, 2usize, 128usize, 1_024usize, 32_768usize);
    let query_values = (0..query_heads * head_dim)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let key_values = (0..kv_len * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.047) - 1.0).cos())
        .collect::<Vec<_>>();
    let value_values = (0..kv_len * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.013) - 0.75).sin())
        .collect::<Vec<_>>();
    let query = upload_fp32_as_bf16(
        &ctx,
        &query_values,
        vec![1, query_heads, head_dim],
    )
    .unwrap();
    let key = upload_fp32_as_bf16(&ctx, &key_values, vec![kv_len, kv_heads, head_dim])
        .unwrap();
    let value = upload_fp32_as_bf16(
        &ctx,
        &value_values,
        vec![kv_len, kv_heads, head_dim],
    )
    .unwrap();
    let cache = CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    cache.append(&ctx, 0, &key, &value, kv_len).unwrap();
    let accepted = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_heads * head_dim],
        vec![1, query_heads * head_dim],
    )
    .unwrap();
    let candidate = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_heads * head_dim],
        vec![1, query_heads * head_dim],
    )
    .unwrap();
    let position = CudaBuffer::alloc(std::mem::size_of::<u32>(), ctx.device_id()).unwrap();
    position
        .copy_from_host(&u32::try_from(kv_len - 1).unwrap().to_ne_bytes())
        .unwrap();
    let query = CudaBuffer::from_tensor(&query).unwrap();
    let accepted_buffer = CudaBuffer::from_tensor(&accepted).unwrap();
    let candidate_buffer = CudaBuffer::from_tensor(&candidate).unwrap();
    let scale = (head_dim as f32).sqrt().recip();

    crate::kernels::attention::flash_bf16_into(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        &accepted_buffer,
        query_heads,
        kv_heads,
        head_dim,
        max_seq_len,
        max_seq_len,
        scale,
        position.address(),
    )
    .unwrap();
    short_w32_write(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        &candidate_buffer,
        max_seq_len,
        max_seq_len,
        scale,
        position.address(),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&candidate).unwrap(),
        &download_bf16_as_fp32(&accepted).unwrap(),
    );

    assert!(short_w32_write(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        &candidate_buffer,
        1_024,
        max_seq_len,
        scale,
        position.address(),
    )
    .unwrap_err()
    .to_string()
    .contains("W32 attention contract mismatch"));
}

#[test]
fn qwen25_packed_qkv_prelude_matches_three_node_contract() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (query_heads, kv_heads, head_dim, max_seq_len, cache_pos) =
        (16usize, 2usize, 128usize, 32_768usize, 1_024usize);
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let fused_width = query_width + 2 * kv_width;
    let qkv_values = (0..fused_width)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let bias_values = (0..fused_width)
        .map(|index| ((index as f32 * 0.017) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let baseline_qkv = upload_fp32_as_bf16(&ctx, &qkv_values, vec![1, fused_width]).unwrap();
    let candidate_qkv = upload_fp32_as_bf16(&ctx, &qkv_values, vec![1, fused_width]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![fused_width]).unwrap();
    let baseline_qkv_buffer = CudaBuffer::from_tensor(&baseline_qkv).unwrap();
    let candidate_qkv_buffer = CudaBuffer::from_tensor(&candidate_qkv).unwrap();
    let bias_buffer = CudaBuffer::from_tensor(&bias).unwrap();
    crate::kernels::elementwise::add_bias_bf16_into(
        &ctx,
        &baseline_qkv_buffer,
        &bias_buffer,
        &baseline_qkv_buffer,
        fused_width,
        1,
    )
    .unwrap();
    let element_bytes = DType::BF16.size_in_bytes();
    let baseline_q = baseline_qkv_buffer
        .view(0, query_width * element_bytes)
        .unwrap();
    let baseline_k = baseline_qkv_buffer
        .view(query_width * element_bytes, kv_width * element_bytes)
        .unwrap();
    let baseline_v = baseline_qkv_buffer
        .view(
            (query_width + kv_width) * element_bytes,
            kv_width * element_bytes,
        )
        .unwrap();
    let baseline_query = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_width],
        vec![1, query_width],
    )
    .unwrap();
    let candidate_query = upload_fp32_as_bf16(
        &ctx,
        &vec![0.0; query_width],
        vec![1, query_width],
    )
    .unwrap();
    let baseline_query_buffer = CudaBuffer::from_tensor(&baseline_query).unwrap();
    let candidate_query_buffer = CudaBuffer::from_tensor(&candidate_query).unwrap();
    let baseline_cache =
        CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    let candidate_cache =
        CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    let position_values = [1_024u32, 768u32, 512u32];
    let position_bytes = position_values
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect::<Vec<_>>();
    let positions = CudaBuffer::alloc(position_bytes.len(), ctx.device_id()).unwrap();
    positions.copy_from_host(&position_bytes).unwrap();
    let cache_position_buffer = CudaBuffer::alloc(4, ctx.device_id()).unwrap();
    cache_position_buffer
        .copy_from_host(&(cache_pos as u32).to_ne_bytes())
        .unwrap();
    let cache_position = cache_position_buffer.address();
    let theta = 1_000_000.0f32;
    let sections = [16usize, 24usize, 24usize];
    crate::kernels::rope::apply_tmrope_bf16_into(
        &ctx,
        &baseline_q,
        &baseline_query_buffer,
        head_dim,
        query_heads,
        theta,
        sections,
        positions.address(),
    )
    .unwrap();
    crate::kernels::rope::apply_tmrope_kv_write_bf16(
        &ctx,
        &baseline_k,
        &baseline_v,
        baseline_cache.k_buffer(0),
        baseline_cache.v_buffer(0),
        head_dim,
        kv_heads,
        max_seq_len,
        theta,
        sections,
        positions.address(),
        cache_position,
    )
    .unwrap();
    packed_qkv_prelude_write(
        &ctx,
        &candidate_qkv_buffer,
        &bias_buffer,
        &candidate_query_buffer,
        candidate_cache.k_buffer(0),
        candidate_cache.v_buffer(0),
        theta,
        positions.address(),
        cache_position,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        download_bf16_as_fp32(&candidate_query).unwrap(),
        download_bf16_as_fp32(&baseline_query).unwrap()
    );

    for head in 0..kv_heads {
        let offset = (head * max_seq_len * head_dim + cache_pos * head_dim) * element_bytes;
        for (baseline, candidate) in [
            (baseline_cache.k_buffer(0), candidate_cache.k_buffer(0)),
            (baseline_cache.v_buffer(0), candidate_cache.v_buffer(0)),
        ] {
            let baseline = baseline
                .view(offset, head_dim * element_bytes)
                .unwrap()
                .into_tensor(Shape::new(vec![head_dim]), DType::BF16);
            let candidate = candidate
                .view(offset, head_dim * element_bytes)
                .unwrap()
                .into_tensor(Shape::new(vec![head_dim]), DType::BF16);
            assert_eq!(
                download_bf16_as_fp32(&candidate).unwrap(),
                download_bf16_as_fp32(&baseline).unwrap()
            );
        }
    }

    assert!(packed_qkv_prelude_write(
        &ctx,
        &candidate_qkv_buffer,
        &bias_buffer,
        &candidate_query_buffer,
        candidate_cache.k_buffer(0),
        candidate_cache.v_buffer(0),
        10_000.0,
        positions.address(),
        cache_position,
    )
    .unwrap_err()
    .to_string()
    .contains("packed QKV prelude contract mismatch"));
}

#[test]
fn qwen25_prefill_packed_qkv_prelude_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (rows, query_heads, kv_heads, head_dim, max_seq_len, start_position) =
        (1_024usize, 16usize, 2usize, 128usize, 32_768usize, 17usize);
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let packed_width = query_width + 2 * kv_width;
    let packed_values = (0..rows * packed_width)
        .map(|index| ((index as f32 * 0.007) - 3.0).sin())
        .collect::<Vec<_>>();
    let bias_values = (0..packed_width)
        .map(|index| ((index as f32 * 0.013) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let packed = upload_fp32_as_bf16(&ctx, &packed_values, vec![rows, packed_width]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![packed_width]).unwrap();
    let positions_values = (0..rows)
        .flat_map(|row| {
            let position = (start_position + row) as u32;
            [position, position / 2 + 3, position / 3 + 7]
        })
        .collect::<Vec<_>>();
    let position_bytes = positions_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let positions = CudaBuffer::alloc(position_bytes.len(), ctx.device_id()).unwrap();
    positions.copy_from_host(&position_bytes).unwrap();
    let split = split_gqa_qkv_bias_bf16(
        &ctx,
        &packed,
        Some(&bias),
        query_heads,
        kv_heads,
        head_dim,
    )
    .unwrap();
    let theta = 1_000_000.0f32;
    let sections = [16usize, 24usize, 24usize];
    let baseline_query = apply_tmrope(
        &ctx,
        &split.q,
        query_heads,
        head_dim,
        theta,
        sections,
        &positions,
    )
    .unwrap();
    let baseline_key = apply_tmrope(
        &ctx,
        &split.k,
        kv_heads,
        head_dim,
        theta,
        sections,
        &positions,
    )
    .unwrap();
    let baseline_cache =
        CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    append(
        &ctx,
        baseline_cache.k_buffer(0),
        &baseline_key,
        kv_heads,
        head_dim,
        max_seq_len,
        start_position,
        rows,
    )
    .unwrap();
    append(
        &ctx,
        baseline_cache.v_buffer(0),
        &split.v,
        kv_heads,
        head_dim,
        max_seq_len,
        start_position,
        rows,
    )
    .unwrap();
    let candidate_cache =
        CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    let candidate_query = prefill_packed_qkv_prelude_write(
        &ctx,
        &packed,
        &bias,
        candidate_cache.k_buffer(0),
        candidate_cache.v_buffer(0),
        &positions,
        start_position,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        download_bf16_as_fp32(&candidate_query).unwrap(),
        download_bf16_as_fp32(&baseline_query).unwrap()
    );

    let element_bytes = DType::BF16.size_in_bytes();
    let slice_bytes = rows * head_dim * element_bytes;
    for head in 0..kv_heads {
        let offset = (head * max_seq_len + start_position) * head_dim * element_bytes;
        for (baseline, candidate) in [
            (baseline_cache.k_buffer(0), candidate_cache.k_buffer(0)),
            (baseline_cache.v_buffer(0), candidate_cache.v_buffer(0)),
        ] {
            let baseline = baseline
                .view(offset, slice_bytes)
                .unwrap()
                .into_tensor(Shape::new(vec![rows, head_dim]), DType::BF16);
            let candidate = candidate
                .view(offset, slice_bytes)
                .unwrap()
                .into_tensor(Shape::new(vec![rows, head_dim]), DType::BF16);
            assert_eq!(
                download_bf16_as_fp32(&candidate).unwrap(),
                download_bf16_as_fp32(&baseline).unwrap()
            );
        }
    }
}

#[cfg(apxinf_fa2_causal_sm80)]
#[test]
fn causal_fa2_long_gqa_prefill_matches_scalar_contract() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    for key_tokens in [1_024usize, 4_097usize] {
        assert_causal_fa2_matches_scalar(&ctx, key_tokens);
    }
}

#[cfg(apxinf_fa2_causal_sm80)]
fn assert_causal_fa2_matches_scalar(ctx: &CudaContext, key_tokens: usize) {
    let (query_tokens, query_heads, kv_heads, head_dim, max_seq_len) =
        (4usize, 16usize, 2usize, 128usize, key_tokens + 3);
    let q_values = (0..query_tokens * query_heads * head_dim)
        .map(|index| ((index as f32 * 0.031) - 2.0).sin())
        .collect::<Vec<_>>();
    let k_values = (0..key_tokens * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.047) - 1.0).cos())
        .collect::<Vec<_>>();
    let v_values = (0..key_tokens * kv_heads * head_dim)
        .map(|index| ((index as f32 * 0.013) - 0.75).sin())
        .collect::<Vec<_>>();
    let query = upload_fp32_as_bf16(
        &ctx,
        &q_values,
        vec![query_tokens, query_heads, head_dim],
    )
    .unwrap();
    let key = upload_fp32_as_bf16(
        &ctx,
        &k_values,
        vec![key_tokens, kv_heads, head_dim],
    )
    .unwrap();
    let value = upload_fp32_as_bf16(
        &ctx,
        &v_values,
        vec![key_tokens, kv_heads, head_dim],
    )
    .unwrap();
    let cache = CudaKVCache::new(ctx.device_id(), 1, kv_heads, head_dim, max_seq_len).unwrap();
    cache
        .append(&ctx, 0, &key, &value, key_tokens)
        .unwrap();
    let scalar = sdpa_with_batched_prefill(
        &ctx,
        &query,
        &cache,
        0,
        query_heads,
        kv_heads,
        head_dim,
        key_tokens,
        max_seq_len,
        (key_tokens - query_tokens) as u32,
        false,
    )
    .unwrap();
    let candidate = causal_fa2_gqa_prefill(
        &ctx,
        &query,
        cache.k_buffer(0),
        cache.v_buffer(0),
        query_tokens,
        query_heads,
        kv_heads,
        head_dim,
        key_tokens,
        max_seq_len,
    )
    .unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&candidate).unwrap(),
        &download_bf16_as_fp32(&scalar).unwrap(),
    );
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

#[test]
fn kv_cache_clear_zeroes_in_place() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_kv_heads, head_dim, max_seq_len, append_len) = (2usize, 4usize, 8usize, 2usize);
    let values = (0..append_len * n_kv_heads * head_dim)
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(
        &ctx,
        &values,
        vec![append_len, n_kv_heads, head_dim],
    )
    .unwrap();
    let mut cache = CudaKVCache::new(ctx.device_id(), 1, n_kv_heads, head_dim, max_seq_len)
        .unwrap();
    let key_ptr = cache.k_buffer(0).ptr();
    let value_ptr = cache.v_buffer(0).ptr();
    cache.append(&ctx, 0, &tensor, &tensor, append_len).unwrap();
    ctx.synchronize().unwrap();
    apxinf_core::KvCache::advance(&mut cache, append_len);
    apxinf_core::KvCache::clear(&mut cache).unwrap();

    assert_eq!(apxinf_core::KvCache::seq_len(&cache), 0);
    assert_eq!(cache.k_buffer(0).ptr(), key_ptr);
    assert_eq!(cache.v_buffer(0).ptr(), value_ptr);
    let mut key = vec![1u8; cache.k_buffer(0).len()];
    let mut value = vec![1u8; cache.v_buffer(0).len()];
    cache.k_buffer(0).copy_to_host(&mut key).unwrap();
    cache.v_buffer(0).copy_to_host(&mut value).unwrap();
    assert!(key.iter().all(|byte| *byte == 0));
    assert!(value.iter().all(|byte| *byte == 0));
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
fn tmrope_kv_write_matches_separate_rope_and_cache_appends() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (kv_heads, head_dim, max_seq_len) = (2usize, 128usize, 8usize);
    let sections = [16usize, 24usize, 24usize];
    let theta = 1_000_000.0f32;
    let k_values = (0..kv_heads * head_dim)
        .map(|index| (index as f32 * 0.007) - 0.5)
        .collect::<Vec<_>>();
    let v_values = (0..kv_heads * head_dim)
        .map(|index| (index as f32 * -0.005) + 0.75)
        .collect::<Vec<_>>();
    let k_tensor = upload_fp32_as_bf16(&ctx, &k_values, vec![kv_heads, head_dim]).unwrap();
    let v_tensor = upload_fp32_as_bf16(&ctx, &v_values, vec![kv_heads, head_dim]).unwrap();
    let k = CudaBuffer::from_tensor(&k_tensor).unwrap();
    let v = CudaBuffer::from_tensor(&v_tensor).unwrap();
    let positions = HostMappedBuffer::alloc(16, ctx.device_id()).unwrap();
    positions.write_u32s(&[7, 11, 13, 5]).unwrap();
    let tmrope_positions = positions.address_at(0, 12).unwrap();
    let cache_position = positions.address_at(12, 4).unwrap();
    let input_bytes = kv_heads * head_dim * DType::BF16.size_in_bytes();
    let cache_bytes = kv_heads * max_seq_len * head_dim * DType::BF16.size_in_bytes();
    let rotated = CudaBuffer::alloc_zeros(input_bytes, ctx.device_id()).unwrap();
    let reference_k = CudaBuffer::alloc_zeros(cache_bytes, ctx.device_id()).unwrap();
    let reference_v = CudaBuffer::alloc_zeros(cache_bytes, ctx.device_id()).unwrap();
    crate::kernels::rope::apply_tmrope_bf16_into(
        &ctx,
        &k,
        &rotated,
        head_dim,
        kv_heads,
        theta,
        sections,
        tmrope_positions,
    )
    .unwrap();
    crate::kernels::cache::append_at(
        &ctx,
        DType::BF16,
        &reference_k,
        &rotated,
        kv_heads,
        head_dim,
        max_seq_len,
        cache_position,
    )
    .unwrap();
    crate::kernels::cache::append_at(
        &ctx,
        DType::BF16,
        &reference_v,
        &v,
        kv_heads,
        head_dim,
        max_seq_len,
        cache_position,
    )
    .unwrap();

    let fused_k = CudaBuffer::alloc_zeros(cache_bytes, ctx.device_id()).unwrap();
    let fused_v = CudaBuffer::alloc_zeros(cache_bytes, ctx.device_id()).unwrap();
    crate::kernels::rope::apply_tmrope_kv_write_bf16(
        &ctx,
        &k,
        &v,
        &fused_k,
        &fused_v,
        head_dim,
        kv_heads,
        max_seq_len,
        theta,
        sections,
        tmrope_positions,
        cache_position,
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let mut reference_k_bytes = vec![0u8; cache_bytes];
    let mut reference_v_bytes = vec![0u8; cache_bytes];
    let mut fused_k_bytes = vec![0u8; cache_bytes];
    let mut fused_v_bytes = vec![0u8; cache_bytes];
    reference_k.copy_to_host(&mut reference_k_bytes).unwrap();
    reference_v.copy_to_host(&mut reference_v_bytes).unwrap();
    fused_k.copy_to_host(&mut fused_k_bytes).unwrap();
    fused_v.copy_to_host(&mut fused_v_bytes).unwrap();
    assert_eq!(fused_k_bytes, reference_k_bytes);
    assert_eq!(fused_v_bytes, reference_v_bytes);
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
fn qwen25_tmrope_bf16_matches_contiguous_section_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (1usize, 1usize, 128usize);
    let sections = [16usize, 24usize, 24usize];
    let positions = [3u32, 7u32, 11u32];
    let theta = 1_000_000.0f32;
    let input = (0..head_dim)
        .map(|index| (index as f32 * 0.03 - 1.0).sin())
        .collect::<Vec<_>>();
    let half = head_dim / 2;
    let boundaries = [2 * sections[0], 2 * (sections[0] + sections[1])];
    let expected = (0..head_dim)
        .map(|dimension| {
            let axis = if dimension < boundaries[0] { 0 } else if dimension < boundaries[1] { 1 } else { 2 };
            let pair = dimension % half;
            let angle = positions[axis] as f32
                / theta.powf(2.0 * pair as f32 / head_dim as f32);
            let rotated = if dimension < half { -input[dimension + half] } else { input[dimension - half] };
            input[dimension] * angle.cos() + rotated * angle.sin()
        })
        .collect::<Vec<_>>();
    let tensor = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let position_bytes = positions.into_iter().flat_map(u32::to_ne_bytes).collect::<Vec<_>>();
    let position_buffer = CudaBuffer::alloc(position_bytes.len(), ctx.device_id()).unwrap();
    position_buffer.copy_from_host(&position_bytes).unwrap();
    let output = apply_tmrope(
        &ctx,
        &tensor,
        n_heads,
        head_dim,
        theta,
        sections,
        &position_buffer,
    )
    .unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn qwen25_vision_qkv_bias_rope_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (sequence, heads, head_dim) = (64usize, 16usize, 80usize);
    let hidden = heads * head_dim;
    let values = (0..sequence * hidden)
        .map(|index| ((index as f32 * 0.017) - 3.0).sin())
        .collect::<Vec<_>>();
    let bias_values = (0..hidden)
        .map(|index| ((index as f32 * 0.013) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let query = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let query_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let key_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let value_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let positions = (0..sequence)
        .flat_map(|token| [u32::try_from(token / 8).unwrap(), u32::try_from(token % 8).unwrap()])
        .collect::<Vec<_>>();
    let position_bytes = positions
        .iter()
        .flat_map(|position| position.to_ne_bytes())
        .collect::<Vec<_>>();
    let position_buffer = CudaBuffer::alloc(position_bytes.len(), ctx.device_id()).unwrap();
    position_buffer.copy_from_host(&position_bytes).unwrap();

    let baseline_query = add_bias(&ctx, &query, &query_bias).unwrap();
    let baseline_key = add_bias(&ctx, &key, &key_bias).unwrap();
    let baseline_value = add_bias(&ctx, &value, &value_bias).unwrap();
    let baseline_query = baseline_query
        .reshape(vec![sequence, heads, head_dim])
        .unwrap();
    let baseline_key = baseline_key
        .reshape(vec![sequence, heads, head_dim])
        .unwrap();
    let baseline_value = baseline_value
        .reshape(vec![sequence, heads, head_dim])
        .unwrap();
    let baseline_query =
        apply_vision_2d(&ctx, &baseline_query, heads, head_dim, 10_000.0, &position_buffer)
            .unwrap();
    let baseline_key =
        apply_vision_2d(&ctx, &baseline_key, heads, head_dim, 10_000.0, &position_buffer)
            .unwrap();
    let (candidate_query, candidate_key, candidate_value) = qwen25_vision_qkv_bias_rope(
        &ctx,
        &query,
        &key,
        &value,
        &query_bias,
        &key_bias,
        &value_bias,
        10_000.0,
        &position_buffer,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    for (candidate, baseline) in [
        (&candidate_query, &baseline_query),
        (&candidate_key, &baseline_key),
        (&candidate_value, &baseline_value),
    ] {
        assert_eq!(
            download_bf16_as_fp32(candidate).unwrap(),
            download_bf16_as_fp32(baseline).unwrap()
        );
    }
    assert!(qwen25_vision_qkv_bias_rope(
        &ctx,
        &query,
        &key,
        &value,
        &query_bias,
        &key_bias,
        &value_bias,
        1_000_000.0,
        &position_buffer,
    )
    .unwrap_err()
    .to_string()
    .contains("vision QKV bias/RoPE contract mismatch"));
}

#[test]
fn qwen25_vision_grouped_qkv_bias_rope_writes_consumer_order_exactly() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (sequence, heads, head_dim) = (64usize, 16usize, 80usize);
    let hidden = heads * head_dim;
    let values = (0..sequence * hidden)
        .map(|index| ((index as f32 * 0.017) - 3.0).sin())
        .collect::<Vec<_>>();
    let bias_values = (0..hidden)
        .map(|index| ((index as f32 * 0.013) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let query = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &values, vec![sequence, hidden]).unwrap();
    let query_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let key_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let value_bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let positions = (0..sequence)
        .flat_map(|token| [u32::try_from(token / 8).unwrap(), u32::try_from(token % 8).unwrap()])
        .collect::<Vec<_>>();
    let position_bytes = positions
        .iter()
        .flat_map(|position| position.to_ne_bytes())
        .collect::<Vec<_>>();
    let position_buffer = CudaBuffer::alloc(position_bytes.len(), ctx.device_id()).unwrap();
    position_buffer.copy_from_host(&position_bytes).unwrap();
    let groups = (0..sequence)
        .map(|token| ((token / 2) % 5) as u32)
        .collect::<Vec<_>>();
    let (_, indices) = vision_group_plan(&groups).unwrap();
    let index_bytes = indices
        .iter()
        .flat_map(|index| index.to_ne_bytes())
        .collect::<Vec<_>>();
    let index_buffer = CudaBuffer::alloc(index_bytes.len(), ctx.device_id()).unwrap();
    index_buffer.copy_from_host(&index_bytes).unwrap();
    let regular = qwen25_vision_qkv_bias_rope(
        &ctx,
        &query,
        &key,
        &value,
        &query_bias,
        &key_bias,
        &value_bias,
        10_000.0,
        &position_buffer,
    )
    .unwrap();
    let grouped = qwen25_vision_grouped_qkv_bias_rope(
        &ctx,
        &query,
        &key,
        &value,
        &query_bias,
        &key_bias,
        &value_bias,
        10_000.0,
        &position_buffer,
        &index_buffer,
    )
    .unwrap();
    let packed_values = values
        .chunks_exact(hidden)
        .flat_map(|row| row.iter().copied().cycle().take(3 * hidden))
        .collect::<Vec<_>>();
    let packed = upload_fp32_as_bf16(
        &ctx,
        &packed_values,
        vec![sequence, 3 * hidden],
    )
    .unwrap();
    let packed_regular = qwen25_vision_packed_qkv_bias_rope(
        &ctx,
        &packed,
        &query_bias,
        &key_bias,
        &value_bias,
        10_000.0,
        &position_buffer,
        None,
    )
    .unwrap();
    let packed_grouped = qwen25_vision_packed_qkv_bias_rope(
        &ctx,
        &packed,
        &query_bias,
        &key_bias,
        &value_bias,
        10_000.0,
        &position_buffer,
        Some(&index_buffer),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    for (regular, grouped, packed_regular, packed_grouped) in [
        (&regular.0, &grouped.0, &packed_regular.0, &packed_grouped.0),
        (&regular.1, &grouped.1, &packed_regular.1, &packed_grouped.1),
        (&regular.2, &grouped.2, &packed_regular.2, &packed_grouped.2),
    ] {
        let regular = download_bf16_as_fp32(regular).unwrap();
        assert_eq!(download_bf16_as_fp32(packed_regular).unwrap(), regular);
        let expected = indices
            .iter()
            .flat_map(|&row| {
                let start = row as usize * hidden;
                regular[start..start + hidden].iter().copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(download_bf16_as_fp32(grouped).unwrap(), expected);
        assert_eq!(download_bf16_as_fp32(packed_grouped).unwrap(), expected);
    }
}

#[test]
fn qwen25_vision_bias_residual_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (sequence, hidden) = (64usize, 1_280usize);
    let mut projection_values = (0..sequence * hidden)
        .map(|index| ((index as f32 * 0.017) - 3.0).sin())
        .collect::<Vec<_>>();
    let mut bias_values = (0..hidden)
        .map(|index| ((index as f32 * 0.013) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let mut residual_values = (0..sequence * hidden)
        .map(|index| ((index as f32 * 0.011) + 0.5).cos())
        .collect::<Vec<_>>();
    projection_values[0] = 1.0;
    bias_values[0] = 2.0f32.powi(-8);
    residual_values[0] = 2.0f32.powi(-8);
    let projection =
        upload_fp32_as_bf16(&ctx, &projection_values, vec![sequence, hidden]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias_values, vec![hidden]).unwrap();
    let residual = upload_fp32_as_bf16(&ctx, &residual_values, vec![sequence, hidden]).unwrap();
    let rounded_projection = add_bias(&ctx, &projection, &bias).unwrap();
    let baseline = add(&ctx, &rounded_projection, &residual).unwrap();
    let candidate =
        qwen25_vision_bias_residual_exact(&ctx, &projection, &bias, &residual).unwrap();
    ctx.synchronize().unwrap();
    let baseline = download_bf16_as_fp32(&baseline).unwrap();
    let candidate = download_bf16_as_fp32(&candidate).unwrap();
    assert_eq!(candidate, baseline);
    assert_eq!(candidate[0], 1.0);
}

#[test]
fn qwen25_vision_gate_up_bias_silu_mul_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    if ctx.caps().sm != 89 {
        return;
    }
    let (sequence, intermediate) = (64usize, 3_420usize);
    let mut gate_values = (0..sequence * intermediate)
        .map(|index| ((index as f32 * 0.017) - 3.0).sin() * 4.0)
        .collect::<Vec<_>>();
    let mut up_values = (0..sequence * intermediate)
        .map(|index| ((index as f32 * 0.011) + 0.5).cos() * 3.0)
        .collect::<Vec<_>>();
    let mut gate_bias_values = (0..intermediate)
        .map(|index| ((index as f32 * 0.013) - 1.0).cos() * 0.25)
        .collect::<Vec<_>>();
    let mut up_bias_values = (0..intermediate)
        .map(|index| ((index as f32 * 0.019) + 0.25).sin() * 0.25)
        .collect::<Vec<_>>();
    gate_values[0] = 1.0;
    up_values[0] = 1.0;
    gate_bias_values[0] = 2.0f32.powi(-8);
    up_bias_values[0] = 2.0f32.powi(-8);
    let gate = upload_fp32_as_bf16(&ctx, &gate_values, vec![sequence, intermediate]).unwrap();
    let up = upload_fp32_as_bf16(&ctx, &up_values, vec![sequence, intermediate]).unwrap();
    let gate_bias = upload_fp32_as_bf16(&ctx, &gate_bias_values, vec![intermediate]).unwrap();
    let up_bias = upload_fp32_as_bf16(&ctx, &up_bias_values, vec![intermediate]).unwrap();
    let rounded_gate = add_bias(&ctx, &gate, &gate_bias).unwrap();
    let rounded_up = add_bias(&ctx, &up, &up_bias).unwrap();
    let baseline = silu_mul(&ctx, &rounded_gate, &rounded_up).unwrap();
    let candidate = qwen25_vision_gate_up_bias_silu_mul_exact(
        &ctx,
        &gate,
        &gate_bias,
        &up,
        &up_bias,
    )
    .unwrap();
    let packed_values = gate_values
        .chunks_exact(intermediate)
        .zip(up_values.chunks_exact(intermediate))
        .flat_map(|(gate, up)| gate.iter().chain(up).copied())
        .collect::<Vec<_>>();
    let packed = upload_fp32_as_bf16(
        &ctx,
        &packed_values,
        vec![sequence, 2 * intermediate],
    )
    .unwrap();
    let packed_candidate = qwen25_vision_packed_gate_up_bias_silu_mul_exact(
        &ctx,
        &packed,
        &gate_bias,
        &up_bias,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        download_bf16_as_fp32(&candidate).unwrap(),
        download_bf16_as_fp32(&baseline).unwrap()
    );
    assert_eq!(
        download_bf16_as_fp32(&packed_candidate).unwrap(),
        download_bf16_as_fp32(&baseline).unwrap()
    );
}

// ── Vision SDPA (non-causal full attention) ──────────────────────

#[test]
fn vision_sdpa_bf16_matches_reference() {
    assert_vision_sdpa_bf16_case(64);
}

#[test]
fn vision_sdpa_bf16_head72_matches_reference() {
    assert_vision_sdpa_bf16_case(72);
}

#[test]
fn vision_sdpa_bf16_head80_matches_reference() {
    assert_vision_sdpa_bf16_case(80);
}

#[test]
fn vision_sdpa_bf16_head128_matches_reference() {
    assert_vision_sdpa_bf16_case(128);
}

#[test]
fn grouped_sdpa_bf16_respects_window_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 1usize, 64usize);
    let q = upload_fp32_as_bf16(&ctx, &vec![1.0; seq_len * head_dim], vec![seq_len, n_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &vec![1.0; seq_len * head_dim], vec![seq_len, n_heads, head_dim]).unwrap();
    let mut values = vec![2.0; head_dim];
    values.extend(vec![8.0; head_dim]);
    let v = upload_fp32_as_bf16(&ctx, &values, vec![seq_len, n_heads, head_dim]).unwrap();
    let bytes = [0_u32, 1_u32]
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect::<Vec<_>>();
    let groups = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    groups.copy_from_host(&bytes).unwrap();
    let output = grouped(&ctx, &q, &k, &v, seq_len, n_heads, head_dim, &groups).unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&output).unwrap(),
        &values,
    );
}

#[test]
fn vision_group_plan_preserves_original_key_order() {
    let (offsets, indices) = vision_group_plan(&[0, 1, 0, 2, 1, 0]).unwrap();
    assert_eq!(offsets, [0, 3, 5, 6]);
    assert_eq!(indices, [0, 2, 5, 1, 4, 3]);
    assert!(vision_group_plan(&[]).is_err());
    assert!(vision_group_plan(&[3]).is_err());
}

#[test]
fn grouped_indexed_sdpa_bf16_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (65usize, 2usize, 80usize);
    let elements = seq_len * n_heads * head_dim;
    let q_values = (0..elements)
        .map(|index| (index as f32 * 0.013 - 2.0).sin() * 0.5)
        .collect::<Vec<_>>();
    let k_values = (0..elements)
        .map(|index| (index as f32 * 0.017 - 1.0).cos() * 0.4)
        .collect::<Vec<_>>();
    let v_values = (0..elements)
        .map(|index| (index as f32 * 0.019 - 0.5).sin() * 2.0)
        .collect::<Vec<_>>();
    let shape = vec![seq_len, n_heads, head_dim];
    let q = upload_fp32_as_bf16(&ctx, &q_values, shape.clone()).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k_values, shape.clone()).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v_values, shape).unwrap();
    let group_values = (0..seq_len)
        .map(|index| ((index / 2) % 5) as u32)
        .collect::<Vec<_>>();
    let (offset_values, index_values) = vision_group_plan(&group_values).unwrap();
    let upload_u32s = |values: &[u32]| {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
        buffer.copy_from_host(&bytes).unwrap();
        buffer
    };
    let groups = upload_u32s(&group_values);
    let offsets = upload_u32s(&offset_values);
    let indices = upload_u32s(&index_values);
    let baseline = grouped(
        &ctx, &q, &k, &v, seq_len, n_heads, head_dim, &groups,
    )
    .unwrap();
    let candidate = grouped_indexed(
        &ctx,
        &q,
        &k,
        &v,
        seq_len,
        n_heads,
        head_dim,
        &groups,
        &offsets,
        &indices,
        offset_values.len() - 1,
    )
    .unwrap();
    assert_eq!(
        download_bf16_as_fp32(&candidate).unwrap(),
        download_bf16_as_fp32(&baseline).unwrap()
    );
}

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
#[test]
fn grouped_varlen_fa2_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (96usize, 2usize, 80usize);
    let elements = seq_len * n_heads * head_dim;
    let q_values = (0..elements)
        .map(|index| (index as f32 * 0.013 - 2.0).sin() * 0.5)
        .collect::<Vec<_>>();
    let k_values = (0..elements)
        .map(|index| (index as f32 * 0.017 - 1.0).cos() * 0.4)
        .collect::<Vec<_>>();
    let v_values = (0..elements)
        .map(|index| (index as f32 * 0.019 - 0.5).sin() * 2.0)
        .collect::<Vec<_>>();
    let shape = vec![seq_len, n_heads, head_dim];
    let q = upload_fp32_as_bf16(&ctx, &q_values, shape.clone()).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k_values, shape.clone()).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v_values, shape).unwrap();
    let group_values = (0..seq_len)
        .map(|index| {
            if index % 4 == 0 {
                2
            } else if index % 3 == 0 {
                1
            } else {
                0
            }
        })
        .collect::<Vec<u32>>();
    let (offset_values, index_values) = vision_group_plan(&group_values).unwrap();
    let upload_u32s = |values: &[u32]| {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
        buffer.copy_from_host(&bytes).unwrap();
        buffer
    };
    let groups = upload_u32s(&group_values);
    let offsets = upload_u32s(&offset_values);
    let indices = upload_u32s(&index_values);
    let max_group_size = offset_values
        .windows(2)
        .map(|window| (window[1] - window[0]) as usize)
        .max()
        .unwrap();
    let baseline = grouped(
        &ctx, &q, &k, &v, seq_len, n_heads, head_dim, &groups,
    )
    .unwrap();
    let candidate = grouped_varlen_fa2(
        &ctx,
        &q,
        &k,
        &v,
        seq_len,
        n_heads,
        head_dim,
        &offsets,
        &indices,
        offset_values.len() - 1,
        max_group_size,
        false,
    )
    .unwrap();
    let row_elements = n_heads * head_dim;
    let pack = |values: &[f32]| {
        index_values
            .iter()
            .flat_map(|&row| {
                let start = row as usize * row_elements;
                values[start..start + row_elements].iter().copied()
            })
            .collect::<Vec<_>>()
    };
    let shape = vec![seq_len, n_heads, head_dim];
    let packed_q = upload_fp32_as_bf16(&ctx, &pack(&q_values), shape.clone()).unwrap();
    let packed_k = upload_fp32_as_bf16(&ctx, &pack(&k_values), shape.clone()).unwrap();
    let packed_v = upload_fp32_as_bf16(&ctx, &pack(&v_values), shape).unwrap();
    let prepacked = grouped_varlen_fa2(
        &ctx,
        &packed_q,
        &packed_k,
        &packed_v,
        seq_len,
        n_heads,
        head_dim,
        &offsets,
        &indices,
        offset_values.len() - 1,
        max_group_size,
        true,
    )
    .unwrap();
    assert_bf16_close_reduction(
        &download_bf16_as_fp32(&candidate).unwrap(),
        &download_bf16_as_fp32(&baseline).unwrap(),
    );
    assert_eq!(
        download_bf16_as_fp32(&prepacked).unwrap(),
        download_bf16_as_fp32(&candidate).unwrap()
    );
}

#[test]
fn audio_im2col_and_average_pool_match_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = upload_fp32_as_bf16(&ctx, &[1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let columns = im2col1d_bf16(&ctx, &input, 3, 1, 1).unwrap();
    assert_bf16_close_elementwise(
        &download_bf16_as_fp32(&columns).unwrap(),
        &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0],
    );
    let pooled = avg_pool1d_bf16(&ctx, &input, 2, 1).unwrap();
    assert_bf16_close_elementwise(
        &download_bf16_as_fp32(&pooled).unwrap(),
        &[1.5, 2.5],
    );
}

fn assert_vision_sdpa_bf16_case(head_dim: usize) {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq, n_heads) = (6usize, 2usize);
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
fn split_gqa_qkv_bias_bf16_handles_unequal_widths() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (tokens, q_heads, kv_heads, head_dim) = (2usize, 2usize, 1usize, 2usize);
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let total = q_width + 2 * kv_width;
    let values = (0..tokens * total).map(|i| i as f32).collect::<Vec<_>>();
    let bias = (0..total).map(|i| 0.25 * i as f32).collect::<Vec<_>>();
    let qkv = upload_fp32_as_bf16(&ctx, &values, vec![tokens, total]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &bias, vec![total]).unwrap();
    let split =
        split_gqa_qkv_bias_bf16(&ctx, &qkv, Some(&bias), q_heads, kv_heads, head_dim).unwrap();
    let mut expected_q = Vec::new();
    let mut expected_k = Vec::new();
    let mut expected_v = Vec::new();
    for token in 0..tokens {
        let row = token * total;
        expected_q.extend((0..q_width).map(|i| values[row + i] + 0.25 * i as f32));
        expected_k
            .extend((0..kv_width).map(|i| values[row + q_width + i] + 0.25 * (q_width + i) as f32));
        expected_v.extend((0..kv_width).map(|i| {
            values[row + q_width + kv_width + i] + 0.25 * (q_width + kv_width + i) as f32
        }));
    }
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&split.q).unwrap(), &expected_q);
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&split.k).unwrap(), &expected_k);
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&split.v).unwrap(), &expected_v);
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
