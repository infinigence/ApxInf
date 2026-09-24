use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::activation::{gelu_tanh, silu};
use crate::kernels::attention::{causal_mask, softmax, softmax_causal, vision};
use crate::kernels::cache::append;
use crate::kernels::elementwise::{add, add_bias, concat_columns_bf16, mul, scale};
use crate::kernels::embedding::lookup;
use crate::kernels::norm::{layer, rms};
use crate::kernels::rope::{apply, apply_batched, apply_mrope, apply_vision_2d};

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

#[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
#[test]
fn sdpa_fa2_prefill_supports_direct_output_projection() {
    assert!(
        std::env::var_os("APXINF_PREFILL_SDPA_LEGACY").is_none(),
        "unset APXINF_PREFILL_SDPA_LEGACY to exercise the FA2 fast path"
    );
    let _guard = super::gpu_smem_guard();
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kv_heads, head_dim, max_seq) = (4, 2, 128, 8);
    let width = heads * head_dim;
    // Select one channel per head, as a small stand-in for Llama's wo.
    let mut weights = vec![0.0; width * heads];
    for head in 0..heads {
        weights[(head * head_dim) * heads + head] = 1.0;
    }
    let wo = upload_fp32_as_bf16(&ctx, &weights, vec![width, heads]).unwrap();
    for tokens in [1, 5] {
        let q = upload_fp32_as_bf16(
            &ctx,
            &vec![0.0; tokens * width],
            vec![tokens, heads, head_dim],
        )
        .unwrap();
        let kv_shape = vec![tokens, kv_heads, head_dim];
        let k = upload_fp32_as_bf16(
            &ctx,
            &vec![0.0; tokens * kv_heads * head_dim],
            kv_shape.clone(),
        )
        .unwrap();
        let values: Vec<f32> = (0..tokens * kv_heads * head_dim)
            .map(|i| (i / (kv_heads * head_dim) + 1) as f32)
            .collect();
        let v = upload_fp32_as_bf16(&ctx, &values, kv_shape).unwrap();
        let cache = crate::CudaKVCache::new(0, 1, kv_heads, head_dim, max_seq).unwrap();
        cache.append(&ctx, 0, &k, &v, tokens).unwrap();
        let out = crate::kernels::attention::sdpa(
            &ctx, &q, &cache, 0, heads, kv_heads, head_dim, tokens, max_seq, 0,
        )
        .unwrap();
        assert_eq!(out.shape().dims(), &[tokens, width]);
        // No caller-side reshape: this is the LlamaModel calling convention.
        let projected = crate::kernels::gemm::matmul(&ctx, &out, &wo).unwrap();
        assert_eq!(projected.shape().dims(), &[tokens, heads]);
        // Zero Q/K gives uniform causal attention: mean of values 1..=t+1.
        let expected: Vec<f32> = (0..tokens * heads)
            .map(|i| (i / heads + 2) as f32 / 2.0)
            .collect();
        let actual = download_bf16_as_fp32(&projected).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
        assert_bf16_close_reduction(&actual, &expected);
    }
}

#[test]
fn gdn_preparation_separates_prefill_and_decode_precision() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let mut values = (1..=8).map(|v| v as f32).collect::<Vec<_>>();
    values.extend((1..=8).rev().map(|v| v as f32));
    values.extend([0.0; 8]);
    let input = upload_fp32_as_bf16(&ctx, &values, vec![1, 24]).unwrap();
    let q = CudaBuffer::alloc(32, 0).unwrap();
    let k = CudaBuffer::alloc(32, 0).unwrap();
    for recurrent in [false, true] {
        crate::kernels::linear_attention::gdn_qk_prep(
            &ctx, &input, &q, &k, 1, 1, 1, 8, 8, recurrent, 1e-6,
        )
        .unwrap();
        let read = |b: &CudaBuffer| {
            crate::transfers::to_cpu(&b.as_tensor(Shape::new(vec![1, 8]), DType::F32).unwrap())
                .unwrap()
                .to_f32_vec()
                .unwrap()
        };
        let (actual_q, actual_k) = (read(&q), read(&k));
        for i in 0..8 {
            let qn = (values[i] as f64 / (204.0f64 + 1e-6).sqrt()) as f32;
            let kn = (values[8 + i] as f64 / (204.0f64 + 1e-6).sqrt()) as f32;
            if recurrent {
                assert!((actual_q[i] - qn * (1.0f64 / 8.0f64.sqrt()) as f32).abs() < 1e-6);
                assert!((actual_k[i] - kn).abs() < 1e-6);
            } else {
                assert_eq!(actual_q[i], half::bf16::from_f32(qn).to_f32());
                assert_eq!(actual_k[i], half::bf16::from_f32(kn).to_f32());
            }
        }
    }
}

#[test]
fn bf16_linear_bias_rounds_after_accumulation() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // The product is 1.015686..., which rounds to 1.015625 in BF16.
    // Adding the bias before rounding preserves the nonzero residual.
    let mut x = vec![0.0; 4 * 8];
    let mut w = vec![0.0; 8 * 8];
    for row in 0..4 {
        x[row * 8] = 1.0078125;
    }
    for col in 0..8 {
        w[col] = 1.0078125;
    }
    let xt = upload_fp32_as_bf16(&ctx, &x, vec![4, 8]).unwrap();
    let wt = upload_fp32_as_bf16(&ctx, &w, vec![8, 8]).unwrap();
    let bt = upload_fp32_as_bf16(&ctx, &[-1.015625; 8], vec![8]).unwrap();
    let out = crate::kernels::gemm::bf16_bias(&ctx, &xt, &wt, &bt).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    assert_eq!(actual, vec![1.0 / 16384.0; 32]);
}

#[test]
fn rounded_swiglu_preserves_bf16_activation_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, inner) = (3, 129);
    let input: Vec<f32> = (0..rows * inner * 2)
        .map(|i| half::bf16::from_f32((i % 101) as f32 / 8.0 - 6.0).to_f32())
        .collect();
    let expected: Vec<f32> = (0..rows * inner)
        .map(|i| {
            let row = i / inner;
            let col = i % inner;
            let gate = input[row * 2 * inner + col];
            let up = input[row * 2 * inner + inner + col];
            half::bf16::from_f32(half::bf16::from_f32(silu_ref(gate)).to_f32() * up).to_f32()
        })
        .collect();
    let x = upload_fp32_as_bf16(&ctx, &input, vec![rows, 2 * inner]).unwrap();
    let y = crate::kernels::activation::swiglu_bf16_rounded(&ctx, &x).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&y).unwrap(), &expected);
}

#[test]
fn composed_joint_gqa_is_unmasked_and_respects_head_groups() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Include more queries than keys and a query tile boundary. Noncausal
    // attention must attend all keys in both cases without offset underflow.
    for (queries, keys, heads, kv_heads, dim) in
        [(3, 5, 4, 2, 256), (5, 3, 4, 2, 256), (1025, 3, 2, 1, 16)]
    {
        let values = |n, shift| {
            (0..n)
                .map(|i| half::bf16::from_f32((((i + shift) % 37) as f32 - 18.0) / 32.0).to_f32())
                .collect::<Vec<f32>>()
        };
        let q = values(queries * heads * dim, 0);
        let k = values(keys * kv_heads * dim, 7);
        let v = values(keys * kv_heads * dim, 19);
        let mut expected = vec![0.0f32; queries * heads * dim];
        for row in 0..queries {
            for head in 0..heads {
                let group = head / (heads / kv_heads);
                let mut scores = vec![0.0f32; keys];
                for token in 0..keys {
                    scores[token] = (0..dim)
                        .map(|d| {
                            q[(row * heads + head) * dim + d]
                                * k[(token * kv_heads + group) * dim + d]
                        })
                        .sum::<f32>()
                        / (dim as f32).sqrt();
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                for d in 0..dim {
                    expected[(row * heads + head) * dim + d] = (0..keys)
                        .map(|t| {
                            (scores[t] - max).exp() / total * v[(t * kv_heads + group) * dim + d]
                        })
                        .sum();
                }
            }
        }
        let qt = upload_fp32_as_bf16(&ctx, &q, vec![queries, heads, dim]).unwrap();
        let kt = upload_fp32_as_bf16(&ctx, &k, vec![keys, kv_heads, dim]).unwrap();
        let vt = upload_fp32_as_bf16(&ctx, &v, vec![keys, kv_heads, dim]).unwrap();
        let actual =
            crate::kernels::attention::composed_gqa_bf16(&ctx, &qt, &kt, &vt, keys, false).unwrap();
        assert_bf16_close_reduction(&download_bf16_as_fp32(&actual).unwrap(), &expected);
    }
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

#[test]
fn concat_columns_bf16_matches_reference_and_can_be_captured() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let left = upload_fp32_as_bf16(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let right =
        upload_fp32_as_bf16(&ctx, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
    let expected = [1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0];
    let workspace = crate::workspace::GraphWorkspace::new(4096, 0).unwrap();

    let eager = crate::workspace::prepare_with_workspace(&workspace, || {
        concat_columns_bf16(&ctx, &[&left, &right])
    })
    .unwrap();
    ctx.synchronize().unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&eager).unwrap(), &expected);
    drop(eager);

    crate::graph::begin(&ctx, crate::graph::CaptureMode::ThreadLocal).unwrap();
    let captured = crate::workspace::with_workspace(&workspace, || {
        concat_columns_bf16(&ctx, &[&left, &right])
    })
    .unwrap();
    let graph = crate::graph::end(&ctx).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_bf16_close_elementwise(&download_bf16_as_fp32(&captured).unwrap(), &expected);
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

// ── vision segmented MHA against an fp64 oracle ───────────────────
//
// The end-to-end gate cannot decide whether a change to this operator cost
// precision. It reports where greedy decoding first leaves a reference the
// port does not match anyway, and that index is a chaotic function of last-bit
// rounding: the same kernel swap moves it 170 -> 102 while moving the
// trajectory RMS by 6%. This compares the operator itself, on one fixed input,
// against a double-precision reference of the same math, so a route can be
// called more or less accurate on its own terms.
//
// Run it per route:
//   cargo test --release -p apxinf-cuda vision_segmented_mha -- --nocapture
//   APXINF_VISION_FA2=1 cargo test ... (FlashAttention-2 head-64)
#[test]
fn vision_segmented_mha_error_against_fp64_oracle() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, dim, seg_tokens, segments) = (16usize, 64usize, 256usize, 2usize);
    let tokens = seg_tokens * segments;

    // Deterministic operands in the range a post-norm projection produces.
    let gen = |salt: u64| -> Vec<f32> {
        (0..tokens * heads * dim)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                ((x & 0xFFFF) as f32 / 32768.0 - 1.0) * 0.8
            })
            .collect()
    };
    let (qh, kh, vh) = (gen(1), gen(2), gen(3));
    let shape = vec![tokens, heads, dim];
    let q = upload_fp32_as_bf16(&ctx, &qh, shape.clone()).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &kh, shape.clone()).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &vh, shape.clone()).unwrap();

    // The operands the device sees are the BF16 roundings of the vectors
    // above, so the oracle has to start from those, not from the fp32 draws.
    let to_bf16 = |x: f32| -> f64 {
        let bits = x.to_bits();
        let rounded = ((bits >> 16)
            + (((bits >> 15) & 1) & ((bits & 0x7FFF != 0) as u32 | ((bits >> 16) & 1))))
            << 16;
        f32::from_bits(rounded) as f64
    };

    let host_offsets: Vec<u32> = (0..=segments).map(|s| (s * seg_tokens) as u32).collect();
    let offsets = CudaBuffer::alloc(host_offsets.len() * 4, 0).unwrap();
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            offsets.ptr(),
            host_offsets.as_ptr() as *const std::ffi::c_void,
            host_offsets.len() * 4,
            crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))
        .unwrap();
    }

    let out = crate::kernels::attention::segmented_mha_bf16(
        &ctx,
        &q,
        &k,
        &v,
        &offsets,
        &host_offsets,
        segments,
        seg_tokens,
    )
    .unwrap();
    let got = download_bf16_as_fp32(&out).unwrap();

    // fp64 reference: per segment, per head, scores/sqrt(dim), softmax, P*V.
    let scale = 1.0f64 / (dim as f64).sqrt();
    let mut worst = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut count = 0usize;
    let mut row = vec![0.0f64; seg_tokens];
    for seg in 0..segments {
        let base = seg * seg_tokens;
        for h in 0..heads {
            let at = |t: usize, d: usize, src: &Vec<f32>| -> f64 {
                to_bf16(src[((base + t) * heads + h) * dim + d])
            };
            for i in 0..seg_tokens {
                let mut max = f64::NEG_INFINITY;
                for j in 0..seg_tokens {
                    let mut acc = 0.0f64;
                    for d in 0..dim {
                        acc += at(i, d, &qh) * at(j, d, &kh);
                    }
                    row[j] = acc * scale;
                    if row[j] > max {
                        max = row[j];
                    }
                }
                let mut denom = 0.0f64;
                for j in 0..seg_tokens {
                    row[j] = (row[j] - max).exp();
                    denom += row[j];
                }
                for d in 0..dim {
                    let mut acc = 0.0f64;
                    for j in 0..seg_tokens {
                        acc += row[j] * at(j, d, &vh);
                    }
                    let expect = acc / denom;
                    let actual = got[((base + i) * heads + h) * dim + d] as f64;
                    let delta = (actual - expect).abs();
                    worst = worst.max(delta / expect.abs().max(1e-6));
                    sum_abs += delta;
                    sum_ref += expect.abs();
                    count += 1;
                }
            }
        }
    }
    #[cfg(apxinf_fa2_head_special)]
    let specialized = crate::kernels::attention::vision_fa2_enabled();
    #[cfg(not(apxinf_fa2_head_special))]
    let specialized = false;
    let route = if specialized {
        "fa2-head64"
    } else if cfg!(apxinf_fa2_sm80) {
        "fa2-generic"
    } else {
        "composed"
    };
    println!(
        "vision_mha_oracle route={route} elements={count} \
         mean_abs={:.6e} rel_l1={:.6e} max_rel={:.6e}",
        sum_abs / count as f64,
        sum_abs / sum_ref,
        worst
    );
    // A BF16 output carries about 2^-8 of relative resolution, so an aggregate
    // relative L1 above a few times that is a real loss, not rounding.
    assert!(
        sum_abs / sum_ref < 0.02,
        "vision segmented MHA relative L1 {} against fp64",
        sum_abs / sum_ref
    );
}

// ── GDN decode recurrence against an fp64 oracle ──────────────────
//
// The split kernel regroups the two k-term sums: each thread sums its own
// contiguous stripe of rows and the stripes are combined in index order,
// instead of one sequential sum over all of them. Same terms, same fp32
// arithmetic, different association. This says what that costs, against a
// double-precision reference of the same recurrence.
//
//   cargo test --release -p apxinf-cuda gdn_recurrent_decode -- --nocapture
//   APXINF_GDN_RECURRENT_SPLIT=1 cargo test ...   (the scalar kernel)
#[test]
fn gdn_recurrent_decode_error_against_fp64_oracle() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kdim, vdim) = (32usize, 128usize, 128usize);
    let draw = |salt: u64, n: usize, scale: f64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                (((x & 0xFFFF) as f64 / 32768.0 - 1.0) * scale) as f32
            })
            .collect()
    };
    // A flat draw does not exercise the regrouping: partial sums of equal-scale
    // terms round the same way whatever order they are added in, and split 1,
    // 2 and 4 come out bit-identical. Give k a magnitude ramp across the
    // reduction axis instead, so the stripes carry different scales and the
    // association actually matters.
    let q = draw(11, heads * kdim, 1.0);
    let k: Vec<f32> = draw(22, heads * kdim, 1.0)
        .iter()
        .enumerate()
        .map(|(i, x)| x * (2.0f32).powi(((i % kdim) as i32 - 64) / 8))
        .collect();
    let v = draw(33, heads * vdim, 1.0);
    let beta = draw(44, heads, 0.5);
    // g is a log-decay: keep it negative so exp(g) is a contraction.
    let g: Vec<f32> = draw(55, heads, 0.5).iter().map(|x| -x.abs()).collect();
    let state0 = draw(66, heads * kdim * vdim, 0.3);

    let upload = |data: &[f32]| -> CudaBuffer {
        let buf = CudaBuffer::alloc(data.len() * 4, 0).unwrap();
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
                buf.ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * 4,
                crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))
            .unwrap();
        }
        buf
    };
    let (qb, kb, vb, bb, gb) = (
        upload(&q),
        upload(&k),
        upload(&v),
        upload(&beta),
        upload(&g),
    );
    let sb = upload(&state0);
    let out = CudaBuffer::alloc(heads * vdim * 2, 0).unwrap();

    crate::kernels::linear_attention::gdn_recurrent(
        &ctx,
        &qb,
        &kb,
        &vb,
        &bb,
        &gb,
        &sb,
        &out.as_tensor(Shape::new(vec![1, heads * vdim]), DType::BF16)
            .unwrap(),
        heads,
        kdim,
        vdim,
    )
    .unwrap();
    let got = download_bf16_as_fp32(
        &out.as_tensor(Shape::new(vec![1, heads * vdim]), DType::BF16)
            .unwrap(),
    )
    .unwrap();

    // fp64 reference of the same recurrence.
    let mut sum_abs = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut worst = 0.0f64;
    let mut state = vec![0.0f64; kdim * vdim];
    for h in 0..heads {
        let g_exp = (g[h] as f64).exp();
        for i in 0..kdim * vdim {
            state[i] = state0[h * kdim * vdim + i] as f64 * g_exp;
        }
        for j in 0..vdim {
            let mut kv = 0.0f64;
            for m in 0..kdim {
                kv += state[m * vdim + j] * k[h * kdim + m] as f64;
            }
            let delta = (v[h * vdim + j] as f64 - kv) * beta[h] as f64;
            let mut acc = 0.0f64;
            for m in 0..kdim {
                let updated = state[m * vdim + j] + k[h * kdim + m] as f64 * delta;
                acc += updated * q[h * kdim + m] as f64;
            }
            let actual = got[h * vdim + j] as f64;
            let delta_abs = (actual - acc).abs();
            sum_abs += delta_abs;
            sum_ref += acc.abs();
            worst = worst.max(delta_abs / acc.abs().max(1e-6));
        }
    }
    let split = std::env::var("APXINF_GDN_RECURRENT_SPLIT").unwrap_or_else(|_| "default".into());
    println!(
        "gdn_recurrent_oracle split={split} elements={} mean_abs={:.6e} rel_l1={:.6e} max_rel={:.6e}",
        heads * vdim,
        sum_abs / (heads * vdim) as f64,
        sum_abs / sum_ref,
        worst
    );
    assert!(
        sum_abs / sum_ref < 0.02,
        "GDN decode recurrence relative L1 {} against fp64",
        sum_abs / sum_ref
    );
}

// ── GDN chunk-state scan against an fp64 oracle ───────────────────
//
// This kernel is the largest single one in a VQA scene -- 452 ms, 38% of
// prefill -- and it is scalar fp32 on CUDA cores, where Thor is 1.59x Orin
// while its tensor cores are 11x. Its four inner products are GEMM-shaped and
// in every one of them the right-hand operand is already on the BF16 grid:
// the carried state is rounded to BF16 by the kernel itself, and v_new is
// rounded before both the intra term and the state update. So rounding the
// left operand to BF16 as well does not open a new order of error, it roughly
// doubles an error the term already has -- which is a claim to measure, not to
// assert.
//
// This fixes the input and reports the error of whatever the kernel does
// against a double-precision reference of the scan.
//
//   cargo test --release -p apxinf-cuda gdn_chunk_state_scan -- --nocapture
#[test]
fn gdn_chunk_state_scan_error_against_fp64_oracle() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kdim, vdim, chunk, chunks) = (4usize, 128usize, 128usize, 64usize, 3usize);
    let seq_pad = chunk * chunks;
    let seq = seq_pad;

    let draw = |salt: u64, n: usize, scale: f64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                (((x & 0xFFFF) as f64 / 32768.0 - 1.0) * scale) as f32
            })
            .collect()
    };
    let q = draw(1, heads * seq_pad * kdim, 0.5);
    let k = draw(2, heads * seq_pad * kdim, 0.5);
    let t = draw(3, heads * chunks * chunk * chunk, 0.25);
    let vt = draw(4, heads * chunks * chunk * vdim, 0.5);
    let kcd = draw(5, heads * chunks * chunk * kdim, 0.25);
    let state0 = draw(6, heads * kdim * vdim, 0.2);
    // g_cum is a cumulative log-decay: non-increasing within a chunk.
    let mut g_cum = vec![0.0f32; heads * seq_pad];
    for h in 0..heads {
        for c in 0..chunks {
            let mut acc = 0.0f32;
            for i in 0..chunk {
                acc -= 0.01 * ((h + c + i) % 5) as f32;
                g_cum[h * seq_pad + c * chunk + i] = acc;
            }
        }
    }

    let upload = |data: &[f32]| -> CudaBuffer {
        let buf = CudaBuffer::alloc(data.len() * 4, 0).unwrap();
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
                buf.ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * 4,
                crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))
            .unwrap();
        }
        buf
    };
    let (qb, kb, gb, tb, vtb, kcdb) = (
        upload(&q),
        upload(&k),
        upload(&g_cum),
        upload(&t),
        upload(&vt),
        upload(&kcd),
    );
    let sb = upload(&state0);
    let out = CudaBuffer::alloc(seq * heads * vdim * 2, 0).unwrap();
    let out_t = out
        .as_tensor(Shape::new(vec![seq, heads * vdim]), DType::BF16)
        .unwrap();
    crate::kernels::linear_attention::gdn_chunk_state(
        &ctx, &qb, &kb, &gb, &tb, &vtb, &kcdb, &sb, &out_t, seq_pad, heads, kdim, vdim, chunk,
    )
    .unwrap();
    let got = download_bf16_as_fp32(&out_t).unwrap();

    // fp64 reference. The BF16 roundings the kernel performs deliberately are
    // part of the definition here, not error: the carried state and v_new are
    // rounded before use, and the reference does the same.
    let bf = |x: f64| -> f64 {
        let v = x as f32;
        let bits = v.to_bits();
        let r = ((bits >> 16)
            + (((bits >> 15) & 1) & ((bits & 0x7FFF != 0) as u32 | ((bits >> 16) & 1))))
            << 16;
        f32::from_bits(r) as f64
    };
    let scale = 1.0f64 / (kdim as f64).sqrt();
    let mut sum_abs = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut worst = 0.0f64;
    for h in 0..heads {
        let mut state = vec![0.0f64; kdim * vdim];
        for i in 0..kdim * vdim {
            state[i] = state0[h * kdim * vdim + i] as f64;
        }
        for c in 0..chunks {
            let tok = h * seq_pad + c * chunk;
            let mut v_new = vec![0.0f64; chunk * vdim];
            let mut inter = vec![0.0f64; chunk * vdim];
            for i in 0..chunk {
                let qg = (g_cum[tok + i] as f64).exp2();
                for j in 0..vdim {
                    let mut vp = 0.0f64;
                    let mut ai = 0.0f64;
                    for m in 0..kdim {
                        let sv = bf(state[m * vdim + j]);
                        vp += kcd[((h * chunks + c) * chunk + i) * kdim + m] as f64 * sv;
                        ai += q[(tok + i) * kdim + m] as f64 * sv;
                    }
                    v_new[i * vdim + j] = vt[((h * chunks + c) * chunk + i) * vdim + j] as f64 - vp;
                    inter[i * vdim + j] = ai * qg;
                }
            }
            for i in 0..chunk {
                for j in 0..vdim {
                    let mut intra = 0.0f64;
                    for m in 0..chunk {
                        intra += t[((h * chunks + c) * chunk + i) * chunk + m] as f64
                            * bf(v_new[m * vdim + j]);
                    }
                    let acc = inter[i * vdim + j] * scale + intra * scale;
                    let token = c * chunk + i;
                    let actual = got[token * heads * vdim + h * vdim + j] as f64;
                    let delta = (actual - acc).abs();
                    sum_abs += delta;
                    sum_ref += acc.abs();
                    worst = worst.max(delta / acc.abs().max(1e-6));
                }
            }
            let g_last = g_cum[tok + chunk - 1] as f64;
            let decay = g_last.exp2();
            let mut vr = vec![0.0f64; chunk * vdim];
            for i in 0..chunk {
                let w = (g_last - g_cum[tok + i] as f64).exp2();
                for j in 0..vdim {
                    vr[i * vdim + j] = bf(v_new[i * vdim + j] * w);
                }
            }
            for m in 0..kdim {
                for j in 0..vdim {
                    let mut acc = 0.0f64;
                    for i in 0..chunk {
                        acc += k[(tok + i) * kdim + m] as f64 * vr[i * vdim + j];
                    }
                    state[m * vdim + j] = state[m * vdim + j] * decay + acc;
                }
            }
        }
    }
    println!(
        "gdn_chunk_state_oracle tile={} elements={} mean_abs={:.6e} rel_l1={:.6e} max_rel={:.6e}",
        std::env::var("APXINF_GDN_CHUNK_TILE").unwrap_or_else(|_| "default".into()),
        seq * heads * vdim,
        sum_abs / (seq * heads * vdim) as f64,
        sum_abs / sum_ref,
        worst
    );
    assert!(
        sum_abs / sum_ref < 0.05,
        "GDN chunk-state relative L1 {} against fp64",
        sum_abs / sum_ref
    );
}

// ── GDN chunk-state value split is bit-exact ──────────────────────
//
// Splitting a head's scan across several blocks along the value dimension is
// a claim about the arithmetic, not a measurement: nothing in the kernel
// crosses that dimension, so each slice sums the same terms in the same order
// and the result should be identical to the last bit, not merely within an
// oracle's tolerance. That is a claim a test can settle exactly, so it does --
// the output words and the carried state are compared as bits.
//
//   cargo test --release -p apxinf-cuda gdn_chunk_state_v_split -- --nocapture
#[test]
fn gdn_chunk_state_v_split_is_bit_exact() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kdim, vdim, chunk, chunks) = (4usize, 128usize, 128usize, 64usize, 3usize);
    let seq_pad = chunk * chunks;
    let seq = seq_pad;

    let draw = |salt: u64, n: usize, scale: f64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                (((x & 0xFFFF) as f64 / 32768.0 - 1.0) * scale) as f32
            })
            .collect()
    };
    let q = draw(1, heads * seq_pad * kdim, 0.5);
    let k = draw(2, heads * seq_pad * kdim, 0.5);
    let t = draw(3, heads * chunks * chunk * chunk, 0.25);
    let vt = draw(4, heads * chunks * chunk * vdim, 0.5);
    let kcd = draw(5, heads * chunks * chunk * kdim, 0.25);
    let state0 = draw(6, heads * kdim * vdim, 0.2);
    let mut g_cum = vec![0.0f32; heads * seq_pad];
    for h in 0..heads {
        for c in 0..chunks {
            let mut acc = 0.0f32;
            for i in 0..chunk {
                acc -= 0.01 * ((h + c + i) % 5) as f32;
                g_cum[h * seq_pad + c * chunk + i] = acc;
            }
        }
    }

    let upload = |data: &[f32]| -> CudaBuffer {
        let buf = CudaBuffer::alloc(data.len() * 4, 0).unwrap();
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
                buf.ptr(),
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * 4,
                crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))
            .unwrap();
        }
        buf
    };
    let (qb, kb, gb, tb, vtb, kcdb) = (
        upload(&q),
        upload(&k),
        upload(&g_cum),
        upload(&t),
        upload(&vt),
        upload(&kcd),
    );

    // The scan carries its state in the buffer it was given, so each run needs
    // its own copy of the initial state to start from.
    let run = |split: &str| -> (Vec<f32>, Vec<f32>) {
        std::env::set_var("APXINF_GDN_CHUNK_STATE_V_SPLIT", split);
        let sb = upload(&state0);
        let out = CudaBuffer::alloc(seq * heads * vdim * 2, 0).unwrap();
        let out_t = out
            .as_tensor(Shape::new(vec![seq, heads * vdim]), DType::BF16)
            .unwrap();
        crate::kernels::linear_attention::gdn_chunk_state(
            &ctx, &qb, &kb, &gb, &tb, &vtb, &kcdb, &sb, &out_t, seq_pad, heads, kdim, vdim, chunk,
        )
        .unwrap();
        let produced = download_bf16_as_fp32(&out_t).unwrap();
        let mut state = vec![0.0f32; heads * kdim * vdim];
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
                state.as_mut_ptr() as *mut std::ffi::c_void,
                sb.ptr(),
                state.len() * 4,
                crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))
            .unwrap();
        }
        (produced, state)
    };

    let (base_out, base_state) = run("1");
    for split in ["2", "4"] {
        let (split_out, split_state) = run(split);
        let out_diff = base_out
            .iter()
            .zip(&split_out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let state_diff = base_state
            .iter()
            .zip(&split_state)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "gdn_chunk_state_v_split split={split} out_words={} out_differing={out_diff} state_differing={state_diff}",
            base_out.len()
        );
        assert_eq!(out_diff, 0, "value split {split} changed the output");
        assert_eq!(
            state_diff, 0,
            "value split {split} changed the carried state"
        );
    }
    std::env::remove_var("APXINF_GDN_CHUNK_STATE_V_SPLIT");
}

// ── GDN chunk GEMM against an fp64 oracle ─────────────────────────
//
// Companion to the chunk-state one. Both products here are [C,C] @ [C,V] with
// the right operand already on the BF16 grid -- vb is bf16(v * beta), kb is
// bf16(bf16(k * beta) * exp2(g)) -- and both outputs are rounded to BF16 on
// the way out, so the reference performs those roundings too and what is
// measured is only what the multiply does.
//
//   cargo test --release -p apxinf-cuda gdn_chunk_gemm_error -- --nocapture
//   APXINF_GDN_CHUNK_STATE_WMMA=0 / lossy for the other two forms
#[test]
fn gdn_chunk_gemm_error_against_fp64_oracle() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kdim, vdim, chunk, chunks) = (4usize, 128usize, 128usize, 64usize, 3usize);
    let seq_pad = chunk * chunks;
    let draw = |salt: u64, n: usize, scale: f64| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                (((x & 0xFFFF) as f64 / 32768.0 - 1.0) * scale) as f32
            })
            .collect()
    };
    let a = draw(7, heads * chunks * chunk * chunk, 0.3);
    let v = draw(8, heads * seq_pad * vdim, 0.6);
    let k = draw(9, heads * seq_pad * kdim, 0.6);
    let beta = draw(10, heads * seq_pad, 0.5);
    let mut g_cum = vec![0.0f32; heads * seq_pad];
    for h in 0..heads {
        for c in 0..chunks {
            let mut acc = 0.0f32;
            for i in 0..chunk {
                acc -= 0.02 * ((h + c + i) % 4) as f32;
                g_cum[h * seq_pad + c * chunk + i] = acc;
            }
        }
    }
    let upload = |d: &[f32]| -> CudaBuffer {
        let b = CudaBuffer::alloc(d.len() * 4, 0).unwrap();
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
                b.ptr(),
                d.as_ptr() as *const std::ffi::c_void,
                d.len() * 4,
                crate::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))
            .unwrap();
        }
        b
    };
    let (ab, vb, kb, bb, gb) = (
        upload(&a),
        upload(&v),
        upload(&k),
        upload(&beta),
        upload(&g_cum),
    );
    let n_vt = heads * chunks * chunk * vdim;
    let n_kcd = heads * chunks * chunk * kdim;
    let vt = CudaBuffer::alloc(n_vt * 4, 0).unwrap();
    let kcd = CudaBuffer::alloc(n_kcd * 4, 0).unwrap();
    crate::kernels::linear_attention::gdn_chunk_gemm(
        &ctx, &ab, &vb, &kb, &bb, &gb, &vt, &kcd, seq_pad, heads, kdim, vdim, chunk,
    )
    .unwrap();
    let read = |b: &CudaBuffer, n: usize| -> Vec<f32> {
        crate::transfers::to_cpu(&b.as_tensor(Shape::new(vec![1, n]), DType::F32).unwrap())
            .unwrap()
            .to_f32_vec()
            .unwrap()
    };
    let got_vt = read(&vt, n_vt);
    let got_kcd = read(&kcd, n_kcd);

    let bf = |x: f64| -> f64 {
        let f = x as f32;
        let bits = f.to_bits();
        let r = ((bits >> 16)
            + (((bits >> 15) & 1) & ((bits & 0x7FFF != 0) as u32 | ((bits >> 16) & 1))))
            << 16;
        f32::from_bits(r) as f64
    };
    let mut sum_abs = 0.0f64;
    let mut sum_ref = 0.0f64;
    for h in 0..heads {
        for c in 0..chunks {
            let tok = h * seq_pad + c * chunk;
            let base = ((h * chunks) + c) * chunk;
            for i in 0..chunk {
                for j in 0..vdim {
                    let mut svt = 0.0f64;
                    let mut skcd = 0.0f64;
                    for m in 0..chunk {
                        let am = a[(base + i) * chunk + m] as f64;
                        let bm = beta[tok + m] as f64;
                        svt += am * bf(v[(tok + m) * vdim + j] as f64 * bm);
                        let kb0 = bf(k[(tok + m) * kdim + j] as f64 * bm);
                        skcd += am * bf(kb0 * (g_cum[tok + m] as f64).exp2());
                    }
                    for (expect, actual) in [
                        (bf(svt), got_vt[(base + i) * vdim + j] as f64),
                        (bf(skcd), got_kcd[(base + i) * kdim + j] as f64),
                    ] {
                        sum_abs += (actual - expect).abs();
                        sum_ref += expect.abs();
                    }
                }
            }
        }
    }
    println!(
        "gdn_chunk_gemm_oracle mode={} elements={} rel_l1={:.6e}",
        std::env::var("APXINF_GDN_CHUNK_STATE_WMMA").unwrap_or_else(|_| "default".into()),
        2 * n_vt,
        sum_abs / sum_ref
    );
    assert!(
        sum_abs / sum_ref < 0.05,
        "GDN chunk GEMM relative L1 {} against fp64",
        sum_abs / sum_ref
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

// ── Vision SDPA (non-causal full attention) ──────────────────────

#[test]
fn vision_sdpa_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq, n_heads, head_dim) = (6usize, 2usize, 64usize);
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

/// The fused contract must retain all unit-offset and BF16 rounding boundaries,
/// including the vector reduction used by planner-sized matrices.
#[test]
fn weighted_adaln_residual_fusion_matches_composition() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    for (rows, cols) in [
        (1, 7),
        (3, 129),
        (15, 256),
        (16, 256),
        (50, 1024),
        (2, 8192),
    ] {
        for amplitude in [0.0, 0.3, 10.0] {
            let matrix = |phase: f32| {
                let data: Vec<f32> = (0..rows * cols)
                    .map(|i| ((i as f32) * 0.13 + phase).sin() * amplitude)
                    .collect();
                upload_fp32_as_bf16(&ctx, &data, vec![rows, cols]).unwrap()
            };
            let vector = |phase: f32| {
                let data: Vec<f32> = (0..cols)
                    .map(|i| ((i as f32) * 0.17 + phase).cos())
                    .collect();
                upload_fp32_as_bf16(&ctx, &data, vec![cols]).unwrap()
            };
            let (projection, residual) = (matrix(0.1), matrix(0.9));
            let (gate, weight, scale, shift) = (vector(0.2), vector(0.7), vector(1.2), vector(1.7));
            let expected_hidden = crate::kernels::linear_attention::adaln_gate_residual(
                &ctx,
                &projection,
                &residual,
                &gate,
            )
            .unwrap();
            let expected_norm = crate::kernels::linear_attention::adaln_rms_norm(
                &ctx,
                &expected_hidden,
                &weight,
                &scale,
                &shift,
                1e-6,
            )
            .unwrap();
            let actual = crate::kernels::fused::adaln_gate_residual_rms_bf16(
                &ctx,
                &projection,
                &residual,
                &gate,
                &weight,
                &scale,
                &shift,
                1e-6,
            )
            .unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                download_bf16_as_fp32(&actual.hidden).unwrap(),
                download_bf16_as_fp32(&expected_hidden).unwrap(),
                "hidden {rows}x{cols}"
            );
            assert_eq!(
                download_bf16_as_fp32(&actual.normalized).unwrap(),
                download_bf16_as_fp32(&expected_norm).unwrap(),
                "norm {rows}x{cols}"
            );
        }
    }
}

// The causal conv stages a window of tokens in shared memory so each input is
// read once instead of kernel_size times. The shapes that can break that are
// the ones where the window is not a whole tile: a sequence that straddles the
// CONV_TOKENS boundary, one shorter than a tile, and the single-token decode
// that carries its left context entirely in the cached state. The reference
// below is the per-token form the kernel replaced, rounding where it rounded.
#[test]
fn causal_conv_tiling_matches_the_per_token_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let draw = |salt: u64, n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                half::bf16::from_f32(((x & 0xFFFF) as f32 / 32768.0 - 1.0) * scale).to_f32()
            })
            .collect()
    };
    // 70 straddles the 32-token tile, 5 is shorter than one, 1 is decode, and
    // 64 lands exactly on a tile edge. 300 channels crosses the 256-wide slab.
    for &seq in &[70usize, 5, 1, 64] {
        for &cached in &[false, true] {
            let (channels, kernel_size) = (300usize, 4usize);
            let x_stride = channels + 37; // the real caller passes a wider row
            let x = draw(1, seq * x_stride, 0.8);
            let weight = draw(2, channels * kernel_size, 0.5);
            let state = draw(3, channels * kernel_size, 0.7);

            let mut expected = vec![0.0f32; seq * channels];
            for token in 0..seq {
                for channel in 0..channels {
                    let mut acc = 0.0f32;
                    for i in 0..kernel_size {
                        let src = token as isize - (kernel_size as isize - 1) + i as isize;
                        let value = if src >= 0 {
                            x[src as usize * x_stride + channel]
                        } else if cached {
                            state[channel * kernel_size + (kernel_size as isize + src) as usize]
                        } else {
                            0.0
                        };
                        acc += value * weight[channel * kernel_size + i];
                    }
                    let conv = half::bf16::from_f32(acc).to_f32();
                    expected[token * channels + channel] =
                        half::bf16::from_f32(silu_ref(conv)).to_f32();
                }
            }

            let xt = upload_fp32_as_bf16(&ctx, &x, vec![seq, x_stride]).unwrap();
            let wt = upload_fp32_as_bf16(&ctx, &weight, vec![channels, kernel_size]).unwrap();
            let st = upload_fp32_as_bf16(&ctx, &state, vec![channels, kernel_size]).unwrap();
            let out =
                upload_fp32_as_bf16(&ctx, &vec![0.0; seq * channels], vec![seq, channels]).unwrap();
            let new_state = upload_fp32_as_bf16(
                &ctx,
                &vec![0.0; channels * kernel_size],
                vec![channels, kernel_size],
            )
            .unwrap();
            crate::kernels::linear_attention::causal_conv1d_silu_bf16(
                &ctx,
                &xt,
                &wt,
                if cached { Some(&st) } else { None },
                &out,
                &new_state,
                kernel_size,
            )
            .unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                download_bf16_as_fp32(&out).unwrap(),
                expected,
                "seq={seq} cached={cached}"
            );

            // The cache handoff: the last kernel_size pre-conv activations.
            let mut expected_state = vec![0.0f32; channels * kernel_size];
            for channel in 0..channels {
                for i in 0..kernel_size {
                    expected_state[channel * kernel_size + i] = if seq + i < kernel_size {
                        if cached {
                            state[channel * kernel_size + seq + i]
                        } else {
                            0.0
                        }
                    } else {
                        x[(seq + i - kernel_size) * x_stride + channel]
                    };
                }
            }
            assert_eq!(
                download_bf16_as_fp32(&new_state).unwrap(),
                expected_state,
                "new_state seq={seq} cached={cached}"
            );
        }
    }
}

// The chunk-state scan runs one BF16 pass over each product rather than the
// two the split form uses. That is only sound because every left operand it
// reads -- q, k, the chunk transition and the key-decay product -- is written
// through __float2bfloat16 by the kernel that produces it, so `x - bf16(x)` is
// zero and the split's low pass accumulates nothing.
//
// Nothing in the type system holds the producers to that. This does: feed the
// scan operands that are exactly BF16, run both forms, and require the outputs
// to agree bit for bit. If a producer starts emitting a value BF16 cannot hold,
// the two forms diverge here rather than silently in a trajectory.
//
//   cargo test --release -p apxinf-cuda chunk_state_split_is_dead -- --nocapture
#[test]
fn chunk_state_split_is_dead_when_operands_are_bf16() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (heads, kdim, vdim, chunk, chunks) = (2usize, 128usize, 128usize, 64usize, 2usize);
    let seq_pad = chunk * chunks;
    // Every value is round-tripped through BF16, exactly as the producers leave
    // them. g_cum stays fp32: the scan reads it as a scalar, not as an operand.
    let bf = |salt: u64, n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ salt;
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                x ^= x >> 32;
                half::bf16::from_f32(((x & 0xFFFF) as f32 / 32768.0 - 1.0) * scale).to_f32()
            })
            .collect()
    };
    let upload = |data: &[f32]| -> CudaBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
        buf.copy_from_host(&bytes).unwrap();
        buf
    };
    let q = upload(&bf(1, heads * seq_pad * kdim, 0.5));
    let k = upload(&bf(2, heads * seq_pad * kdim, 0.5));
    let t = upload(&bf(3, heads * chunks * chunk * chunk, 0.3));
    let vt = upload(&bf(4, heads * seq_pad * vdim, 0.6));
    let kcd = upload(&bf(5, heads * seq_pad * kdim, 0.4));
    let mut g = vec![0.0f32; heads * seq_pad];
    for (i, value) in g.iter_mut().enumerate() {
        *value = -((i % chunk) as f32) * 0.01;
    }
    let g_cum = upload(&g);

    let run = |mode: &str| -> Vec<f32> {
        // SAFETY: single-threaded test; the policy reads this at launch.
        unsafe { std::env::set_var("APXINF_GDN_CHUNK_STATE_WMMA", mode) };
        let state = upload(&vec![0.0f32; heads * kdim * vdim]);
        let out = upload_fp32_as_bf16(
            &ctx,
            &vec![0.0; seq_pad * heads * vdim],
            vec![seq_pad, heads * vdim],
        )
        .unwrap();
        crate::kernels::linear_attention::gdn_chunk_state(
            &ctx, &q, &k, &g_cum, &t, &vt, &kcd, &state, &out, seq_pad, heads, kdim, vdim, chunk,
        )
        .unwrap();
        ctx.synchronize().unwrap();
        download_bf16_as_fp32(&out).unwrap()
    };
    let split = run("2");
    let single = run("1");
    unsafe { std::env::remove_var("APXINF_GDN_CHUNK_STATE_WMMA") };
    assert_eq!(
        split, single,
        "a producer is emitting a value BF16 cannot hold; the chunk-state scan's \
         single-pass default is no longer exact"
    );
}

// Is the vendor's own pick the best one available?
//
// With no tuned tactic the BF16 path falls through to cublasGemmEx and takes
// whatever heuristic cuBLAS chose. cuBLASLt exposes its candidate list ranked,
// and rank 0 is not always the fastest -- the ordering is a prediction. This
// walks the ranks for the shapes qwen_drive actually issues and prints what the
// vendor default costs beside the best rank found, so the gap is a measurement
// rather than an assumption.
//
// Note: as written this reports nothing. Every rank returns a non-zero
// cuBLASLt status outside the normal call path -- the plan wants the context
// state that `gemm::bf16` sets up around it, which a raw FFI call does not
// carry. The supported way to get this comparison is the tuner itself, whose
// report records vendor against winner per shape with an L2 evictor between
// candidates; run it with APXINF_QWEN_GRAPH unset and `autotune=True` now that
// the direct runner traverses once eagerly first. Kept for the record.
//
//   cargo test --release -p apxinf-cuda vendor_gemm_heuristic_sweep -- --nocapture --ignored
#[test]
#[ignore = "superseded by the tuner's own report; see the note above"]
fn vendor_gemm_heuristic_sweep() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Dumped from a direct-planning request via APXINF_GEMM_DUMP_SHAPES.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (3385, 18432, 2560, "L ffn gate/up"),
        (3385, 12352, 2560, "L gdn zba"),
        (3385, 10240, 2560, "L (10240)"),
        (3385, 2560, 9216, "L ffn down"),
        (3385, 2560, 4096, "L gdn out"),
        (12216, 1024, 1536, "V patch/qkv"),
        (50, 10240, 1024, "A ffn gate/up"),
        (50, 7168, 1024, "A qkv"),
        (50, 1024, 4096, "A ffn down"),
        (50, 1024, 3584, "A out"),
    ];
    let iters = 20;
    println!(
        "{:<16}{:>7}{:>7}{:>7}{:>11}{:>11}{:>7}{:>8}",
        "op", "m", "n", "k", "vendor ms", "best ms", "rank", "gain"
    );
    let mut total_vendor = 0.0f64;
    let mut total_best = 0.0f64;
    for &(m, n, k, label) in shapes {
        let a = CudaBuffer::alloc(m * k * 2, ctx.device_id()).unwrap();
        let b = CudaBuffer::alloc(k * n * 2, ctx.device_id()).unwrap();
        let c = CudaBuffer::alloc(m * n * 2, ctx.device_id()).unwrap();
        let time = |run: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                run();
            }
            ctx.synchronize().unwrap();
            let start = std::time::Instant::now();
            for _ in 0..iters {
                run();
            }
            ctx.synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1000.0 / iters as f64
        };
        let vendor = time(&|| {
            ctx.cublas()
                .gemm(DType::BF16, m, n, k, 1.0, &a, &b, 0.0, &c)
                .unwrap();
        });
        let lt_status = || unsafe {
            crate::ffi::apxinf_static_bf16_gemm(
                a.ptr(),
                b.ptr(),
                c.ptr(),
                m as i32,
                n as i32,
                k as i32,
                1.0,
                ctx.stream().handle(),
            )
        };
        let lt = || {
            let status = lt_status();
            assert_eq!(status, 0, "cuBLASLt BF16 gemm returned {status}");
        };
        let (mut best, mut best_rank) = (f64::INFINITY, -1i32);
        for rank in 0..12i32 {
            // A rank the library does not offer for this shape is skipped.
            if unsafe {
                crate::ffi::apxinf_static_set_cublaslt_bf16_gemm_heuristic(
                    m as i32, n as i32, k as i32, rank,
                )
            } != 0
            {
                continue;
            }
            if lt_status() != 0 {
                continue;
            }
            let ms = time(&lt);
            if ms < best {
                best = ms;
                best_rank = rank;
            }
        }
        total_vendor += vendor;
        total_best += best.min(vendor);
        println!(
            "{label:<16}{m:>7}{n:>7}{k:>7}{vendor:>11.4}{best:>11.4}{best_rank:>7}{:>7.1}%",
            (vendor - best) / vendor * 100.0
        );
    }
    println!(
        "\nsum over one call each: vendor {total_vendor:.3} ms, best-of {total_best:.3} ms, \
         {:.1}% off the vendor pick",
        (total_vendor - total_best) / total_vendor * 100.0
    );
}
