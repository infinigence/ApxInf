use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::activation::{gelu_tanh, silu};
use crate::kernels::attention::{causal_mask, softmax, softmax_causal, vision};
use crate::kernels::cache::append;
use crate::kernels::elementwise::{add, add_bias, mul, scale};
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
#[ignore = "requires cuDNN v9 on the loader path and a CUDA GPU"]
fn cudnn_3d_convolution_matches_scalar_volume() {
    use crate::kernels::convolution::{conv3d, Conv3dSpec};
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (ci, co, d, h, w) = (8, 16, 3, 4, 5);
    let x = (0..ci * d * h * w)
        .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
        .collect::<Vec<_>>();
    let weight = (0..co * ci * 8)
        .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
        .collect::<Vec<_>>();
    let bias = (0..co).map(|i| i as f32 / 16.0).collect::<Vec<_>>();
    let mut expected = Vec::new();
    for oc in 0..co {
        for z in 0..d - 1 {
            for y in 0..h - 1 {
                for xx in 0..w - 1 {
                    let mut sum = 0.0f32;
                    for ic in 0..ci {
                        for kz in 0..2 {
                            for ky in 0..2 {
                                for kx in 0..2 {
                                    let xi = ((ic * d + z + kz) * h + y + ky) * w + xx + kx;
                                    let wi = (((oc * ci + ic) * 2 + kz) * 2 + ky) * 2 + kx;
                                    sum += x[xi] * weight[wi];
                                }
                            }
                        }
                    }
                    expected.push(
                        half::bf16::from_f32(half::bf16::from_f32(sum).to_f32() + bias[oc])
                            .to_f32(),
                    );
                }
            }
        }
    }
    let xt = upload_fp32_as_bf16(&ctx, &x, vec![1, ci, d, h, w]).unwrap();
    let wt = upload_fp32_as_bf16(&ctx, &weight, vec![co, ci, 2, 2, 2]).unwrap();
    let bt = upload_fp32_as_bf16(&ctx, &bias, vec![co]).unwrap();
    let output = conv3d(&ctx, &xt, &wt, Some(&bt), Conv3dSpec::default()).unwrap();
    assert_eq!(output.shape().dims(), [1, co, d - 1, h - 1, w - 1]);
    assert_eq!(download_bf16_as_fp32(&output).unwrap(), expected);
}

#[test]
fn perception_feature_layout_preserves_batches() {
    use crate::kernels::{activation, elementwise, pooling};
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let a = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0];
    let b = [10.0, 11.0, -10.0, -11.0];
    let a = upload_fp32_as_bf16(&ctx, &a, vec![2, 3, 1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &b, vec![2, 2, 1, 1]).unwrap();
    let a = elementwise::expand_spatial_bf16(&ctx, &a, 2, 3).unwrap();
    let b = elementwise::expand_spatial_bf16(&ctx, &b, 2, 3).unwrap();
    let joined = elementwise::concat_channels_bf16(&ctx, &[&a, &b]).unwrap();
    let channels = [-3.0, -2.0, -1.0, 10.0, 11.0, 0.0, 1.0, 2.0, -10.0, -11.0];
    let expected = channels
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(6))
        .collect::<Vec<_>>();
    assert_eq!(download_bf16_as_fp32(&joined).unwrap(), expected);
    let pooled = pooling::global_mean_bf16(&ctx, &joined).unwrap();
    assert_eq!(download_bf16_as_fp32(&pooled).unwrap(), channels);
    let relu = activation::relu_bf16(&ctx, &joined).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&relu).unwrap(),
        expected.iter().map(|&v| v.max(0.0)).collect::<Vec<_>>()
    );
}

#[test]
fn group_norm_materializes_bf16_statistics() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let x = [
        1.0, 2.0, 4.0, 1.0, 2.0, 4.0, -1.0, -2.0, -4.0, -1.0, -2.0, -4.0,
    ];
    let gamma = [0.5, 1.0, 1.5, 2.0];
    let beta = [0.0, 1.0, -1.0, 0.0];
    let xt = upload_fp32_as_bf16(&ctx, &x, vec![1, 4, 1, 3]).unwrap();
    let wt = upload_fp32_as_bf16(&ctx, &gamma, vec![4]).unwrap();
    let bt = upload_fp32_as_bf16(&ctx, &beta, vec![4]).unwrap();
    let output = crate::kernels::norm::group_bf16_rounded(&ctx, &xt, &wt, &bt, 2, 1e-5).unwrap();
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    let inv = bf((14.0f32 / 9.0 + bf(1e-5)).sqrt().recip());
    let expected = x
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let c = i / 3;
            let mean = bf(if c < 2 { 7.0 / 3.0 } else { -7.0 / 3.0 });
            let scale = inv * gamma[c];
            bf(scale.mul_add(value, (-mean).mul_add(scale, beta[c])))
        })
        .collect::<Vec<_>>();
    assert_eq!(download_bf16_as_fp32(&output).unwrap(), expected);
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
#[ignore = "requires cuDNN v9 on the dynamic-loader path and a CUDA GPU"]
fn cudnn_convolution_matches_scalar_cross_correlation() {
    use crate::kernels::convolution::{conv2d, conv_transpose2d, Conv2dSpec};
    let ctx = CudaContext::new(0).expect("CUDA device required");
    for transpose in [false, true] {
        let (channels, outputs, height, width, kernel) = (8, 16, 3, 5, 2);
        let stride = if transpose { 2 } else { 1 };
        let oh = if transpose { height * 2 } else { height - 1 };
        let ow = if transpose { width * 2 } else { width - 1 };
        let x: Vec<f32> = (0..channels * height * width)
            .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
            .collect();
        let w: Vec<f32> = (0..channels * outputs * kernel * kernel)
            .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
            .collect();
        let bias: Vec<f32> = (0..outputs).map(|i| i as f32 / 16.0).collect();
        let mut expected = vec![0.0f32; outputs * oh * ow];
        for co in 0..outputs {
            for ci in 0..channels {
                for iy in 0..height {
                    for ix in 0..width {
                        for ky in 0..kernel {
                            for kx in 0..kernel {
                                let (oy, ox, wi) = if transpose {
                                    (
                                        iy * stride + ky,
                                        ix * stride + kx,
                                        ((ci * outputs + co) * kernel + ky) * kernel + kx,
                                    )
                                } else {
                                    if iy < ky || ix < kx {
                                        continue;
                                    }
                                    (
                                        iy - ky,
                                        ix - kx,
                                        ((co * channels + ci) * kernel + ky) * kernel + kx,
                                    )
                                };
                                if oy < oh && ox < ow {
                                    expected[(co * oh + oy) * ow + ox] +=
                                        x[(ci * height + iy) * width + ix] * w[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
        for (i, value) in expected.iter_mut().enumerate() {
            *value =
                half::bf16::from_f32(half::bf16::from_f32(*value).to_f32() + bias[i / (oh * ow)])
                    .to_f32();
        }
        let xt = upload_fp32_as_bf16(&ctx, &x, vec![1, channels, height, width]).unwrap();
        let filter_shape = if transpose {
            vec![channels, outputs, kernel, kernel]
        } else {
            vec![outputs, channels, kernel, kernel]
        };
        let wt = upload_fp32_as_bf16(&ctx, &w, filter_shape).unwrap();
        let bt = upload_fp32_as_bf16(&ctx, &bias, vec![outputs]).unwrap();
        let spec = Conv2dSpec {
            stride: [stride, stride],
            ..Default::default()
        };
        let y = if transpose {
            conv_transpose2d(&ctx, &xt, &wt, Some(&bt), spec)
        } else {
            conv2d(&ctx, &xt, &wt, Some(&bt), spec)
        }
        .unwrap();
        assert_eq!(y.shape().dims(), [1, outputs, oh, ow]);
        assert_eq!(download_bf16_as_fp32(&y).unwrap(), expected);
    }
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
