//! CPU backend implementation.

use crate::kv_cache::{CpuKVCache, KvCache};
use crate::sampling::{CpuNormalGenerator, CpuTokenSampler};
use crate::{
    Backend, Device, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor, TokenSampler,
    TokenSamplingSpec,
};

/// CPU backend — all ops execute synchronously on the host.
pub struct CpuBackend;

impl SamplingBackend for CpuBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(CpuTokenSampler::new(spec)?))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Ok(Box::new(CpuNormalGenerator::new(output)?))
    }
}

impl Backend for CpuBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let w = weight.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = dims[0];
        let hidden = dims[1];

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let row_start = s * hidden;
            let row = &data[row_start..row_start + hidden];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let rms = (mean_sq + eps).sqrt();
            for (i, &val) in row.iter().enumerate() {
                out[row_start + i] = (val / rms) * w[i];
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Tensor::from_f32(x.shape().dims().to_vec(), &out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v * factor).collect();
        Tensor::from_f32(input.shape().dims().to_vec(), &out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul_cpu(b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        let half_dim = head_dim / 2;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let pos = pos_offset as usize + s;
            for h in 0..n_heads {
                let base = s * n_heads * head_dim + h * head_dim;
                for i in 0..half_dim {
                    let angle = pos as f32 * freqs[i];
                    let cos_v = angle.cos();
                    let sin_v = angle.sin();
                    let x1 = data[base + i];
                    let x2 = data[base + half_dim + i];
                    out[base + i] = x1 * cos_v - x2 * sin_v;
                    out[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
                }
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn rope_tmrope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = dims[0];
        if pos_ids.len() != seq_len * 3 || sections.iter().sum::<usize>() != head_dim / 2 {
            return Err(Error::Other(
                "rope_tmrope: invalid positions or sections".into(),
            ));
        }
        let half = head_dim / 2;
        let first_boundary = sections[0] * 2;
        let second_boundary = first_boundary + sections[1] * 2;
        let mut output = vec![0.0_f32; data.len()];
        for seq in 0..seq_len {
            for head in 0..n_heads {
                let base = seq * n_heads * head_dim + head * head_dim;
                for dimension in 0..head_dim {
                    let axis = if dimension < first_boundary {
                        0
                    } else if dimension < second_boundary {
                        1
                    } else {
                        2
                    };
                    let pair = dimension % half;
                    let frequency = 1.0 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                    let angle = pos_ids[seq * 3 + axis] as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let rotated = if dimension < half {
                        -data[base + dimension + half]
                    } else {
                        data[base + dimension - half]
                    };
                    output[base + dimension] = data[base + dimension] * cos + rotated * sin;
                }
            }
        }
        Tensor::from_f32(dims.to_vec(), &output)
    }

    fn grouped_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        group_ids: &[u32],
    ) -> Result<Tensor> {
        if group_ids.len() != seq_len {
            return Err(Error::Other(format!(
                "grouped_sdpa: {} group ids for {seq_len} tokens",
                group_ids.len()
            )));
        }
        grouped_attention_cpu(q, k, v, seq_len, n_heads, head_dim, group_ids)
    }

    fn im2col1d(
        &self,
        input: &Tensor,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        if dims.len() != 2 || kernel == 0 || stride == 0 {
            return Err(Error::Other(
                "im2col1d: expected rank-2 input and positive kernel/stride".into(),
            ));
        }
        let (frames, channels) = (dims[0], dims[1]);
        let padded = frames
            .checked_add(2 * padding)
            .ok_or_else(|| Error::Other("im2col1d: padded length overflow".into()))?;
        if padded < kernel {
            return Err(Error::Other("im2col1d: kernel exceeds padded input".into()));
        }
        let output_frames = (padded - kernel) / stride + 1;
        let mut output = vec![0.0_f32; output_frames * channels * kernel];
        for out_frame in 0..output_frames {
            for channel in 0..channels {
                for tap in 0..kernel {
                    let padded_frame = out_frame * stride + tap;
                    if let Some(frame) = padded_frame
                        .checked_sub(padding)
                        .filter(|frame| *frame < frames)
                    {
                        output[out_frame * channels * kernel + channel * kernel + tap] =
                            data[frame * channels + channel];
                    }
                }
            }
        }
        Tensor::from_f32(vec![output_frames, channels * kernel], &output)
    }

    fn avg_pool1d(&self, input: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        if dims.len() != 2 || kernel == 0 || stride == 0 || dims[0] < kernel {
            return Err(Error::Other(
                "avg_pool1d: invalid input or parameters".into(),
            ));
        }
        let (frames, channels) = (dims[0], dims[1]);
        let output_frames = (frames - kernel) / stride + 1;
        let mut output = vec![0.0_f32; output_frames * channels];
        for out_frame in 0..output_frames {
            for channel in 0..channels {
                let sum = (0..kernel)
                    .map(|tap| data[(out_frame * stride + tap) * channels + channel])
                    .sum::<f32>();
                output[out_frame * channels + channel] = sum / kernel as f32;
            }
        }
        Tensor::from_f32(vec![output_frames, channels], &output)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let table_data = table.as_f32()?;
        let embed_dim = table.shape().dims()[1];
        let seq_len = ids.len();

        let mut out = vec![0.0f32; seq_len * embed_dim];
        for (i, &tid) in ids.iter().enumerate() {
            let src_offset = tid as usize * embed_dim;
            let dst_offset = i * embed_dim;
            out[dst_offset..dst_offset + embed_dim]
                .copy_from_slice(&table_data[src_offset..src_offset + embed_dim]);
        }
        Tensor::from_f32(vec![seq_len, embed_dim], &out)
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let cache = kv
            .as_any_mut()
            .downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h * n_kv_heads / n_heads;
            let mut scores = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                for d in 0..head_dim {
                    scores[t] += q_data[h * head_dim + d] * k_cached[kv_h][t][d];
                }
                scores[t] *= scale;
            }
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..kv_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }
            for t in 0..kv_len {
                for d in 0..head_dim {
                    output[h * head_dim + d] += scores[t] * v_cached[kv_h][t][d];
                }
            }
        }
        Tensor::from_f32(vec![1, n_heads * head_dim], &output)
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let seq_len = q.shape().dims()[0];
        let cache = kv
            .as_any_mut()
            .downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * n_heads * head_dim];

        for s in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h * n_kv_heads / n_heads;
                let valid_len = kv_len.min(s + 1 + kv_len - seq_len);
                let mut scores = vec![0.0f32; kv_len];
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        scores[t] += q_data[s * n_heads * head_dim + h * head_dim + d]
                            * k_cached[kv_h][t][d];
                    }
                    scores[t] *= scale;
                }
                let max_score = scores[..valid_len]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores[..valid_len]
                    .iter()
                    .map(|&s| (s - max_score).exp())
                    .sum();
                for t in 0..valid_len {
                    scores[t] = (scores[t] - max_score).exp() / exp_sum;
                }
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        output[s * n_heads * head_dim + h * head_dim + d] +=
                            scores[t] * v_cached[kv_h][t][d];
                    }
                }
            }
        }
        Tensor::from_f32(vec![seq_len, n_heads * head_dim], &output)
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(CpuKVCache::new(n_layers, n_kv_heads, head_dim, max_seq_len))
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        kv.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn begin_capture(&self) -> Result<()> {
        Ok(())
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(NoopGraph))
    }

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// No-op graph (CPU has nothing to capture).
struct NoopGraph;

impl Graph for NoopGraph {
    fn replay(&self) -> Result<()> {
        Ok(())
    }
}

fn grouped_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    group_ids: &[u32],
) -> Result<Tensor> {
    let q = q.as_f32()?;
    let k = k.as_f32()?;
    let v = v.as_f32()?;
    let expected = seq_len * n_heads * head_dim;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(Error::Other("non-causal attention: shape mismatch".into()));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; expected];
    for query in 0..seq_len {
        for head in 0..n_heads {
            let query_offset = (query * n_heads + head) * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for key in 0..seq_len {
                if group_ids[query] != group_ids[key] {
                    continue;
                }
                let key_offset = (key * n_heads + head) * head_dim;
                scores[key] = (0..head_dim)
                    .map(|dim| q[query_offset + dim] * k[key_offset + dim])
                    .sum::<f32>()
                    * scale;
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .filter(|score| score.is_finite())
                .map(|score| (score - maximum).exp())
                .sum::<f32>();
            for key in 0..seq_len {
                if !scores[key].is_finite() {
                    continue;
                }
                let probability = (scores[key] - maximum).exp() / denominator;
                let value_offset = (key * n_heads + head) * head_dim;
                for dim in 0..head_dim {
                    output[query_offset + dim] += probability * v[value_offset + dim];
                }
            }
        }
    }
    Tensor::from_f32(vec![seq_len, n_heads * head_dim], &output)
}

#[cfg(test)]
mod omni_operator_tests {
    use super::*;

    #[test]
    fn im2col_and_average_pool_match_small_reference() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![3, 1], &[1.0, 2.0, 3.0]).unwrap();
        let columns = backend.im2col1d(&input, 3, 1, 1).unwrap();
        assert_eq!(columns.shape().dims(), [3, 3]);
        assert_eq!(
            columns.to_f32_vec().unwrap(),
            vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0]
        );
        let pooled = backend.avg_pool1d(&input, 2, 1).unwrap();
        assert_eq!(pooled.to_f32_vec().unwrap(), vec![1.5, 2.5]);
    }

    #[test]
    fn grouped_attention_cannot_cross_group_boundary() {
        let backend = CpuBackend;
        let q = Tensor::from_f32(vec![2, 1, 1], &[1.0, 1.0]).unwrap();
        let k = Tensor::from_f32(vec![2, 1, 1], &[1.0, 1.0]).unwrap();
        let v = Tensor::from_f32(vec![2, 1, 1], &[2.0, 8.0]).unwrap();
        let isolated = backend.grouped_sdpa(&q, &k, &v, 2, 1, 1, &[0, 1]).unwrap();
        assert_eq!(isolated.to_f32_vec().unwrap(), vec![2.0, 8.0]);
    }

    #[test]
    fn tmrope_uses_contiguous_full_dimension_axes() {
        let backend = CpuBackend;
        let input = Tensor::from_f32(vec![1, 1, 6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = backend
            .rope_tmrope(&input, 1, 6, 10_000.0, [1, 1, 1], &[0, 0, 1])
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(&output[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_ne!(&output[4..], &[5.0, 6.0]);
    }
}
