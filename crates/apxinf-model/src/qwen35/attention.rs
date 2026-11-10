use super::PackedLinearLayout;
use thiserror::Error;

pub(crate) const QWEN35_ROPE_THETA: f32 = 10_000_000.0;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AttentionError {
    #[error("{name} dimensions are inconsistent")]
    Shape { name: &'static str },
    #[error("token id {token_id} is outside embedding vocabulary {vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    #[error("{name} contains a non-finite value")]
    NonFinite { name: &'static str },
}

#[derive(Debug, Clone)]
pub struct PackedLinearReference {
    pub layout: PackedLinearLayout,
    pub weight_packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub zero_points: Vec<u32>,
}

impl PackedLinearReference {
    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, AttentionError> {
        self.layout
            .matvec_f32(&self.weight_packed, &self.scales, &self.zero_points, input)
            .map_err(|_| AttentionError::Shape {
                name: "packed projection",
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FullAttentionReferenceConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_query_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct FullAttentionReferenceLayer {
    pub config: FullAttentionReferenceConfig,
    pub input_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub q_proj: PackedLinearReference,
    pub k_proj: PackedLinearReference,
    pub v_proj: PackedLinearReference,
    pub o_proj: PackedLinearReference,
    pub gate_proj: PackedLinearReference,
    pub up_proj: PackedLinearReference,
    pub down_proj: PackedLinearReference,
}

impl FullAttentionReferenceLayer {
    pub fn prefill(
        &self,
        hidden: &[f32],
        seq_len: usize,
        position: usize,
        cache: &mut FullAttentionKv,
    ) -> Result<Vec<f32>, AttentionError> {
        let c = self.config;
        if seq_len != 1 || hidden.len() != c.hidden_size {
            return Err(AttentionError::Shape {
                name: "full-attention input",
            });
        }
        let normalized = rms_norm_f32(hidden, c.hidden_size, &self.input_norm, c.rms_epsilon)?;
        let qkv_qgate = self.q_proj.apply(&normalized)?;
        let (mut q, gate) = split_q_gate_f32(&qkv_qgate, c.n_query_heads, c.head_dim)?;
        let mut k = self.k_proj.apply(&normalized)?;
        let v = self.v_proj.apply(&normalized)?;
        q = qk_norm_f32(
            &q,
            1,
            c.n_query_heads,
            c.head_dim,
            &self.q_norm,
            c.rms_epsilon,
        )?;
        k = qk_norm_f32(&k, 1, c.n_kv_heads, c.head_dim, &self.k_norm, c.rms_epsilon)?;
        q = apply_partial_rope_f32(
            &q,
            1,
            c.n_query_heads,
            c.head_dim,
            c.rotary_dim,
            c.rope_theta,
            position,
        )?;
        k = apply_partial_rope_f32(
            &k,
            1,
            c.n_kv_heads,
            c.head_dim,
            c.rotary_dim,
            c.rope_theta,
            position,
        )?;
        cache.append(&k, &v, 1)?;
        let attended = causal_gqa_decode_f32(&q, cache, c.n_query_heads, c.head_dim)?;
        let gated = apply_output_gate_f32(&attended, &gate)?;
        let attention_update = self.o_proj.apply(&gated)?;
        let residual = residual_add_f32(hidden, &attention_update)?;
        let mlp_input = rms_norm_f32(
            &residual,
            c.hidden_size,
            &self.post_attention_norm,
            c.rms_epsilon,
        )?;
        let mlp = residual_add_f32(
            &residual,
            &self.down_proj.apply(&swiglu_f32(
                &self.gate_proj.apply(&mlp_input)?,
                &self.up_proj.apply(&mlp_input)?,
            )?)?,
        )?;
        Ok(mlp)
    }
}

pub fn embedding_f32(
    table: &[f32],
    vocab_size: usize,
    hidden_size: usize,
    token_ids: &[u32],
) -> Result<Vec<f32>, AttentionError> {
    if vocab_size == 0
        || hidden_size == 0
        || table.len() != vocab_size.checked_mul(hidden_size).unwrap_or(usize::MAX)
    {
        return Err(AttentionError::Shape { name: "embedding" });
    }
    let mut output = Vec::with_capacity(token_ids.len() * hidden_size);
    for &token_id in token_ids {
        let row = usize::try_from(token_id).map_err(|_| AttentionError::TokenOutOfRange {
            token_id,
            vocab_size,
        })?;
        if row >= vocab_size {
            return Err(AttentionError::TokenOutOfRange {
                token_id,
                vocab_size,
            });
        }
        output.extend_from_slice(&table[row * hidden_size..(row + 1) * hidden_size]);
    }
    require_finite("embedding output", &output)?;
    Ok(output)
}

pub fn rms_norm_f32(
    input: &[f32],
    hidden_size: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, AttentionError> {
    if hidden_size == 0
        || input.len() % hidden_size != 0
        || weight.len() != hidden_size
        || !epsilon.is_finite()
        || epsilon < 0.0
    {
        return Err(AttentionError::Shape { name: "RMSNorm" });
    }
    require_finite("RMSNorm input", input)?;
    require_finite("RMSNorm weight", weight)?;
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(hidden_size) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32;
        let denominator = (mean_square + epsilon).sqrt();
        if denominator == 0.0 {
            output.resize(output.len() + hidden_size, 0.0);
            continue;
        }
        let inverse_rms = denominator.recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(value, scale)| value * inverse_rms * (1.0 + scale)),
        );
    }
    require_finite("RMSNorm output", &output)?;
    Ok(output)
}

/// Split Qwen3.5's fused attention projection. The checkpoint stores each
/// query head as `[q_head, gate_head]`, so the gate is not a contiguous global
/// half of the projection output.
pub fn split_q_gate_f32(
    q_gate: &[f32],
    n_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), AttentionError> {
    if n_heads == 0
        || head_dim == 0
        || q_gate.len() != n_heads.saturating_mul(head_dim).saturating_mul(2)
    {
        return Err(AttentionError::Shape {
            name: "Qwen3.5 q/gate projection",
        });
    }
    require_finite("Qwen3.5 q/gate projection", q_gate)?;
    let mut query = Vec::with_capacity(n_heads * head_dim);
    let mut gate = Vec::with_capacity(n_heads * head_dim);
    for head in 0..n_heads {
        let base = head * head_dim * 2;
        query.extend_from_slice(&q_gate[base..base + head_dim]);
        gate.extend_from_slice(&q_gate[base + head_dim..base + 2 * head_dim]);
    }
    Ok((query, gate))
}

pub fn apply_partial_rope_f32(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    pos_offset: usize,
) -> Result<Vec<f32>, AttentionError> {
    if seq_len == 0
        || n_heads == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
        || input.len() != seq_len * n_heads * head_dim
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(AttentionError::Shape {
            name: "partial RoPE",
        });
    }
    require_finite("partial RoPE input", input)?;
    let mut output = input.to_vec();
    for position in 0..seq_len {
        for head in 0..n_heads {
            let base = (position * n_heads + head) * head_dim;
            for pair in 0..rotary_dim / 2 {
                let angle = (pos_offset + position) as f32
                    * theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let x = input[base + pair];
                let y = input[base + rotary_dim / 2 + pair];
                output[base + pair] = x * cos - y * sin;
                output[base + rotary_dim / 2 + pair] = x * sin + y * cos;
            }
        }
    }
    require_finite("partial RoPE output", &output)?;
    Ok(output)
}

pub fn gqa_expand_f32(
    query: &[f32],
    kv: &[f32],
    seq_len: usize,
    n_query_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    if seq_len == 0
        || n_query_heads == 0
        || n_kv_heads == 0
        || head_dim == 0
        || n_query_heads % n_kv_heads != 0
        || query.len() != seq_len * n_query_heads * head_dim
        || kv.len() != seq_len * n_kv_heads * head_dim
    {
        return Err(AttentionError::Shape { name: "GQA" });
    }
    require_finite("GQA query", query)?;
    require_finite("GQA key/value", kv)?;
    let ratio = n_query_heads / n_kv_heads;
    let mut output = vec![0.0; query.len()];
    for position in 0..seq_len {
        for q_head in 0..n_query_heads {
            let kv_head = q_head / ratio;
            let src = (position * n_kv_heads + kv_head) * head_dim;
            let dst = (position * n_query_heads + q_head) * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&kv[src..src + head_dim]);
        }
    }
    Ok(output)
}

pub fn qk_norm_f32(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, AttentionError> {
    if input.len() != checked_elements(&[seq_len, n_heads, head_dim], "Q/K norm")? {
        return Err(AttentionError::Shape { name: "Q/K norm" });
    }
    rms_norm_f32(input, head_dim, weight, epsilon)
}

pub fn causal_gqa_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    n_query_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    let query_len = checked_elements(&[seq_len, n_query_heads, head_dim], "attention query")?;
    let kv_len = checked_elements(&[seq_len, n_kv_heads, head_dim], "attention KV")?;
    if seq_len == 0
        || n_query_heads == 0
        || n_kv_heads == 0
        || head_dim == 0
        || n_query_heads % n_kv_heads != 0
        || query.len() != query_len
        || key.len() != kv_len
        || value.len() != kv_len
    {
        return Err(AttentionError::Shape {
            name: "causal GQA attention",
        });
    }
    require_finite("attention query", query)?;
    require_finite("attention key", key)?;
    require_finite("attention value", value)?;
    let ratio = n_query_heads / n_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; query_len];
    for position in 0..seq_len {
        for q_head in 0..n_query_heads {
            let kv_head = q_head / ratio;
            let q_base = (position * n_query_heads + q_head) * head_dim;
            let mut scores = Vec::with_capacity(position + 1);
            for source in 0..=position {
                let k_base = (source * n_kv_heads + kv_head) * head_dim;
                let score = query[q_base..q_base + head_dim]
                    .iter()
                    .zip(&key[k_base..k_base + head_dim])
                    .map(|(q, k)| q * k)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter_mut()
                .map(|score| {
                    *score = (*score - maximum).exp();
                    *score
                })
                .sum::<f32>();
            let out_base = (position * n_query_heads + q_head) * head_dim;
            for (source, score) in scores.into_iter().enumerate() {
                let v_base = (source * n_kv_heads + kv_head) * head_dim;
                let probability = score / denominator;
                for dimension in 0..head_dim {
                    output[out_base + dimension] += probability * value[v_base + dimension];
                }
            }
        }
    }
    require_finite("attention output", &output)?;
    Ok(output)
}

pub fn causal_gqa_decode_f32(
    query: &[f32],
    cache: &FullAttentionKv,
    n_query_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AttentionError> {
    if query.len() != n_query_heads * head_dim
        || n_query_heads == 0
        || head_dim == 0
        || cache.is_empty()
        || n_query_heads % cache.n_kv_heads != 0
    {
        return Err(AttentionError::Shape {
            name: "causal GQA decode",
        });
    }
    let ratio = n_query_heads / cache.n_kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; query.len()];
    for q_head in 0..n_query_heads {
        let kv_head = q_head / ratio;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache.len);
        for position in 0..cache.len {
            let k_base = (position * cache.n_kv_heads + kv_head) * cache.head_dim;
            scores.push(
                query[q_base..q_base + head_dim]
                    .iter()
                    .zip(&cache.keys[k_base..k_base + head_dim])
                    .map(|(q, k)| q * k)
                    .sum::<f32>()
                    * scale,
            );
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f32>();
        for (position, score) in scores.into_iter().enumerate() {
            let v_base = (position * cache.n_kv_heads + kv_head) * cache.head_dim;
            for dimension in 0..head_dim {
                output[q_base + dimension] +=
                    score / denominator * cache.values[v_base + dimension];
            }
        }
    }
    require_finite("attention decode output", &output)?;
    Ok(output)
}

pub fn apply_output_gate_f32(attention: &[f32], gate: &[f32]) -> Result<Vec<f32>, AttentionError> {
    if attention.len() != gate.len() {
        return Err(AttentionError::Shape {
            name: "attention output gate",
        });
    }
    require_finite("attention output", attention)?;
    require_finite("attention gate", gate)?;
    let output = attention
        .iter()
        .zip(gate)
        .map(|(value, gate)| value / (1.0 + (-gate).exp()))
        .collect::<Vec<_>>();
    require_finite("gated attention output", &output)?;
    Ok(output)
}

pub fn swiglu_f32(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, AttentionError> {
    if gate.len() != up.len() {
        return Err(AttentionError::Shape { name: "SwiGLU" });
    }
    require_finite("SwiGLU gate", gate)?;
    require_finite("SwiGLU up", up)?;
    let output = gate
        .iter()
        .zip(up)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    require_finite("SwiGLU output", &output)?;
    Ok(output)
}

pub fn residual_add_f32(residual: &[f32], update: &[f32]) -> Result<Vec<f32>, AttentionError> {
    if residual.len() != update.len() {
        return Err(AttentionError::Shape { name: "residual" });
    }
    require_finite("residual", residual)?;
    require_finite("residual update", update)?;
    let output = residual
        .iter()
        .zip(update)
        .map(|(residual, update)| residual + update)
        .collect::<Vec<_>>();
    require_finite("residual output", &output)?;
    Ok(output)
}

pub fn lm_head_f32(
    hidden: &[f32],
    hidden_size: usize,
    weight: &[f32],
    vocab_size: usize,
) -> Result<Vec<f32>, AttentionError> {
    if hidden_size == 0
        || vocab_size == 0
        || hidden.len() % hidden_size != 0
        || weight.len() != vocab_size * hidden_size
    {
        return Err(AttentionError::Shape { name: "LM head" });
    }
    require_finite("LM head hidden", hidden)?;
    require_finite("LM head weight", weight)?;
    let rows = hidden.len() / hidden_size;
    let mut output = vec![0.0; rows * vocab_size];
    for row in 0..rows {
        for token in 0..vocab_size {
            output[row * vocab_size + token] = hidden[row * hidden_size..(row + 1) * hidden_size]
                .iter()
                .zip(&weight[token * hidden_size..(token + 1) * hidden_size])
                .map(|(hidden, weight)| hidden * weight)
                .sum();
        }
    }
    require_finite("LM head output", &output)?;
    Ok(output)
}

#[derive(Debug, Clone)]
pub struct FullAttentionKv {
    n_kv_heads: usize,
    head_dim: usize,
    max_len: usize,
    len: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl FullAttentionKv {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_len: usize) -> Result<Self, AttentionError> {
        let elements = checked_elements(&[n_kv_heads, head_dim, max_len], "KV cache")?;
        if elements == 0 {
            return Err(AttentionError::Shape { name: "KV cache" });
        }
        Ok(Self {
            n_kv_heads,
            head_dim,
            max_len,
            len: 0,
            keys: vec![0.0; elements],
            values: vec![0.0; elements],
        })
    }

    pub fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        append_len: usize,
    ) -> Result<(), AttentionError> {
        let append_elements =
            checked_elements(&[append_len, self.n_kv_heads, self.head_dim], "KV append")?;
        if append_len == 0
            || keys.len() != append_elements
            || values.len() != append_elements
            || self.len.checked_add(append_len).is_none()
            || self.len + append_len > self.max_len
        {
            return Err(AttentionError::Shape { name: "KV append" });
        }
        require_finite("KV key", keys)?;
        require_finite("KV value", values)?;
        let start = self.len * self.n_kv_heads * self.head_dim;
        self.keys[start..start + append_elements].copy_from_slice(keys);
        self.values[start..start + append_elements].copy_from_slice(values);
        self.len += append_len;
        Ok(())
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys[..self.len * self.n_kv_heads * self.head_dim]
    }

    pub fn values(&self) -> &[f32] {
        &self.values[..self.len * self.n_kv_heads * self.head_dim]
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }
}

fn checked_elements(dimensions: &[usize], name: &'static str) -> Result<usize, AttentionError> {
    dimensions.iter().try_fold(1usize, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or(AttentionError::Shape { name })
    })
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), AttentionError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(AttentionError::NonFinite { name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::PackedLinearLayout;

    fn packed_linear(
        out: usize,
        input: usize,
        entries: &[(usize, usize, u32)],
    ) -> PackedLinearReference {
        let layout = PackedLinearLayout::new(out, input, 32);
        let mut weight_packed = vec![0u32; out * layout.packed_k_columns()];
        for &(row, column, value) in entries {
            weight_packed[row * layout.packed_k_columns() + column / 8] |=
                value << (4 * (column % 8));
        }
        PackedLinearReference {
            layout,
            weight_packed,
            scales: vec![1.0; out * layout.groups()],
            zero_points: vec![0; layout.packed_n_rows() * layout.groups()],
        }
    }

    #[test]
    fn embedding_reference_checks_vocab_and_selects_rows() {
        let table = [
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
            7.0, 8.0, 9.0, // token 2
        ];
        assert_eq!(
            embedding_f32(&table, 3, 3, &[2, 0]).unwrap(),
            [7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        );
        assert!(embedding_f32(&table, 3, 3, &[3]).is_err());
    }

    #[test]
    fn rms_norm_reference_normalizes_each_row_and_applies_weight() {
        let input = [3.0, 4.0, 0.0, 0.0];
        let weight = [2.0, 0.5];
        let output = rms_norm_f32(&input, 2, &weight, 0.0).unwrap();
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!((output[0] - 9.0 / rms).abs() < 1e-6);
        assert!((output[1] - 6.0 / rms).abs() < 1e-6);
        assert_eq!(&output[2..], &[0.0, 0.0]);
    }

    #[test]
    fn full_attention_q_gate_split_preserves_per_head_checkpoint_order() {
        let q_gate = (0..8).map(|value| value as f32).collect::<Vec<_>>();
        let (query, gate) = split_q_gate_f32(&q_gate, 2, 2).unwrap();
        assert_eq!(query, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(gate, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn partial_rope_rotates_only_the_configured_prefix() {
        let input = [1.0, 0.0, 0.0, 0.0, 9.0, 8.0];
        let output = apply_partial_rope_f32(&input, 1, 1, 6, 4, 10_000.0, 0).unwrap();
        assert_eq!(&output[4..], &[9.0, 8.0]);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!(output[1].abs() < 1e-6);
    }

    #[test]
    fn gqa_repeats_each_kv_head_for_its_query_group() {
        let query = vec![0.0; 4 * 2];
        let kv = vec![1.0, 2.0, 3.0, 4.0];
        let output = gqa_expand_f32(&query, &kv, 1, 4, 2, 2).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert!(gqa_expand_f32(&query, &kv, 1, 3, 2, 2).is_err());
    }

    #[test]
    fn qk_norm_normalizes_each_head_independently() {
        let input = [3.0, 4.0, 0.0, 5.0];
        let output = qk_norm_f32(&input, 1, 2, 2, &[0.0, 0.0], 0.0).unwrap();
        let first_rms = (12.5f32).sqrt();
        let second_rms = (12.5f32).sqrt();
        assert!((output[0] - 3.0 / first_rms).abs() < 1e-6);
        assert!((output[1] - 4.0 / first_rms).abs() < 1e-6);
        assert!((output[3] - 5.0 / second_rms).abs() < 1e-6);
    }

    #[test]
    fn causal_gqa_prefill_cannot_read_future_values() {
        let query = [0.0, 0.0, 0.0, 0.0];
        let key = [0.0, 0.0];
        let value = [2.0, 4.0];
        let output = causal_gqa_attention_f32(&query, &key, &value, 2, 2, 1, 1).unwrap();
        assert_eq!(output, vec![2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn full_attention_kv_append_read_and_reset_are_request_local() {
        let mut cache = FullAttentionKv::new(1, 2, 3).unwrap();
        cache.append(&[1.0, 2.0], &[3.0, 4.0], 1).unwrap();
        assert_eq!(cache.len(), 1);
        cache.append(&[5.0, 6.0], &[7.0, 8.0], 1).unwrap();
        assert_eq!(cache.keys(), &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cache.values(), &[3.0, 4.0, 7.0, 8.0]);
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.keys().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn output_gate_multiplies_attention_by_sigmoid_gate() {
        let output = apply_output_gate_f32(&[2.0, -4.0], &[0.0, 0.0]).unwrap();
        assert_eq!(output, vec![1.0, -2.0]);
    }

    #[test]
    fn causal_gqa_decode_reads_all_appended_cache_positions() {
        let mut cache = FullAttentionKv::new(1, 1, 4).unwrap();
        cache.append(&[0.0, 0.0], &[2.0, 4.0], 2).unwrap();
        let output = causal_gqa_decode_f32(&[0.0, 0.0], &cache, 2, 1).unwrap();
        assert_eq!(output, vec![3.0, 3.0]);
    }

    #[test]
    fn swiglu_mlp_and_residual_are_applied_in_model_order() {
        let gate = [0.0, 1.0];
        let up = [2.0, 3.0];
        let activated = swiglu_f32(&gate, &up).unwrap();
        assert_eq!(activated[0], 0.0);
        assert!((activated[1] - 3.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        assert_eq!(
            residual_add_f32(&[1.0, 2.0], &[3.0, 4.0]).unwrap(),
            [4.0, 6.0]
        );
    }

    #[test]
    fn lm_head_reference_returns_rows_times_vocab_logits() {
        let hidden = [1.0, 2.0, -1.0, 1.0];
        let weight = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(
            lm_head_f32(&hidden, 2, &weight, 3).unwrap(),
            [1.0, 2.0, 3.0, -1.0, 1.0, 0.0]
        );
    }

    #[test]
    fn packed_full_attention_reference_executes_complete_layer_order() {
        let identity2 = [(0, 0, 1), (1, 1, 1)];
        let layer = FullAttentionReferenceLayer {
            config: FullAttentionReferenceConfig {
                hidden_size: 2,
                intermediate_size: 1,
                n_query_heads: 1,
                n_kv_heads: 1,
                head_dim: 2,
                rotary_dim: 2,
                rope_theta: 10_000.0,
                rms_epsilon: 0.0,
            },
            input_norm: vec![0.0; 2],
            q_norm: vec![0.0; 2],
            k_norm: vec![0.0; 2],
            post_attention_norm: vec![0.0; 2],
            q_proj: packed_linear(4, 2, &identity2),
            k_proj: packed_linear(2, 2, &identity2),
            v_proj: packed_linear(2, 2, &identity2),
            o_proj: packed_linear(2, 2, &identity2),
            gate_proj: packed_linear(1, 2, &[]),
            up_proj: packed_linear(1, 2, &[]),
            down_proj: packed_linear(2, 1, &[]),
        };
        let mut cache = FullAttentionKv::new(1, 2, 4).unwrap();
        let output = layer.prefill(&[3.0, 4.0], 1, 0, &mut cache).unwrap();
        let rms = (12.5f32).sqrt();
        assert!((output[0] - (3.0 + 1.5 / rms)).abs() < 1e-6);
        assert!((output[1] - (4.0 + 2.0 / rms)).abs() < 1e-6);
        assert_eq!(cache.len(), 1);
    }
}
