use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GdnError {
    #[error("GDN dimensions are inconsistent")]
    Dimensions,
    #[error("GDN {name} shape is inconsistent")]
    Shape { name: &'static str },
    #[error("GDN {name} contains a non-finite value")]
    NonFinite { name: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDimensions {
    pub conv_kernel: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GdnDimensions {
    pub fn new(
        conv_kernel: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self, GdnError> {
        if conv_kernel == 0
            || key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || value_heads % key_heads != 0
        {
            return Err(GdnError::Dimensions);
        }
        Ok(Self {
            conv_kernel,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        })
    }

    pub fn qwen35() -> Result<Self, GdnError> {
        Self::new(4, 16, 48, 128, 128)
    }

    pub const fn conv_channels(self) -> usize {
        self.key_heads * self.key_dim * 2 + self.value_heads * self.value_dim
    }

    const fn recurrent_elements(self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

#[derive(Debug, Clone)]
pub struct GdnState {
    dimensions: GdnDimensions,
    conv_ring: Vec<f32>,
    conv_cursor: usize,
    recurrent: Vec<f32>,
}

impl GdnState {
    pub fn new(dimensions: GdnDimensions) -> Result<Self, GdnError> {
        let conv_elements = dimensions
            .conv_channels()
            .checked_mul(dimensions.conv_kernel)
            .ok_or(GdnError::Dimensions)?;
        Ok(Self {
            dimensions,
            conv_ring: vec![0.0; conv_elements],
            conv_cursor: 0,
            recurrent: vec![0.0; dimensions.recurrent_elements()],
        })
    }

    pub fn causal_conv_silu(
        &mut self,
        input: &[f32],
        weights: &[f32],
    ) -> Result<Vec<f32>, GdnError> {
        let channels = self.dimensions.conv_channels();
        if input.len() != channels || weights.len() != channels * self.dimensions.conv_kernel {
            return Err(GdnError::Shape {
                name: "convolution",
            });
        }
        require_finite("convolution input", input)?;
        require_finite("convolution weight", weights)?;
        for (channel, value) in input.iter().enumerate() {
            self.conv_ring[channel * self.dimensions.conv_kernel + self.conv_cursor] = *value;
        }
        self.conv_cursor = (self.conv_cursor + 1) % self.dimensions.conv_kernel;
        let mut output = vec![0.0; channels];
        for channel in 0..channels {
            let base = channel * self.dimensions.conv_kernel;
            let sum = (0..self.dimensions.conv_kernel)
                .map(|offset| {
                    let ring_index = (self.conv_cursor + offset) % self.dimensions.conv_kernel;
                    self.conv_ring[base + ring_index] * weights[base + offset]
                })
                .sum::<f32>();
            output[channel] = sum / (1.0 + (-sum).exp());
        }
        require_finite("convolution output", &output)?;
        Ok(output)
    }

    pub fn conv_ring_channel(&self, channel: usize) -> Result<Vec<f32>, GdnError> {
        if channel >= self.dimensions.conv_channels() {
            return Err(GdnError::Shape {
                name: "convolution channel",
            });
        }
        let base = channel * self.dimensions.conv_kernel;
        Ok((0..self.dimensions.conv_kernel)
            .map(|offset| {
                self.conv_ring[base + (self.conv_cursor + offset) % self.dimensions.conv_kernel]
            })
            .collect())
    }

    pub fn recurrent_step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        g: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, GdnError> {
        let d = self.dimensions;
        if query.len() != d.key_heads * d.key_dim
            || key.len() != d.key_heads * d.key_dim
            || value.len() != d.value_heads * d.value_dim
            || g.len() != d.value_heads
            || beta.len() != d.value_heads
        {
            return Err(GdnError::Shape {
                name: "recurrent step",
            });
        }
        for (name, values) in [
            ("query", query),
            ("key", key),
            ("value", value),
            ("decay", g),
            ("beta", beta),
        ] {
            require_finite(name, values)?;
        }
        let query = l2_normalize_heads(query, d.key_heads, d.key_dim)?;
        let key = l2_normalize_heads(key, d.key_heads, d.key_dim)?;
        let ratio = d.value_heads / d.key_heads;
        let query_scale = (d.key_dim as f32).sqrt().recip();
        let mut next = self.recurrent.clone();
        let mut output = vec![0.0; d.value_heads * d.value_dim];
        for value_head in 0..d.value_heads {
            let key_head = value_head / ratio;
            let q_base = key_head * d.key_dim;
            let state_base = value_head * d.key_dim * d.value_dim;
            let value_base = value_head * d.value_dim;
            let decay = g[value_head].exp();
            for element in &mut next[state_base..state_base + d.key_dim * d.value_dim] {
                *element *= decay;
            }
            for value_dimension in 0..d.value_dim {
                let memory = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * key[q_base + key_dimension]
                    })
                    .sum::<f32>();
                let delta = (value[value_base + value_dimension] - memory) * beta[value_head];
                for key_dimension in 0..d.key_dim {
                    next[state_base + key_dimension * d.value_dim + value_dimension] +=
                        key[q_base + key_dimension] * delta;
                }
                output[value_base + value_dimension] = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * query[q_base + key_dimension]
                            * query_scale
                    })
                    .sum();
            }
        }
        require_finite("recurrent state", &next)?;
        require_finite("recurrent output", &output)?;
        self.recurrent = next;
        Ok(output)
    }

    /// Apply the Qwen3.5 recurrent gated-delta update. The query and key
    /// slices are already expanded to one head per value head. State remains
    /// FP32 even when the surrounding activations are BF16.
    pub fn gated_delta_step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        decay: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>, GdnError> {
        let d = self.dimensions;
        if query.len() != d.value_heads * d.key_dim
            || key.len() != query.len()
            || value.len() != d.value_heads * d.value_dim
            || decay.len() != d.value_heads
            || beta.len() != d.value_heads
        {
            return Err(GdnError::Shape {
                name: "gated delta step",
            });
        }
        for (name, values) in [
            ("query", query),
            ("key", key),
            ("value", value),
            ("decay", decay),
            ("beta", beta),
        ] {
            require_finite(name, values)?;
        }
        let query = l2_normalize_heads(query, d.value_heads, d.key_dim)?;
        let key = l2_normalize_heads(key, d.value_heads, d.key_dim)?;
        let query_scale = (d.key_dim as f32).sqrt().recip();
        let mut next = self.recurrent.clone();
        let mut output = vec![0.0; d.value_heads * d.value_dim];
        for head in 0..d.value_heads {
            let state_base = head * d.key_dim * d.value_dim;
            let q_base = head * d.key_dim;
            let value_base = head * d.value_dim;
            let decay_factor = decay[head].exp();
            for element in &mut next[state_base..state_base + d.key_dim * d.value_dim] {
                *element *= decay_factor;
            }
            for value_dimension in 0..d.value_dim {
                let memory = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * key[q_base + key_dimension]
                    })
                    .sum::<f32>();
                let delta = (value[value_base + value_dimension] - memory) * beta[head];
                for key_dimension in 0..d.key_dim {
                    next[state_base + key_dimension * d.value_dim + value_dimension] +=
                        key[q_base + key_dimension] * delta;
                }
                output[value_base + value_dimension] = (0..d.key_dim)
                    .map(|key_dimension| {
                        next[state_base + key_dimension * d.value_dim + value_dimension]
                            * query[q_base + key_dimension]
                            * query_scale
                    })
                    .sum();
            }
        }
        require_finite("recurrent state", &next)?;
        require_finite("recurrent output", &output)?;
        self.recurrent = next;
        Ok(output)
    }

    pub fn eager_prefill(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        g: &[f32],
        beta: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, GdnError> {
        let d = self.dimensions;
        if seq_len == 0
            || query.len() != seq_len * d.key_heads * d.key_dim
            || key.len() != query.len()
            || value.len() != seq_len * d.value_heads * d.value_dim
            || g.len() != seq_len * d.value_heads
            || beta.len() != g.len()
        {
            return Err(GdnError::Shape {
                name: "eager prefill",
            });
        }
        let mut output = Vec::with_capacity(seq_len * d.value_heads * d.value_dim);
        for position in 0..seq_len {
            output.extend(self.recurrent_step(
                &query
                    [position * d.key_heads * d.key_dim..(position + 1) * d.key_heads * d.key_dim],
                &key[position * d.key_heads * d.key_dim..(position + 1) * d.key_heads * d.key_dim],
                &value[position * d.value_heads * d.value_dim
                    ..(position + 1) * d.value_heads * d.value_dim],
                &g[position * d.value_heads..(position + 1) * d.value_heads],
                &beta[position * d.value_heads..(position + 1) * d.value_heads],
            )?);
        }
        Ok(output)
    }

    pub const fn recurrent_dtype(&self) -> &'static str {
        "f32"
    }

    pub fn reset(&mut self) {
        self.conv_ring.fill(0.0);
        self.conv_cursor = 0;
        self.recurrent.fill(0.0);
    }

    pub fn checksum(&self) -> u64 {
        self.conv_ring.iter().chain(&self.recurrent).fold(
            0xcbf29ce484222325u64 ^ self.conv_cursor as u64,
            |hash, value| (hash ^ value.to_bits() as u64).wrapping_mul(0x100000001b3),
        )
    }
}

/// Small dense matrix used by the CPU reference fixture. Production CUDA
/// layers retain packed W4 projections and do not expand the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, GdnError> {
        if rows == 0 || cols == 0 || values.len() != rows * cols {
            return Err(GdnError::Shape {
                name: "dense matrix",
            });
        }
        require_finite("dense matrix", &values)?;
        Ok(Self { rows, cols, values })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, GdnError> {
        if input.len() != self.cols {
            return Err(GdnError::Shape {
                name: "dense projection",
            });
        }
        require_finite("dense activation", input)?;
        let mut output = vec![0.0; self.rows];
        for (row, result) in output.iter_mut().enumerate() {
            *result = self.values[row * self.cols..(row + 1) * self.cols]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        require_finite("dense projection output", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct GdnReferenceLayerConfig {
    pub hidden_size: usize,
    pub dimensions: GdnDimensions,
    pub rms_epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct GdnReferenceWeights {
    pub in_proj_qkv: DenseMatrix,
    pub in_proj_z: DenseMatrix,
    pub in_proj_a: DenseMatrix,
    pub in_proj_b: DenseMatrix,
    pub conv_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub out_proj: DenseMatrix,
}

#[derive(Debug, Clone)]
pub struct GdnReferenceLayer {
    config: GdnReferenceLayerConfig,
    weights: Option<GdnReferenceWeights>,
}

impl GdnReferenceLayer {
    pub fn new(config: GdnReferenceLayerConfig) -> Result<Self, GdnError> {
        if config.hidden_size == 0 || config.rms_epsilon <= 0.0 || !config.rms_epsilon.is_finite() {
            return Err(GdnError::Dimensions);
        }
        Ok(Self {
            config,
            weights: None,
        })
    }

    pub fn set_weights(&mut self, weights: GdnReferenceWeights) -> Result<(), GdnError> {
        let d = self.config.dimensions;
        if weights.in_proj_qkv.rows() != d.conv_channels()
            || weights.in_proj_qkv.cols() != self.config.hidden_size
            || weights.in_proj_z.rows() != d.value_heads * d.value_dim
            || weights.in_proj_z.cols() != self.config.hidden_size
            || weights.in_proj_a.rows() != d.value_heads
            || weights.in_proj_a.cols() != self.config.hidden_size
            || weights.in_proj_b.rows() != d.value_heads
            || weights.in_proj_b.cols() != self.config.hidden_size
            || weights.out_proj.rows() != self.config.hidden_size
            || weights.out_proj.cols() != d.value_heads * d.value_dim
            || weights.conv_weight.len() != d.conv_channels() * d.conv_kernel
            || weights.a_log.len() != d.value_heads
            || weights.dt_bias.len() != d.value_heads
            || weights.norm_weight.len() != d.value_dim
        {
            return Err(GdnError::Shape {
                name: "GDN weights",
            });
        }
        require_finite("GDN convolution weight", &weights.conv_weight)?;
        require_finite("GDN A_log", &weights.a_log)?;
        require_finite("GDN dt_bias", &weights.dt_bias)?;
        require_finite("GDN norm weight", &weights.norm_weight)?;
        self.weights = Some(weights);
        Ok(())
    }

    pub const fn weights_configured(&self) -> bool {
        self.weights.is_some()
    }

    pub fn decode_token(&self, hidden: &[f32], state: &mut GdnState) -> Result<Vec<f32>, GdnError> {
        let weights = self.weights.as_ref().ok_or(GdnError::Shape {
            name: "GDN weights",
        })?;
        if hidden.len() != self.config.hidden_size {
            return Err(GdnError::Shape { name: "GDN hidden" });
        }
        if state.dimensions != self.config.dimensions {
            return Err(GdnError::Shape { name: "GDN state" });
        }
        require_finite("GDN hidden", hidden)?;

        // Work on a clone so a failed launch or non-finite intermediate never
        // leaves a partially advanced request state.
        let mut working = state.clone();
        let mixed_qkv = weights.in_proj_qkv.apply(hidden)?;
        let convolved = working.causal_conv_silu(&mixed_qkv, &weights.conv_weight)?;
        let d = self.config.dimensions;
        let key_width = d.key_heads * d.key_dim;
        let value_width = d.value_heads * d.value_dim;
        let query_base = &convolved[..key_width];
        let key_base = &convolved[key_width..2 * key_width];
        let value = &convolved[2 * key_width..2 * key_width + value_width];
        let ratio = d.value_heads / d.key_heads;
        let mut query = vec![0.0; d.value_heads * d.key_dim];
        let mut key = vec![0.0; query.len()];
        for head in 0..d.value_heads {
            let source = head / ratio;
            query[head * d.key_dim..(head + 1) * d.key_dim]
                .copy_from_slice(&query_base[source * d.key_dim..(source + 1) * d.key_dim]);
            key[head * d.key_dim..(head + 1) * d.key_dim]
                .copy_from_slice(&key_base[source * d.key_dim..(source + 1) * d.key_dim]);
        }
        let a = weights.in_proj_a.apply(hidden)?;
        let b = weights.in_proj_b.apply(hidden)?;
        let mut decay = vec![0.0; d.value_heads];
        let mut beta = vec![0.0; d.value_heads];
        for head in 0..d.value_heads {
            decay[head] = -weights.a_log[head].exp() * softplus(a[head] + weights.dt_bias[head]);
            beta[head] = sigmoid(b[head]);
        }
        let core = working.gated_delta_step(&query, &key, value, &decay, &beta)?;
        let z = weights.in_proj_z.apply(hidden)?;
        let mut gated = vec![0.0; value_width];
        for head in 0..d.value_heads {
            let base = head * d.value_dim;
            let row = &core[base..base + d.value_dim];
            let rms = (row.iter().map(|value| value * value).sum::<f32>() / d.value_dim as f32
                + self.config.rms_epsilon)
                .sqrt()
                .recip();
            for dimension in 0..d.value_dim {
                let normalized = row[dimension] * rms * weights.norm_weight[dimension];
                gated[base + dimension] = normalized * silu(z[base + dimension]);
            }
        }
        let output = weights.out_proj.apply(&gated)?;
        require_finite("GDN output", &output)?;
        *state = working;
        Ok(output)
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn l2_normalize_heads(values: &[f32], heads: usize, head_dim: usize) -> Result<Vec<f32>, GdnError> {
    let mut output = Vec::with_capacity(values.len());
    for head in 0..heads {
        let row = &values[head * head_dim..(head + 1) * head_dim];
        let inverse = (row.iter().map(|value| value * value).sum::<f32>() + 1e-6)
            .sqrt()
            .recip();
        output.extend(row.iter().map(|value| value * inverse));
    }
    require_finite("normalized query/key", &output)?;
    Ok(output)
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), GdnError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(GdnError::NonFinite { name })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_layer() -> GdnReferenceLayer {
        GdnReferenceLayer::new(GdnReferenceLayerConfig {
            hidden_size: 2,
            dimensions: GdnDimensions::new(2, 1, 1, 2, 2).unwrap(),
            rms_epsilon: 1e-6,
        })
        .unwrap()
    }

    #[test]
    fn production_dimensions_match_checkpoint_contract() {
        let dimensions = GdnDimensions::qwen35().unwrap();
        assert_eq!(dimensions.conv_kernel, 4);
        assert_eq!(dimensions.key_heads, 16);
        assert_eq!(dimensions.value_heads, 48);
        assert_eq!(dimensions.key_dim, 128);
        assert_eq!(dimensions.value_dim, 128);
        assert_eq!(dimensions.conv_channels(), 10_240);
    }

    #[test]
    fn convolution_ring_keeps_only_four_causal_positions() {
        let dimensions = GdnDimensions::new(4, 1, 1, 1, 1).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut fifth = Vec::new();
        for value in 1..=5 {
            fifth = state
                .causal_conv_silu(&[value as f32; 3], &weights)
                .unwrap();
        }
        assert_eq!(state.conv_ring_channel(0).unwrap(), [2.0, 3.0, 4.0, 5.0]);
        let expected = 14.0 / (1.0 + (-14.0f32).exp());
        assert!((fifth[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn fp32_recurrent_delta_update_matches_equation() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let output = state
            .recurrent_step(
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[2.0, 4.0],
                &[-2.0f32.ln()],
                &[0.5],
            )
            .unwrap();
        let norm = (1.0 + 1e-6f32).sqrt();
        assert!((output[0] - 1.0 / norm * 1.0 / norm * 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
        assert!((output[1] - 2.0 / norm * 1.0 / norm * 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
        assert_eq!(state.recurrent_dtype(), "f32");
    }

    #[test]
    fn eager_prefill_matches_repeated_single_token_decode() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = [1.0, 0.0, 0.0, 1.0];
        let value = [2.0, 4.0, 6.0, 8.0];
        let g = [-0.5, -0.25];
        let beta = [0.5, 0.25];
        let mut prefill = GdnState::new(dimensions).unwrap();
        let prefill_output = prefill
            .eager_prefill(&query, &key, &value, &g, &beta, 2)
            .unwrap();
        let mut decode = GdnState::new(dimensions).unwrap();
        let mut decode_output = Vec::new();
        for position in 0..2 {
            decode_output.extend(
                decode
                    .recurrent_step(
                        &query[position * 2..position * 2 + 2],
                        &key[position * 2..position * 2 + 2],
                        &value[position * 2..position * 2 + 2],
                        &g[position..position + 1],
                        &beta[position..position + 1],
                    )
                    .unwrap(),
            );
        }
        assert_eq!(prefill_output, decode_output);
        assert_eq!(prefill.checksum(), decode.checksum());
    }

    #[test]
    fn reset_and_failed_step_leave_no_request_state() {
        let dimensions = GdnDimensions::new(4, 1, 1, 2, 2).unwrap();
        let mut state = GdnState::new(dimensions).unwrap();
        let before = state.checksum();
        assert!(state
            .recurrent_step(&[f32::NAN, 0.0], &[1.0, 0.0], &[1.0, 1.0], &[-1.0], &[0.5])
            .is_err());
        assert_eq!(state.checksum(), before);
        state
            .recurrent_step(&[1.0, 0.0], &[1.0, 0.0], &[1.0, 1.0], &[-1.0], &[0.5])
            .unwrap();
        state.reset();
        assert_eq!(state.checksum(), before);
    }

    #[test]
    fn reference_layer_applies_gated_delta_norm_and_projection_in_order() {
        let mut layer = tiny_layer();
        layer
            .set_weights(GdnReferenceWeights {
                in_proj_qkv: DenseMatrix::new(
                    6,
                    2,
                    vec![
                        1.0, 0.0, // q
                        0.0, 1.0, // k
                        1.0, 0.0, // k
                        0.0, 1.0, // k
                        1.0, 1.0, // v[0]
                        0.0, 0.0, // v[1]
                    ],
                )
                .unwrap(),
                in_proj_z: DenseMatrix::new(2, 2, vec![1.0, 0.0, 1.0, 0.0]).unwrap(),
                in_proj_a: DenseMatrix::new(1, 2, vec![0.0, 0.0]).unwrap(),
                in_proj_b: DenseMatrix::new(1, 2, vec![0.0, 0.0]).unwrap(),
                conv_weight: vec![1.0; 6 * 2],
                a_log: vec![0.0],
                dt_bias: vec![0.0],
                norm_weight: vec![1.0, 1.0],
                out_proj: DenseMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            })
            .unwrap();

        let mut state = GdnState::new(GdnDimensions::new(2, 1, 1, 2, 2).unwrap()).unwrap();
        let output = layer.decode_token(&[1.0, 2.0], &mut state).unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    fn reference_layer_rejects_bad_weights_without_mutating_state() {
        let layer = tiny_layer();
        let before = layer.weights_configured();
        let mut state = GdnState::new(GdnDimensions::new(2, 1, 1, 2, 2).unwrap()).unwrap();
        let checksum = state.checksum();
        assert!(matches!(
            layer.decode_token(&[1.0, 2.0], &mut state),
            Err(GdnError::Shape { .. })
        ));
        assert_eq!(state.checksum(), checksum);
        assert!(!before);
    }
}
