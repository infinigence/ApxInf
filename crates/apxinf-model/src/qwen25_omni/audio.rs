//! Native Qwen2.5-Omni Whisper-style audio encoder.

use std::collections::HashMap;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use crate::llm_trait::AudioInput;

use super::config::Qwen25OmniConfig;
use super::weights::{flatten_and_transpose, transpose_2d};

pub struct Qwen25OmniAudioWeights {
    conv1: Tensor,
    conv1_bias: Tensor,
    conv2: Tensor,
    conv2_bias: Tensor,
    bos_eos: Tensor,
    layers: Vec<AudioLayer>,
    post_norm_weight: Tensor,
    post_norm_bias: Tensor,
    projection: Tensor,
    projection_bias: Tensor,
}

struct AudioLayer {
    attn_norm_weight: Tensor,
    attn_norm_bias: Tensor,
    wq: Tensor,
    bq: Tensor,
    wk: Tensor,
    wv: Tensor,
    bv: Tensor,
    wo: Tensor,
    bo: Tensor,
    final_norm_weight: Tensor,
    final_norm_bias: Tensor,
    fc1: Tensor,
    fc1_bias: Tensor,
    fc2: Tensor,
    fc2_bias: Tensor,
}

impl Qwen25OmniAudioWeights {
    pub fn from_map(
        config: &Qwen25OmniConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        let take = |name: &str, map: &mut HashMap<String, Tensor>| {
            map.remove(name)
                .ok_or_else(|| Error::Other(format!("missing {name}")))
        };
        let audio = &config.audio;
        let conv1 = flatten_and_transpose(
            &take("thinker.audio_tower.conv1.weight", tensors)?,
            audio.hidden_size,
            audio.num_mel_bins * 3,
        )?;
        let conv2 = flatten_and_transpose(
            &take("thinker.audio_tower.conv2.weight", tensors)?,
            audio.hidden_size,
            audio.hidden_size * 3,
        )?;
        let mut layers = Vec::with_capacity(audio.n_layers);
        for index in 0..audio.n_layers {
            let prefix = format!("thinker.audio_tower.layers.{index}");
            layers.push(AudioLayer {
                attn_norm_weight: take(&format!("{prefix}.self_attn_layer_norm.weight"), tensors)?,
                attn_norm_bias: take(&format!("{prefix}.self_attn_layer_norm.bias"), tensors)?,
                wq: transpose_2d(&take(
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    tensors,
                )?)?,
                bq: take(&format!("{prefix}.self_attn.q_proj.bias"), tensors)?,
                wk: transpose_2d(&take(
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    tensors,
                )?)?,
                wv: transpose_2d(&take(
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    tensors,
                )?)?,
                bv: take(&format!("{prefix}.self_attn.v_proj.bias"), tensors)?,
                wo: transpose_2d(&take(
                    &format!("{prefix}.self_attn.out_proj.weight"),
                    tensors,
                )?)?,
                bo: take(&format!("{prefix}.self_attn.out_proj.bias"), tensors)?,
                final_norm_weight: take(&format!("{prefix}.final_layer_norm.weight"), tensors)?,
                final_norm_bias: take(&format!("{prefix}.final_layer_norm.bias"), tensors)?,
                fc1: transpose_2d(&take(&format!("{prefix}.fc1.weight"), tensors)?)?,
                fc1_bias: take(&format!("{prefix}.fc1.bias"), tensors)?,
                fc2: transpose_2d(&take(&format!("{prefix}.fc2.weight"), tensors)?)?,
                fc2_bias: take(&format!("{prefix}.fc2.bias"), tensors)?,
            });
        }
        Ok(Self {
            conv1,
            conv1_bias: take("thinker.audio_tower.conv1.bias", tensors)?,
            conv2,
            conv2_bias: take("thinker.audio_tower.conv2.bias", tensors)?,
            bos_eos: take("thinker.audio_tower.audio_bos_eos_token.weight", tensors)?,
            layers,
            post_norm_weight: take("thinker.audio_tower.ln_post.weight", tensors)?,
            post_norm_bias: take("thinker.audio_tower.ln_post.bias", tensors)?,
            projection: transpose_2d(&take("thinker.audio_tower.proj.weight", tensors)?)?,
            projection_bias: take("thinker.audio_tower.proj.bias", tensors)?,
        })
    }

    pub fn to_device(self, backend: &dyn Backend) -> Result<Self> {
        let layers = self
            .layers
            .into_iter()
            .map(|layer| {
                Ok(AudioLayer {
                    attn_norm_weight: backend.to_device(&layer.attn_norm_weight)?,
                    attn_norm_bias: backend.to_device(&layer.attn_norm_bias)?,
                    wq: backend.to_device(&layer.wq)?,
                    bq: backend.to_device(&layer.bq)?,
                    wk: backend.to_device(&layer.wk)?,
                    wv: backend.to_device(&layer.wv)?,
                    bv: backend.to_device(&layer.bv)?,
                    wo: backend.to_device(&layer.wo)?,
                    bo: backend.to_device(&layer.bo)?,
                    final_norm_weight: backend.to_device(&layer.final_norm_weight)?,
                    final_norm_bias: backend.to_device(&layer.final_norm_bias)?,
                    fc1: backend.to_device(&layer.fc1)?,
                    fc1_bias: backend.to_device(&layer.fc1_bias)?,
                    fc2: backend.to_device(&layer.fc2)?,
                    fc2_bias: backend.to_device(&layer.fc2_bias)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1: backend.to_device(&self.conv1)?,
            conv1_bias: backend.to_device(&self.conv1_bias)?,
            conv2: backend.to_device(&self.conv2)?,
            conv2_bias: backend.to_device(&self.conv2_bias)?,
            bos_eos: backend.to_device(&self.bos_eos)?,
            layers,
            post_norm_weight: backend.to_device(&self.post_norm_weight)?,
            post_norm_bias: backend.to_device(&self.post_norm_bias)?,
            projection: backend.to_device(&self.projection)?,
            projection_bias: backend.to_device(&self.projection_bias)?,
        })
    }

    pub(crate) fn boundary_embeddings(&self) -> &Tensor {
        &self.bos_eos
    }
}

pub fn output_token_count(feature_frames: usize) -> Result<usize> {
    if feature_frames == 0 {
        return Err(Error::Other(
            "qwen2.5-omni audio feature sequence is empty".into(),
        ));
    }
    // conv1 preserves length; conv2 computes ceil(n/2); the checkpoint's
    // weightless average pool then halves the sequence once more.
    let convolution_frames = feature_frames.div_ceil(2);
    if convolution_frames < 2 {
        return Err(Error::Other(
            "qwen2.5-omni audio feature sequence is too short for average pooling".into(),
        ));
    }
    Ok(convolution_frames / 2)
}

pub fn validate_input(config: &Qwen25OmniConfig, input: AudioInput<'_>) -> Result<usize> {
    let dims = input.input_features.shape().dims();
    if dims.len() != 2 || dims[1] != config.audio.num_mel_bins {
        return Err(Error::Other(format!(
            "qwen2.5-omni audio features shape {:?}, expected [frames, {}]",
            dims, config.audio.num_mel_bins
        )));
    }
    if input.feature_lengths.len() != 1 || input.token_counts.len() != 1 {
        return Err(Error::Other(
            "qwen2.5-omni first deployment slice accepts exactly one audio feature group".into(),
        ));
    }
    if input.feature_lengths[0] as usize != dims[0] {
        return Err(Error::Other(format!(
            "qwen2.5-omni audio feature length {} != tensor frames {}",
            input.feature_lengths[0], dims[0]
        )));
    }
    let mask_dims = input.attention_mask.shape().dims();
    if mask_dims != [dims[0]] && mask_dims != [1, dims[0]] {
        return Err(Error::Other(format!(
            "qwen2.5-omni audio attention mask shape {mask_dims:?}, expected [{}] or [1, {}]",
            dims[0], dims[0]
        )));
    }
    let mask = input.attention_mask.to_f32_vec()?;
    if mask
        .iter()
        .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
    {
        return Err(Error::Other(
            "qwen2.5-omni audio attention mask must contain finite 0/1 values".into(),
        ));
    }
    if mask.iter().filter(|value| **value == 1.0).count() != dims[0] {
        return Err(Error::Other(
            "qwen2.5-omni first audio slice requires an unpadded valid feature tensor".into(),
        ));
    }
    let output = output_token_count(dims[0])?;
    if input.token_counts[0] as usize != output {
        return Err(Error::Other(format!(
            "qwen2.5-omni audio token count {} != encoded feature count {output}",
            input.token_counts[0]
        )));
    }
    let convolution_frames = dims[0].div_ceil(2);
    if convolution_frames > config.audio.max_source_positions {
        return Err(Error::Other(format!(
            "qwen2.5-omni audio convolution length {convolution_frames} exceeds {}",
            config.audio.max_source_positions
        )));
    }
    Ok(output)
}

pub fn forward(
    config: &Qwen25OmniConfig,
    weights: &Qwen25OmniAudioWeights,
    backend: &dyn Backend,
    input: AudioInput<'_>,
) -> Result<Tensor> {
    let output_tokens = validate_input(config, input)?;
    let uploaded = if input.input_features.device() != backend.device() {
        Some(backend.to_device(input.input_features)?)
    } else {
        None
    };
    let features = uploaded.as_ref().unwrap_or(input.input_features);
    let mut hidden = backend.im2col1d(features, 3, 1, 1)?;
    hidden = backend.add_bias(
        &backend.matmul(&hidden, &weights.conv1)?,
        &weights.conv1_bias,
    )?;
    hidden = backend.gelu_tanh(&hidden)?;
    hidden = backend.im2col1d(&hidden, 3, 2, 1)?;
    hidden = backend.add_bias(
        &backend.matmul(&hidden, &weights.conv2)?,
        &weights.conv2_bias,
    )?;
    hidden = backend.gelu_tanh(&hidden)?;
    let convolution_tokens = hidden.shape().dims()[0];
    let position =
        sinusoidal_positions(convolution_tokens, config.audio.hidden_size, hidden.dtype())?;
    let position = backend.to_device(&position)?;
    hidden = backend.add(&hidden, &position)?;

    let audio = &config.audio;
    for layer in &weights.layers {
        let normalized = backend.layer_norm(
            &hidden,
            &layer.attn_norm_weight,
            &layer.attn_norm_bias,
            1e-5,
        )?;
        let q = backend.add_bias(&backend.matmul(&normalized, &layer.wq)?, &layer.bq)?;
        let k = backend.matmul(&normalized, &layer.wk)?;
        let v = backend.add_bias(&backend.matmul(&normalized, &layer.wv)?, &layer.bv)?;
        let q = q.reshape(vec![convolution_tokens, audio.n_heads, audio.head_dim])?;
        let k = k.reshape(vec![convolution_tokens, audio.n_heads, audio.head_dim])?;
        let v = v.reshape(vec![convolution_tokens, audio.n_heads, audio.head_dim])?;
        let attention = backend.vision_sdpa(
            &q,
            &k,
            &v,
            convolution_tokens,
            audio.n_heads,
            audio.head_dim,
        )?;
        let attention = backend.add_bias(&backend.matmul(&attention, &layer.wo)?, &layer.bo)?;
        hidden = backend.add(&hidden, &attention)?;
        let normalized = backend.layer_norm(
            &hidden,
            &layer.final_norm_weight,
            &layer.final_norm_bias,
            1e-5,
        )?;
        let mlp = backend.add_bias(&backend.matmul(&normalized, &layer.fc1)?, &layer.fc1_bias)?;
        let mlp = backend.gelu_tanh(&mlp)?;
        let mlp = backend.add_bias(&backend.matmul(&mlp, &layer.fc2)?, &layer.fc2_bias)?;
        hidden = backend.add(&hidden, &mlp)?;
    }
    let hidden = backend.layer_norm(
        &hidden,
        &weights.post_norm_weight,
        &weights.post_norm_bias,
        1e-5,
    )?;
    let hidden = backend.avg_pool1d(&hidden, 2, 2)?;
    if hidden.shape().dims()[0] != output_tokens {
        return Err(Error::Other("qwen2.5-omni audio pool length drift".into()));
    }
    backend.add_bias(
        &backend.matmul(&hidden, &weights.projection)?,
        &weights.projection_bias,
    )
}

fn sinusoidal_positions(length: usize, width: usize, dtype: DType) -> Result<Tensor> {
    if width % 2 != 0 {
        return Err(Error::Other("audio position width must be even".into()));
    }
    let half = width / 2;
    let logarithmic = 10_000.0_f32.ln() / (half.saturating_sub(1).max(1) as f32);
    let mut values = vec![0.0_f32; length * width];
    for position in 0..length {
        for index in 0..half {
            let angle = position as f32 * (-logarithmic * index as f32).exp();
            values[position * width + index] = angle.sin();
            values[position * width + half + index] = angle.cos();
        }
    }
    match dtype {
        DType::BF16 => Tensor::from_bf16(
            vec![length, width],
            &values
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        DType::F32 => Tensor::from_f32(vec![length, width], &values),
        other => Err(Error::Other(format!(
            "qwen2.5-omni audio positions do not support {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_length_matches_stride_two_contract() {
        assert!(output_token_count(1).is_err());
        assert!(output_token_count(2).is_err());
        assert_eq!(output_token_count(3).unwrap(), 1);
        assert_eq!(output_token_count(4).unwrap(), 1);
        assert_eq!(output_token_count(5).unwrap(), 1);
        assert_eq!(output_token_count(7).unwrap(), 2);
        assert!(output_token_count(0).is_err());
    }
}
