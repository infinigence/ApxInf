//! Typed GR00T N1.7 action-head checkpoint weights.
//!
//! Hugging Face `nn.Linear` tensors are stored as `[out, in]` and are
//! physically transposed to `[in, out]` for ApxInf row-major GEMMs. NVIDIA's
//! category-specific matrices are already stored as `[embodiment, in, out]`
//! and must not be transposed.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};
use half::bf16;

use super::{Gr00tConfig, Gr00tVlSelfAttentionConfig};

const ACTION_HEAD: &str = "action_head";

#[derive(Debug)]
pub struct Gr00tLinearWeights {
    /// Physical `[in, out]` matrix used by row-major GEMM.
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Gr00tCategoryLinearWeights {
    /// `[num_embodiments, in, out]`; this NVIDIA parameter is not transposed.
    pub weight: Tensor,
    /// `[num_embodiments, out]`.
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Gr00tLayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Gr00tCategoryMlpWeights {
    pub input: Gr00tCategoryLinearWeights,
    pub output: Gr00tCategoryLinearWeights,
}

#[derive(Debug)]
pub struct Gr00tMlpWeights {
    pub input: Gr00tLinearWeights,
    pub output: Gr00tLinearWeights,
}

#[derive(Debug)]
pub struct Gr00tActionEncoderWeights {
    pub input: Gr00tLinearWeights,
    pub time: Gr00tLinearWeights,
    pub output: Gr00tLinearWeights,
}

/// Category-specific matrices selected for one embodiment. Selection happens
/// once during runtime preparation, so the four flow steps reuse ordinary
/// row-major GEMMs and do not require a GR00T-only CUDA kernel.
#[derive(Debug)]
pub struct Gr00tEmbodimentWeights {
    pub state_encoder: Gr00tMlpWeights,
    pub action_encoder: Gr00tActionEncoderWeights,
    pub action_decoder: Gr00tMlpWeights,
}

#[derive(Debug)]
pub struct Gr00tAttentionWeights {
    pub query: Gr00tLinearWeights,
    pub key: Gr00tLinearWeights,
    pub value: Gr00tLinearWeights,
    pub output: Gr00tLinearWeights,
}

#[derive(Debug)]
pub struct Gr00tFeedForwardWeights {
    pub input: Gr00tLinearWeights,
    pub output: Gr00tLinearWeights,
}

#[derive(Debug)]
pub struct Gr00tDitBlockWeights {
    /// `SiLU(timestep_embedding) -> [scale, shift]`.
    pub adaptive_norm: Gr00tLinearWeights,
    pub attention: Gr00tAttentionWeights,
    pub feed_forward: Gr00tFeedForwardWeights,
}

#[derive(Debug)]
pub struct Gr00tVlSelfAttentionBlockWeights {
    pub attention_norm: Gr00tLayerNormWeights,
    pub attention: Gr00tAttentionWeights,
    pub feed_forward_norm: Gr00tLayerNormWeights,
    pub feed_forward: Gr00tFeedForwardWeights,
}

/// Complete action-head weights for the released GR00T N1.7 checkpoint.
#[derive(Debug)]
pub struct Gr00tActionHeadWeights {
    pub state_encoder: Gr00tCategoryMlpWeights,
    pub action_encoder_input: Gr00tCategoryLinearWeights,
    pub action_encoder_time: Gr00tCategoryLinearWeights,
    pub action_encoder_output: Gr00tCategoryLinearWeights,
    pub action_decoder: Gr00tCategoryMlpWeights,
    pub position_embedding: Option<Tensor>,
    pub backbone_layer_norm: Option<Gr00tLayerNormWeights>,
    pub vl_self_attention: Vec<Gr00tVlSelfAttentionBlockWeights>,
    pub timestep_input: Gr00tLinearWeights,
    pub timestep_output: Gr00tLinearWeights,
    pub dit_blocks: Vec<Gr00tDitBlockWeights>,
    /// Final modulation projection, reordered at load time from NVIDIA's
    /// `[shift, scale]` layout to ApxInf's `[scale, shift]` kernel contract.
    pub output_modulation: Gr00tLinearWeights,
    pub output_projection: Gr00tLinearWeights,
}

impl Gr00tActionHeadWeights {
    pub fn from_safetensors(config: &Gr00tConfig, path: &Path) -> Result<Self> {
        let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load GR00T SafeTensors: {error}")))?;
        Self::from_map(config, &mut tensors)
    }

    /// Consume and validate every action-head tensor required for inference.
    /// Backbone tensors remain in `tensors` for the Qwen3-VL loader.
    pub fn from_map(config: &Gr00tConfig, tensors: &mut HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        let categories = config.max_num_embodiments;
        let state_input = config.state_input_dim()?;
        let action_width = config.input_embedding_dim;
        let hidden = config.hidden_size;

        let state_encoder = Gr00tCategoryMlpWeights {
            input: take_category_linear(
                tensors,
                &format!("{ACTION_HEAD}.state_encoder.layer1"),
                categories,
                state_input,
                hidden,
            )?,
            output: take_category_linear(
                tensors,
                &format!("{ACTION_HEAD}.state_encoder.layer2"),
                categories,
                hidden,
                action_width,
            )?,
        };
        let action_encoder_input = take_category_linear(
            tensors,
            &format!("{ACTION_HEAD}.action_encoder.W1"),
            categories,
            config.max_action_dim,
            action_width,
        )?;
        let action_encoder_time = take_category_linear(
            tensors,
            &format!("{ACTION_HEAD}.action_encoder.W2"),
            categories,
            checked_mul(2, action_width, "action encoder concatenated width")?,
            action_width,
        )?;
        let action_encoder_output = take_category_linear(
            tensors,
            &format!("{ACTION_HEAD}.action_encoder.W3"),
            categories,
            action_width,
            action_width,
        )?;
        let action_decoder = Gr00tCategoryMlpWeights {
            input: take_category_linear(
                tensors,
                &format!("{ACTION_HEAD}.action_decoder.layer1"),
                categories,
                hidden,
                hidden,
            )?,
            output: take_category_linear(
                tensors,
                &format!("{ACTION_HEAD}.action_decoder.layer2"),
                categories,
                hidden,
                config.max_action_dim,
            )?,
        };

        let position_embedding = if config.add_pos_embed {
            Some(take_tensor(
                tensors,
                &format!("{ACTION_HEAD}.position_embedding.weight"),
                &[config.max_seq_len, action_width],
            )?)
        } else {
            None
        };
        let backbone_layer_norm = if config.use_vlln {
            Some(take_layer_norm(
                tensors,
                &format!("{ACTION_HEAD}.vlln"),
                config.backbone_embedding_dim,
            )?)
        } else {
            None
        };

        let mut vl_self_attention = Vec::new();
        if config.use_vl_self_attention {
            let vl_config = config
                .vl_self_attention
                .as_ref()
                .ok_or_else(|| Error::Other("GR00T VL self-attention config is missing".into()))?;
            vl_self_attention.reserve(vl_config.num_layers);
            for layer in 0..vl_config.num_layers {
                vl_self_attention.push(take_vl_self_attention_block(tensors, vl_config, layer)?);
            }
        }

        let timestep_prefix = format!("{ACTION_HEAD}.model.timestep_encoder.timestep_embedder");
        let timestep_input = take_linear(
            tensors,
            &format!("{timestep_prefix}.linear_1"),
            256,
            action_width,
        )?;
        let timestep_output = take_linear(
            tensors,
            &format!("{timestep_prefix}.linear_2"),
            action_width,
            action_width,
        )?;

        let mut dit_blocks = Vec::with_capacity(config.diffusion.num_layers);
        for layer in 0..config.diffusion.num_layers {
            dit_blocks.push(take_dit_block(config, tensors, layer)?);
        }
        let output_modulation = swap_linear_output_halves(take_linear(
            tensors,
            &format!("{ACTION_HEAD}.model.proj_out_1"),
            action_width,
            checked_mul(2, action_width, "DiT output modulation width")?,
        )?)?;
        let output_projection = take_linear(
            tensors,
            &format!("{ACTION_HEAD}.model.proj_out_2"),
            action_width,
            hidden,
        )?;

        let mut unexpected = tensors
            .keys()
            .filter(|name| name.starts_with(&format!("{ACTION_HEAD}.")))
            .cloned()
            .collect::<Vec<_>>();
        unexpected.sort();
        if !unexpected.is_empty() {
            return Err(Error::Other(format!(
                "unconsumed GR00T action-head tensors: {}",
                unexpected.join(", ")
            )));
        }

        Ok(Self {
            state_encoder,
            action_encoder_input,
            action_encoder_time,
            action_encoder_output,
            action_decoder,
            position_embedding,
            backbone_layer_norm,
            vl_self_attention,
            timestep_input,
            timestep_output,
            dit_blocks,
            output_modulation,
            output_projection,
        })
    }

    pub fn select_embodiment(&self, embodiment_id: usize) -> Result<Gr00tEmbodimentWeights> {
        Ok(Gr00tEmbodimentWeights {
            state_encoder: select_category_mlp(&self.state_encoder, embodiment_id)?,
            action_encoder: Gr00tActionEncoderWeights {
                input: select_category_linear(&self.action_encoder_input, embodiment_id)?,
                time: select_category_linear(&self.action_encoder_time, embodiment_id)?,
                output: select_category_linear(&self.action_encoder_output, embodiment_id)?,
            },
            action_decoder: select_category_mlp(&self.action_decoder, embodiment_id)?,
        })
    }
}

pub(super) fn select_category_mlp(
    weights: &Gr00tCategoryMlpWeights,
    category: usize,
) -> Result<Gr00tMlpWeights> {
    Ok(Gr00tMlpWeights {
        input: select_category_linear(&weights.input, category)?,
        output: select_category_linear(&weights.output, category)?,
    })
}

pub(super) fn select_category_linear(
    weights: &Gr00tCategoryLinearWeights,
    category: usize,
) -> Result<Gr00tLinearWeights> {
    let weight_dims = weights.weight.shape().dims();
    let bias_dims = weights.bias.shape().dims();
    if weights.weight.dtype() != DType::BF16
        || weights.bias.dtype() != DType::BF16
        || weight_dims.len() != 3
        || bias_dims.len() != 2
        || weight_dims[0] != bias_dims[0]
        || weight_dims[2] != bias_dims[1]
    {
        return Err(Error::Other(format!(
            "invalid GR00T category linear shapes: {} {:?}, {} {:?}",
            weights.weight.dtype(),
            weight_dims,
            weights.bias.dtype(),
            bias_dims
        )));
    }
    if category >= weight_dims[0] {
        return Err(Error::Other(format!(
            "GR00T category {category} is outside 0..{}",
            weight_dims[0]
        )));
    }
    let input = weight_dims[1];
    let output = weight_dims[2];
    let weight_start = category * input * output;
    let bias_start = category * output;
    let weight = Tensor::from_bf16(
        vec![input, output],
        &weights.weight.as_bf16()?[weight_start..weight_start + input * output],
    )?;
    let bias = Tensor::from_bf16(
        vec![output],
        &weights.bias.as_bf16()?[bias_start..bias_start + output],
    )?;
    Ok(Gr00tLinearWeights { weight, bias })
}

fn take_dit_block(
    config: &Gr00tConfig,
    tensors: &mut HashMap<String, Tensor>,
    layer: usize,
) -> Result<Gr00tDitBlockWeights> {
    let width = config.input_embedding_dim;
    let prefix = format!("{ACTION_HEAD}.model.transformer_blocks.{layer}");
    let key_value_input = if layer % 2 == 1 && config.diffusion.interleave_self_attention {
        width
    } else {
        config.backbone_embedding_dim
    };
    Ok(Gr00tDitBlockWeights {
        adaptive_norm: take_linear(
            tensors,
            &format!("{prefix}.norm1.linear"),
            width,
            checked_mul(2, width, "DiT adaptive norm width")?,
        )?,
        attention: take_attention(
            tensors,
            &format!("{prefix}.attn1"),
            width,
            key_value_input,
            width,
        )?,
        feed_forward: take_feed_forward(tensors, &format!("{prefix}.ff"), width)?,
    })
}

fn take_vl_self_attention_block(
    tensors: &mut HashMap<String, Tensor>,
    config: &Gr00tVlSelfAttentionConfig,
    layer: usize,
) -> Result<Gr00tVlSelfAttentionBlockWeights> {
    let width = config.hidden_size()?;
    let prefix = format!("{ACTION_HEAD}.vl_self_attention.transformer_blocks.{layer}");
    Ok(Gr00tVlSelfAttentionBlockWeights {
        attention_norm: take_layer_norm(tensors, &format!("{prefix}.norm1"), width)?,
        attention: take_attention(tensors, &format!("{prefix}.attn1"), width, width, width)?,
        feed_forward_norm: take_layer_norm(tensors, &format!("{prefix}.norm3"), width)?,
        feed_forward: take_feed_forward(tensors, &format!("{prefix}.ff"), width)?,
    })
}

fn take_attention(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    query_input: usize,
    key_value_input: usize,
    output: usize,
) -> Result<Gr00tAttentionWeights> {
    Ok(Gr00tAttentionWeights {
        query: take_linear(tensors, &format!("{prefix}.to_q"), query_input, output)?,
        key: take_linear(tensors, &format!("{prefix}.to_k"), key_value_input, output)?,
        value: take_linear(tensors, &format!("{prefix}.to_v"), key_value_input, output)?,
        output: take_linear(tensors, &format!("{prefix}.to_out.0"), output, query_input)?,
    })
}

fn take_feed_forward(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    width: usize,
) -> Result<Gr00tFeedForwardWeights> {
    let inner = checked_mul(4, width, "feed-forward inner width")?;
    Ok(Gr00tFeedForwardWeights {
        input: take_linear(tensors, &format!("{prefix}.net.0.proj"), width, inner)?,
        output: take_linear(tensors, &format!("{prefix}.net.2"), inner, width)?,
    })
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<Gr00tLinearWeights> {
    let source_name = format!("{prefix}.weight");
    let source = take(tensors, &source_name)?;
    expect_bf16_shape(&source_name, &source, &[output, input])?;
    let weight = transpose_2d(&source)?;
    let bias_name = format!("{prefix}.bias");
    let bias = take(tensors, &bias_name)?;
    expect_bf16_shape(&bias_name, &bias, &[output])?;
    Ok(Gr00tLinearWeights { weight, bias })
}

fn take_category_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    categories: usize,
    input: usize,
    output: usize,
) -> Result<Gr00tCategoryLinearWeights> {
    let weight_name = format!("{prefix}.W");
    let weight = take(tensors, &weight_name)?;
    expect_bf16_shape(&weight_name, &weight, &[categories, input, output])?;
    let bias_name = format!("{prefix}.b");
    let bias = take(tensors, &bias_name)?;
    expect_bf16_shape(&bias_name, &bias, &[categories, output])?;
    Ok(Gr00tCategoryLinearWeights { weight, bias })
}

fn take_layer_norm(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    width: usize,
) -> Result<Gr00tLayerNormWeights> {
    let weight = take_tensor(tensors, &format!("{prefix}.weight"), &[width])?;
    let bias = take_tensor(tensors, &format!("{prefix}.bias"), &[width])?;
    Ok(Gr00tLayerNormWeights { weight, bias })
}

fn take_tensor(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    expected: &[usize],
) -> Result<Tensor> {
    let tensor = take(tensors, name)?;
    expect_bf16_shape(name, &tensor, expected)?;
    Ok(tensor)
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing GR00T weight {name}")))
}

fn expect_bf16_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "GR00T weight {name}: expected BF16 {expected:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::Other(format!("GR00T {name} overflow")))
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "GR00T linear weight must be 2D, got {dims:?}"
        )));
    }
    let (rows, columns) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::BF16 => {
            let source = tensor.as_bf16()?;
            let mut destination = vec![bf16::ZERO; source.len()];
            for row in 0..rows {
                for column in 0..columns {
                    destination[column * rows + row] = source[row * columns + column];
                }
            }
            Tensor::from_bf16(vec![columns, rows], &destination)
        }
        dtype => Err(Error::Other(format!(
            "GR00T N1.7 linear transpose requires BF16, got {dtype}"
        ))),
    }
}

/// NVIDIA's final DiT projection emits `[shift, scale]`, whereas every
/// adaptive-normalization kernel in ApxInf consumes `[scale, shift]`. Swap the
/// two output halves once at load time so runtime kernels have one contract.
fn swap_linear_output_halves(weights: Gr00tLinearWeights) -> Result<Gr00tLinearWeights> {
    let dims = weights.weight.shape().dims();
    if weights.weight.dtype() != DType::BF16
        || weights.bias.dtype() != DType::BF16
        || dims.len() != 2
        || dims[1] == 0
        || dims[1] % 2 != 0
        || weights.bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(format!(
            "GR00T modulation projection must be BF16 [input, 2 * width], got {:?} and {:?}",
            weights.weight.shape().dims(),
            weights.bias.shape().dims()
        )));
    }
    let rows = dims[0];
    let columns = dims[1];
    let half = columns / 2;
    let source = weights.weight.as_bf16()?;
    let mut reordered_weight = vec![bf16::ZERO; source.len()];
    for row in 0..rows {
        let offset = row * columns;
        reordered_weight[offset..offset + half]
            .copy_from_slice(&source[offset + half..offset + columns]);
        reordered_weight[offset + half..offset + columns]
            .copy_from_slice(&source[offset..offset + half]);
    }
    let source_bias = weights.bias.as_bf16()?;
    let mut reordered_bias = vec![bf16::ZERO; columns];
    reordered_bias[..half].copy_from_slice(&source_bias[half..]);
    reordered_bias[half..].copy_from_slice(&source_bias[..half]);
    Ok(Gr00tLinearWeights {
        weight: Tensor::from_bf16(vec![rows, columns], &reordered_weight)?,
        bias: Tensor::from_bf16(vec![columns], &reordered_bias)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_bf16_values_and_layout() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(bf16::from_f32);
        let source = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let output = transpose_2d(&source).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2]);
        assert_eq!(
            output
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn missing_weight_reports_the_full_checkpoint_key() {
        let mut tensors = HashMap::new();
        let error =
            Gr00tActionHeadWeights::from_map(&Gr00tConfig::default(), &mut tensors).unwrap_err();
        assert!(error
            .to_string()
            .contains("action_head.state_encoder.layer1.W"));
    }

    #[test]
    fn selects_one_embodiment_without_transposing_category_weights() {
        let weight = (0..12)
            .map(|value| bf16::from_f32(value as f32))
            .collect::<Vec<_>>();
        let bias = (0..6)
            .map(|value| bf16::from_f32((100 + value) as f32))
            .collect::<Vec<_>>();
        let category = Gr00tCategoryLinearWeights {
            weight: Tensor::from_bf16(vec![2, 2, 3], &weight).unwrap(),
            bias: Tensor::from_bf16(vec![2, 3], &bias).unwrap(),
        };
        let selected = select_category_linear(&category, 1).unwrap();
        assert_eq!(selected.weight.shape().dims(), &[2, 3]);
        assert_eq!(
            selected
                .weight
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(
            selected
                .bias
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![103.0, 104.0, 105.0]
        );
        assert!(select_category_linear(&category, 2).is_err());
    }

    #[test]
    fn swaps_final_modulation_from_shift_scale_to_scale_shift() {
        let weights = Gr00tLinearWeights {
            weight: Tensor::from_bf16(
                vec![2, 4],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(bf16::from_f32),
            )
            .unwrap(),
            bias: Tensor::from_bf16(vec![4], &[9.0, 10.0, 11.0, 12.0].map(bf16::from_f32)).unwrap(),
        };
        let reordered = swap_linear_output_halves(weights).unwrap();
        assert_eq!(
            reordered
                .weight
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]
        );
        assert_eq!(
            reordered
                .bias
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![11.0, 12.0, 9.0, 10.0]
        );
    }
}
