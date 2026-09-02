//! Typed GR00T action-head weights.
//!
//! Hugging Face `nn.Linear` tensors are `[out, in]`; native GEMMs consume
//! `[in, out]`. Category-specific tensors are already `[category, in, out]`,
//! so the fixed LIBERO category is sliced without transposition.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::GrootN17Config;

#[derive(Debug)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct AttentionWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct FeedForwardWeights {
    pub input: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct VlBlockWeights {
    pub norm1: LayerNormWeights,
    pub attention: AttentionWeights,
    pub norm3: LayerNormWeights,
    pub feed_forward: FeedForwardWeights,
}

#[derive(Debug)]
pub struct DitBlockWeights {
    /// Timestep conditioning projection `[1536, 3072]`.
    pub ada_norm: LinearWeights,
    pub attention: AttentionWeights,
    pub feed_forward: FeedForwardWeights,
}

#[derive(Debug)]
pub struct CategoryMlpWeights {
    pub layer1: LinearWeights,
    pub layer2: LinearWeights,
}

#[derive(Debug)]
pub struct ActionEncoderWeights {
    pub w1: LinearWeights,
    pub w2: LinearWeights,
    pub w3: LinearWeights,
}

#[derive(Debug)]
pub struct GrootN17ActionWeights {
    pub vlln: LayerNormWeights,
    pub vl_blocks: Vec<VlBlockWeights>,
    pub state_encoder: CategoryMlpWeights,
    pub action_encoder: ActionEncoderWeights,
    pub position_embedding: Tensor,
    pub timestep_in: LinearWeights,
    pub timestep_out: LinearWeights,
    pub dit_blocks: Vec<DitBlockWeights>,
    pub final_condition: LinearWeights,
    pub final_projection: LinearWeights,
    pub action_decoder: CategoryMlpWeights,
}

impl GrootN17ActionWeights {
    pub fn from_safetensors(config: &GrootN17Config, path: &Path) -> Result<Self> {
        let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load GR00T SafeTensors: {error}")))?;
        Self::from_map(config, &mut tensors)
    }

    pub fn from_map(
        config: &GrootN17Config,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        config.validate()?;
        let root = "action_head";
        let mut vl_blocks = Vec::with_capacity(config.vl_self_attention_cfg.num_layers);
        for index in 0..config.vl_self_attention_cfg.num_layers {
            let prefix = format!("{root}.vl_self_attention.transformer_blocks.{index}");
            vl_blocks.push(VlBlockWeights {
                norm1: take_layer_norm(tensors, &format!("{prefix}.norm1"))?,
                attention: take_attention(tensors, &prefix)?,
                norm3: take_layer_norm(tensors, &format!("{prefix}.norm3"))?,
                feed_forward: take_feed_forward(tensors, &prefix)?,
            });
        }

        let mut dit_blocks = Vec::with_capacity(config.diffusion_model_cfg.num_layers);
        for index in 0..config.diffusion_model_cfg.num_layers {
            let prefix = format!("{root}.model.transformer_blocks.{index}");
            dit_blocks.push(DitBlockWeights {
                ada_norm: take_linear(tensors, &format!("{prefix}.norm1.linear"), true)?,
                attention: take_attention(tensors, &prefix)?,
                feed_forward: take_feed_forward(tensors, &prefix)?,
            });
        }

        Ok(Self {
            vlln: take_layer_norm(tensors, &format!("{root}.vlln"))?,
            vl_blocks,
            state_encoder: take_category_mlp(
                tensors,
                &format!("{root}.state_encoder"),
                GrootN17Config::LIBERO_EMBODIMENT_ID,
            )?,
            action_encoder: ActionEncoderWeights {
                w1: take_category_linear(
                    tensors,
                    &format!("{root}.action_encoder.W1"),
                    GrootN17Config::LIBERO_EMBODIMENT_ID,
                )?,
                w2: take_category_linear(
                    tensors,
                    &format!("{root}.action_encoder.W2"),
                    GrootN17Config::LIBERO_EMBODIMENT_ID,
                )?,
                w3: take_category_linear(
                    tensors,
                    &format!("{root}.action_encoder.W3"),
                    GrootN17Config::LIBERO_EMBODIMENT_ID,
                )?,
            },
            position_embedding: take_leading_rows(
                &take(tensors, &format!("{root}.position_embedding.weight"))?,
                config.action_horizon,
            )?,
            timestep_in: take_linear(
                tensors,
                &format!("{root}.model.timestep_encoder.timestep_embedder.linear_1"),
                true,
            )?,
            timestep_out: take_linear(
                tensors,
                &format!("{root}.model.timestep_encoder.timestep_embedder.linear_2"),
                true,
            )?,
            dit_blocks,
            final_condition: take_linear(tensors, &format!("{root}.model.proj_out_1"), true)?,
            final_projection: take_linear(tensors, &format!("{root}.model.proj_out_2"), true)?,
            action_decoder: take_category_mlp(
                tensors,
                &format!("{root}.action_decoder"),
                GrootN17Config::LIBERO_EMBODIMENT_ID,
            )?,
        })
    }

    pub fn to_device(mut self, backend: &dyn Backend) -> Result<Self> {
        upload_norm(&mut self.vlln, backend)?;
        for block in &mut self.vl_blocks {
            upload_norm(&mut block.norm1, backend)?;
            upload_attention(&mut block.attention, backend)?;
            upload_norm(&mut block.norm3, backend)?;
            upload_feed_forward(&mut block.feed_forward, backend)?;
        }
        upload_category_mlp(&mut self.state_encoder, backend)?;
        upload_linear(&mut self.action_encoder.w1, backend)?;
        upload_linear(&mut self.action_encoder.w2, backend)?;
        upload_linear(&mut self.action_encoder.w3, backend)?;
        self.position_embedding = backend.to_device(&self.position_embedding)?;
        upload_linear(&mut self.timestep_in, backend)?;
        upload_linear(&mut self.timestep_out, backend)?;
        for block in &mut self.dit_blocks {
            upload_linear(&mut block.ada_norm, backend)?;
            upload_attention(&mut block.attention, backend)?;
            upload_feed_forward(&mut block.feed_forward, backend)?;
        }
        upload_linear(&mut self.final_condition, backend)?;
        upload_linear(&mut self.final_projection, backend)?;
        upload_category_mlp(&mut self.action_decoder, backend)?;
        Ok(self)
    }
}

pub(super) fn upload_linear(linear: &mut LinearWeights, backend: &dyn Backend) -> Result<()> {
    linear.weight = backend.to_device(&linear.weight)?;
    if let Some(bias) = &linear.bias {
        linear.bias = Some(backend.to_device(bias)?);
    }
    Ok(())
}

fn upload_norm(norm: &mut LayerNormWeights, backend: &dyn Backend) -> Result<()> {
    norm.weight = backend.to_device(&norm.weight)?;
    norm.bias = backend.to_device(&norm.bias)?;
    Ok(())
}

fn upload_attention(attention: &mut AttentionWeights, backend: &dyn Backend) -> Result<()> {
    upload_linear(&mut attention.q, backend)?;
    upload_linear(&mut attention.k, backend)?;
    upload_linear(&mut attention.v, backend)?;
    upload_linear(&mut attention.output, backend)
}

fn upload_feed_forward(feed_forward: &mut FeedForwardWeights, backend: &dyn Backend) -> Result<()> {
    upload_linear(&mut feed_forward.input, backend)?;
    upload_linear(&mut feed_forward.output, backend)
}

fn upload_category_mlp(mlp: &mut CategoryMlpWeights, backend: &dyn Backend) -> Result<()> {
    upload_linear(&mut mlp.layer1, backend)?;
    upload_linear(&mut mlp.layer2, backend)
}

fn take_attention(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<AttentionWeights> {
    Ok(AttentionWeights {
        q: take_linear(tensors, &format!("{prefix}.attn1.to_q"), true)?,
        k: take_linear(tensors, &format!("{prefix}.attn1.to_k"), true)?,
        v: take_linear(tensors, &format!("{prefix}.attn1.to_v"), true)?,
        output: take_linear(tensors, &format!("{prefix}.attn1.to_out.0"), true)?,
    })
}

fn take_feed_forward(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<FeedForwardWeights> {
    Ok(FeedForwardWeights {
        input: take_linear(tensors, &format!("{prefix}.ff.net.0.proj"), true)?,
        output: take_linear(tensors, &format!("{prefix}.ff.net.2"), true)?,
    })
}

fn take_category_mlp(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    category: usize,
) -> Result<CategoryMlpWeights> {
    Ok(CategoryMlpWeights {
        layer1: take_category_linear(tensors, &format!("{prefix}.layer1"), category)?,
        layer2: take_category_linear(tensors, &format!("{prefix}.layer2"), category)?,
    })
}

fn take_layer_norm(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    bias: bool,
) -> Result<LinearWeights> {
    let weight = transpose_2d(&take(tensors, &format!("{prefix}.weight"))?)?;
    let bias = bias
        .then(|| take(tensors, &format!("{prefix}.bias")))
        .transpose()?;
    Ok(LinearWeights { weight, bias })
}

fn take_category_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    category: usize,
) -> Result<LinearWeights> {
    let weight = select_category(&take(tensors, &format!("{prefix}.W"))?, category)?;
    let bias = select_category(&take(tensors, &format!("{prefix}.b"))?, category)?;
    Ok(LinearWeights {
        weight,
        bias: Some(bias),
    })
}

pub(super) fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing GR00T tensor `{name}`")))
}

fn select_category(tensor: &Tensor, category: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() < 2 || category >= dims[0] {
        return Err(Error::Other(format!(
            "category {category} is invalid for shape {dims:?}"
        )));
    }
    let category_elements: usize = dims[1..].iter().product();
    let output_shape = dims[1..].to_vec();
    match tensor.dtype() {
        DType::BF16 => {
            let values = tensor.as_bf16()?;
            let start = category * category_elements;
            Tensor::from_bf16(output_shape, &values[start..start + category_elements])
        }
        DType::F32 => {
            let values = tensor.as_f32()?;
            let start = category * category_elements;
            Tensor::from_f32(output_shape, &values[start..start + category_elements])
        }
        dtype => Err(Error::Other(format!(
            "GR00T category weights require BF16/F32, got {dtype}"
        ))),
    }
}

fn take_leading_rows(tensor: &Tensor, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 || rows > dims[0] {
        return Err(Error::Other(format!(
            "cannot take {rows} leading rows from shape {dims:?}"
        )));
    }
    let cols = dims[1];
    match tensor.dtype() {
        DType::BF16 => Tensor::from_bf16(vec![rows, cols], &tensor.as_bf16()?[..rows * cols]),
        DType::F32 => Tensor::from_f32(vec![rows, cols], &tensor.as_f32()?[..rows * cols]),
        dtype => Err(Error::Other(format!(
            "GR00T leading rows require BF16/F32, got {dtype}"
        ))),
    }
}

pub(super) fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "GR00T linear weight must be 2D, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::BF16 => {
            let input = tensor.as_bf16()?;
            let mut output = vec![half::bf16::from_f32(0.0); input.len()];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &output)
        }
        DType::F32 => {
            let input = tensor.as_f32()?;
            let mut output = vec![0.0; input.len()];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &output)
        }
        dtype => Err(Error::Other(format!(
            "GR00T linear weights require BF16/F32, got {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use apxinf_core::Tensor;
    use half::bf16;

    use super::{select_category, transpose_2d};

    #[test]
    fn category_slice_preserves_in_out_layout() {
        let values = (0..24).map(|value| bf16::from_f32(value as f32)).collect::<Vec<_>>();
        let tensor = Tensor::from_bf16(vec![2, 3, 4], &values).unwrap();
        let selected = select_category(&tensor, 1).unwrap();
        assert_eq!(selected.shape().dims(), [3, 4]);
        assert_eq!(selected.to_f32_vec().unwrap(), (12..24).map(|v| v as f32).collect::<Vec<_>>());
    }

    #[test]
    fn transposes_hugging_face_linear_weight() {
        let tensor = Tensor::from_f32(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let transposed = transpose_2d(&tensor).unwrap();
        assert_eq!(transposed.shape().dims(), [3, 2]);
        assert_eq!(transposed.as_f32().unwrap(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
