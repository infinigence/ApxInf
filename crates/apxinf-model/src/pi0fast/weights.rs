//! Typed π0-FAST checkpoint weights.
//!
//! The checkpoint is a LeRobot `pi0_fast` export in Hugging Face's `[out, in]`
//! linear layout. ApxInf row-major GEMMs consume `[in, out]`, so every
//! projection is physically transposed while loading. Gemma RMSNorm parameters
//! are learned offsets; they are converted once to their final `1 + weight`
//! scale, then folded into the consuming Q/K/V and gate/up matrices exactly as
//! the π0.5 loader does. The token embedding and LM head are tied: the
//! checkpoint ships only `...paligemma.lm_head.weight`, so a single
//! `[vocab, width]` tensor serves both the embedding lookup and (transposed
//! on device) the output projection.

use std::collections::HashMap;
use std::path::Path;

use half::bf16;
use apxinf_core::{DType, Error, Result, Tensor};

use super::Pi0FastConfig;

const ROOT: &str = "paligemma_with_expert";

#[derive(Debug)]
pub struct LinearWeights {
    /// Physical `[in, out]` matrix used by row-major GEMM.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct VisionBlockWeights {
    pub norm1: LayerNormWeights,
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub norm2: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

#[derive(Debug)]
pub struct VisionWeights {
    /// Flattened patch convolution `[3 * patch * patch, vision_width]`.
    pub patch_embedding: LinearWeights,
    pub position_embedding: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    pub post_layer_norm: LayerNormWeights,
    pub multimodal_projector: LinearWeights,
}

#[derive(Debug)]
pub struct GemmaAttentionWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
}

#[derive(Debug)]
pub struct GemmaMlpWeights {
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
}

#[derive(Debug)]
pub struct LanguageLayerWeights {
    /// Folded Gemma norm leaves an identity multiplier on the activation path.
    pub input_norm_scale: Tensor,
    pub attention: GemmaAttentionWeights,
    pub post_attention_norm_scale: Tensor,
    pub mlp: GemmaMlpWeights,
}

#[derive(Debug)]
pub struct Pi0FastWeights {
    pub vision: VisionWeights,
    pub language_layers: Vec<LanguageLayerWeights>,
    pub language_final_norm_scale: Tensor,
    /// Tied token embedding / LM head, stored `[vocab_size, language_width]`.
    pub lm_head: Tensor,
}

impl Pi0FastWeights {
    pub fn from_safetensors(config: &Pi0FastConfig, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load π0-FAST SafeTensors: {error}")))?;
        Self::from_map(config, tensors)
    }

    pub fn from_map(config: &Pi0FastConfig, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        normalize_lerobot_prefix(&mut tensors);

        let vision_prefix = format!("{ROOT}.paligemma.model.vision_tower.vision_model");
        let mut vision_blocks = Vec::with_capacity(config.vision_depth);
        for layer in 0..config.vision_depth {
            let p = format!("{vision_prefix}.encoder.layers.{layer}");
            vision_blocks.push(VisionBlockWeights {
                norm1: take_layer_norm(&mut tensors, &format!("{p}.layer_norm1"))?,
                q: take_linear(&mut tensors, &format!("{p}.self_attn.q_proj"), true)?,
                k: take_linear(&mut tensors, &format!("{p}.self_attn.k_proj"), true)?,
                v: take_linear(&mut tensors, &format!("{p}.self_attn.v_proj"), true)?,
                output: take_linear(&mut tensors, &format!("{p}.self_attn.out_proj"), true)?,
                norm2: take_layer_norm(&mut tensors, &format!("{p}.layer_norm2"))?,
                fc1: take_linear(&mut tensors, &format!("{p}.mlp.fc1"), true)?,
                fc2: take_linear(&mut tensors, &format!("{p}.mlp.fc2"), true)?,
            });
        }

        let patch_weight_name = format!("{vision_prefix}.embeddings.patch_embedding.weight");
        let patch_weight = take(&mut tensors, &patch_weight_name)?;
        let expected = config.vision_width * 3 * config.patch_size * config.patch_size;
        if patch_weight.numel() != expected {
            return Err(Error::Other(format!(
                "{patch_weight_name}: expected {expected} elements, got {}",
                patch_weight.numel()
            )));
        }
        let patch_weight = patch_weight.reshape(vec![
            config.vision_width,
            3 * config.patch_size * config.patch_size,
        ])?;
        let patch_embedding = LinearWeights {
            weight: transpose_2d(&patch_weight)?,
            bias: Some(take(
                &mut tensors,
                &format!("{vision_prefix}.embeddings.patch_embedding.bias"),
            )?),
        };

        let vision = VisionWeights {
            patch_embedding,
            position_embedding: take(
                &mut tensors,
                &format!("{vision_prefix}.embeddings.position_embedding.weight"),
            )?,
            blocks: vision_blocks,
            post_layer_norm: take_layer_norm(
                &mut tensors,
                &format!("{vision_prefix}.post_layernorm"),
            )?,
            multimodal_projector: take_linear(
                &mut tensors,
                &format!("{ROOT}.paligemma.model.multi_modal_projector.linear"),
                true,
            )?,
        };

        let language_prefix = format!("{ROOT}.paligemma.model.language_model");
        let mut language_layers = Vec::with_capacity(config.language.depth);
        for layer in 0..config.language.depth {
            let p = format!("{language_prefix}.layers.{layer}");
            let attention_scale =
                add_one(take(&mut tensors, &format!("{p}.input_layernorm.weight"))?)?;
            let mlp_scale = add_one(take(
                &mut tensors,
                &format!("{p}.post_attention_layernorm.weight"),
            )?)?;
            let mut attention = take_attention(&mut tensors, &p)?;
            attention.q = fold_input_scale(attention.q, &attention_scale)?;
            attention.k = fold_input_scale(attention.k, &attention_scale)?;
            attention.v = fold_input_scale(attention.v, &attention_scale)?;
            let mut mlp = take_mlp(&mut tensors, &p)?;
            mlp.gate = fold_input_scale(mlp.gate, &mlp_scale)?;
            mlp.up = fold_input_scale(mlp.up, &mlp_scale)?;
            language_layers.push(LanguageLayerWeights {
                input_norm_scale: ones(attention_scale.shape().dims())?,
                attention,
                post_attention_norm_scale: ones(mlp_scale.shape().dims())?,
                mlp,
            });
        }

        let weights = Self {
            vision,
            language_layers,
            language_final_norm_scale: add_one(take(
                &mut tensors,
                &format!("{language_prefix}.norm.weight"),
            )?)?,
            lm_head: take_any(
                &mut tensors,
                &[
                    format!("{ROOT}.paligemma.lm_head.weight"),
                    format!("{ROOT}.paligemma.model.language_model.embed_tokens.weight"),
                ],
            )?,
        };

        validate_shapes(config, &weights)?;
        Ok(weights)
    }

}

fn validate_shapes(config: &Pi0FastConfig, weights: &Pi0FastWeights) -> Result<()> {
    let lang = config.language;
    expect(&weights.lm_head, &[config.vocab_size, lang.width], "lm_head")?;
    if weights.language_layers.len() != lang.depth {
        return Err(Error::Other(format!(
            "π0-FAST language depth mismatch: config {}, weights {}",
            lang.depth,
            weights.language_layers.len()
        )));
    }
    if weights.vision.blocks.len() != config.vision_depth {
        return Err(Error::Other(format!(
            "π0-FAST vision depth mismatch: config {}, weights {}",
            config.vision_depth,
            weights.vision.blocks.len()
        )));
    }
    let layer = &weights.language_layers[0];
    expect(
        &layer.attention.q.weight,
        &[lang.width, lang.num_heads * lang.head_dim],
        "layers.0.q_proj",
    )?;
    expect(
        &layer.attention.k.weight,
        &[lang.width, lang.num_kv_heads * lang.head_dim],
        "layers.0.k_proj",
    )?;
    expect(
        &layer.mlp.gate.weight,
        &[lang.width, lang.mlp_dim],
        "layers.0.gate_proj",
    )?;
    Ok(())
}

fn expect(tensor: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "π0-FAST {name}: expected shape {shape:?}, got {:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn take_attention(
    tensors: &mut HashMap<String, Tensor>,
    layer: &str,
) -> Result<GemmaAttentionWeights> {
    let p = format!("{layer}.self_attn");
    Ok(GemmaAttentionWeights {
        q: take_linear(tensors, &format!("{p}.q_proj"), false)?,
        k: take_linear(tensors, &format!("{p}.k_proj"), false)?,
        v: take_linear(tensors, &format!("{p}.v_proj"), false)?,
        output: take_linear(tensors, &format!("{p}.o_proj"), false)?,
    })
}

fn take_mlp(tensors: &mut HashMap<String, Tensor>, layer: &str) -> Result<GemmaMlpWeights> {
    let p = format!("{layer}.mlp");
    Ok(GemmaMlpWeights {
        gate: take_linear(tensors, &format!("{p}.gate_proj"), false)?,
        up: take_linear(tensors, &format!("{p}.up_proj"), false)?,
        down: take_linear(tensors, &format!("{p}.down_proj"), false)?,
    })
}

fn take_layer_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    has_bias: bool,
) -> Result<LinearWeights> {
    let weight = transpose_2d(&take(tensors, &format!("{prefix}.weight"))?)?;
    let bias = has_bias
        .then(|| take(tensors, &format!("{prefix}.bias")))
        .transpose()?;
    Ok(LinearWeights { weight, bias })
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing π0-FAST weight `{name}`")))
}

fn take_any(tensors: &mut HashMap<String, Tensor>, names: &[String]) -> Result<Tensor> {
    for name in names {
        if let Some(tensor) = tensors.remove(name) {
            return Ok(tensor);
        }
    }
    Err(Error::Other(format!(
        "missing π0-FAST weight (accepted aliases: {})",
        names.join(", ")
    )))
}

fn normalize_lerobot_prefix(tensors: &mut HashMap<String, Tensor>) {
    let canonical = format!("{ROOT}.");
    let wrapped = format!("model.{ROOT}.");
    if tensors.keys().any(|name| name.starts_with(&canonical))
        || !tensors.keys().any(|name| name.starts_with(&wrapped))
    {
        return;
    }
    *tensors = std::mem::take(tensors)
        .into_iter()
        .map(|(name, tensor)| {
            let name = name.strip_prefix("model.").unwrap_or(&name).to_owned();
            (name, tensor)
        })
        .collect();
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "π0-FAST linear weight must be 2D, got shape {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::F32 => {
            let src = tensor.as_f32()?;
            let mut dst = vec![0.0; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &dst)
        }
        DType::F16 => {
            let src = tensor.as_f16()?;
            let mut dst = vec![half::f16::ZERO; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f16(vec![cols, rows], &dst)
        }
        DType::BF16 => {
            let src = tensor.as_bf16()?;
            let mut dst = vec![bf16::ZERO; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &dst)
        }
        DType::F8E4M3 => {
            let src = tensor.as_f8_e4m3()?;
            let mut dst = vec![0u8; src.len()];
            for row in 0..rows {
                for col in 0..cols {
                    dst[col * rows + row] = src[row * cols + col];
                }
            }
            Tensor::from_f8_e4m3(vec![cols, rows], &dst)
        }
    }
}

fn add_one(tensor: Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    match tensor.dtype() {
        DType::F32 => {
            let values = tensor.as_f32()?.iter().map(|x| x + 1.0).collect::<Vec<_>>();
            Tensor::from_f32(dims, &values)
        }
        DType::F16 => {
            let values = tensor
                .as_f16()?
                .iter()
                .map(|x| half::f16::from_f32(x.to_f32() + 1.0))
                .collect::<Vec<_>>();
            Tensor::from_f16(dims, &values)
        }
        DType::BF16 => {
            let values = tensor
                .as_bf16()?
                .iter()
                .map(|x| bf16::from_f32(x.to_f32() + 1.0))
                .collect::<Vec<_>>();
            Tensor::from_bf16(dims, &values)
        }
        DType::F8E4M3 => Err(Error::Other(
            "π0-FAST RMSNorm parameters cannot be stored as unscaled FP8".into(),
        )),
    }
}

fn ones(shape: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(shape.to_vec(), &vec![1.0; shape.iter().product()])
}

fn fold_input_scale(mut linear: LinearWeights, scale: &Tensor) -> Result<LinearWeights> {
    let dims = linear.weight.shape().dims();
    if dims.len() != 2 || scale.shape().dims() != [dims[0]] {
        return Err(Error::Other(format!(
            "π0-FAST input-scale fold mismatch: weight {dims:?}, scale {:?}",
            scale.shape().dims()
        )));
    }
    let rows = dims[0];
    let cols = dims[1];
    let mut weight = linear.weight.to_f32_vec()?;
    let scale = scale.to_f32_vec()?;
    for (row, multiplier) in weight.chunks_exact_mut(cols).zip(&scale) {
        for value in row {
            *value *= multiplier;
        }
    }
    linear.weight = Tensor::from_f32(vec![rows, cols], &weight)?;
    if let Some(bias) = linear.bias.take() {
        let mut values = bias.to_f32_vec()?;
        for (value, multiplier) in values.iter_mut().zip(&scale) {
            *value *= multiplier;
        }
        linear.bias = Some(Tensor::from_f32(vec![cols], &values)?);
    }
    Ok(linear)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_round_trips_bf16() {
        let values: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
        let tensor = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let transposed = transpose_2d(&tensor).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        let back = transpose_2d(&transposed).unwrap();
        assert_eq!(back.as_bf16().unwrap(), values.as_slice());
    }
}
