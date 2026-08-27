//! Typed DM05 checkpoint weights.
//!
//! Ordinary PyTorch linear weights are stored `[out, in]` and are physically
//! transposed once to ApxInf's row-major `[in, out]` GEMM layout. The Gemma3
//! multimodal projector is an explicit right-matmul matrix already stored
//! `[1152, 2560]`; it is deliberately not transposed. Gemma3 RMSNorm tensors
//! remain raw delta-gamma values so the CUDA operator can apply `1 + weight`
//! in FP32 before the BF16 output boundary.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};

use super::{Dm05Config, Dm05LayerType, Dm05TextConfig};

#[derive(Debug)]
pub struct LinearWeights {
    /// Physical `[in, out]` matrix consumed by ApxInf GEMM.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct GemmaRmsWeights {
    /// Raw checkpoint delta-gamma. Runtime semantics are `1 + raw_weight` in FP32.
    pub raw_weight: Tensor,
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
    pub patch_embedding: LinearWeights,
    pub position_embedding: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    pub post_layer_norm: LayerNormWeights,
    pub projector_norm: GemmaRmsWeights,
    /// Direct `[vision_width, language_width]` right-matmul matrix.
    pub projector: Tensor,
    /// `[vocab, language_width]`, retained once after exact lm-head comparison.
    pub token_embedding: Tensor,
}

#[derive(Debug)]
pub struct GemmaAttentionWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub q_norm: GemmaRmsWeights,
    pub k_norm: GemmaRmsWeights,
}

#[derive(Debug)]
pub struct GemmaMlpWeights {
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
}

#[derive(Debug)]
pub struct LanguageLayerWeights {
    pub layer_type: Dm05LayerType,
    pub input_norm: GemmaRmsWeights,
    pub attention: GemmaAttentionWeights,
    pub post_attention_norm: GemmaRmsWeights,
    pub pre_feedforward_norm: GemmaRmsWeights,
    pub mlp: GemmaMlpWeights,
    pub post_feedforward_norm: GemmaRmsWeights,
}

#[derive(Debug)]
pub struct ActionLayerWeights {
    pub layer_type: Dm05LayerType,
    /// The static input norm weight is validated and intentionally discarded;
    /// OpenDM's adaptive norm uses only epsilon plus this modulation projection.
    pub input_modulator: LinearWeights,
    pub attention: GemmaAttentionWeights,
    pub post_attention_norm: GemmaRmsWeights,
    /// The static pre-FF norm weight is likewise ignored by reference semantics.
    pub mlp_modulator: LinearWeights,
    pub mlp: GemmaMlpWeights,
    pub post_feedforward_norm: GemmaRmsWeights,
}

#[derive(Debug)]
pub struct Dm05Weights {
    pub vision: VisionWeights,
    pub language_layers: Vec<LanguageLayerWeights>,
    pub language_final_norm: GemmaRmsWeights,
    pub action_layers: Vec<ActionLayerWeights>,
    pub action_final_modulator: LinearWeights,
    pub action_in: LinearWeights,
    pub action_out: LinearWeights,
    pub time_mlp_in: LinearWeights,
    pub time_mlp_out: LinearWeights,
}

impl Dm05Weights {
    pub fn from_safetensors(config: &Dm05Config, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load DM05 SafeTensors: {error}")))?;
        Self::from_map(config, tensors)
    }

    pub fn from_map(config: &Dm05Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        require_all_bf16(&tensors)?;

        let vision_prefix = "model.vlm.model.vision_tower.vision_model";
        let mut vision_blocks = Vec::with_capacity(config.vision.depth);
        for layer in 0..config.vision.depth {
            let prefix = format!("{vision_prefix}.encoder.layers.{layer}");
            vision_blocks.push(VisionBlockWeights {
                norm1: take_layer_norm(&mut tensors, &format!("{prefix}.layer_norm1"))?,
                q: take_linear(&mut tensors, &format!("{prefix}.self_attn.q_proj"), true)?,
                k: take_linear(&mut tensors, &format!("{prefix}.self_attn.k_proj"), true)?,
                v: take_linear(&mut tensors, &format!("{prefix}.self_attn.v_proj"), true)?,
                output: take_linear(&mut tensors, &format!("{prefix}.self_attn.out_proj"), true)?,
                norm2: take_layer_norm(&mut tensors, &format!("{prefix}.layer_norm2"))?,
                fc1: take_linear(&mut tensors, &format!("{prefix}.mlp.fc1"), true)?,
                fc2: take_linear(&mut tensors, &format!("{prefix}.mlp.fc2"), true)?,
            });
        }

        let patch_name = format!("{vision_prefix}.embeddings.patch_embedding.weight");
        let patch = take(&mut tensors, &patch_name)?;
        expect_shape(
            &patch_name,
            &patch,
            &[
                config.vision.width,
                3,
                config.vision.patch_size,
                config.vision.patch_size,
            ],
        )?;
        let patch = patch.reshape(vec![
            config.vision.width,
            3 * config.vision.patch_size * config.vision.patch_size,
        ])?;
        let patch_embedding = LinearWeights {
            weight: transpose_2d(&patch)?,
            bias: Some(take(
                &mut tensors,
                &format!("{vision_prefix}.embeddings.patch_embedding.bias"),
            )?),
        };

        let embedding = take(
            &mut tensors,
            "model.vlm.model.language_model.embed_tokens.weight",
        )?;
        let lm_head = take(&mut tensors, "model.vlm.lm_head.weight")?;
        expect_shape(
            "model.vlm.model.language_model.embed_tokens.weight",
            &embedding,
            &[config.language.vocab_size, config.language.width],
        )?;
        expect_shape(
            "model.vlm.lm_head.weight",
            &lm_head,
            &[config.language.vocab_size, config.language.width],
        )?;
        if embedding.as_bf16()? != lm_head.as_bf16()? {
            return Err(Error::Other(
                "DM05 lm_head and token embedding must be byte-identical".into(),
            ));
        }

        let projector_name = "model.vlm.model.multi_modal_projector.mm_input_projection_weight";
        let projector = take(&mut tensors, projector_name)?;
        expect_shape(
            projector_name,
            &projector,
            &[config.vision.width, config.language.width],
        )?;
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
            projector_norm: take_rms(
                &mut tensors,
                "model.vlm.model.multi_modal_projector.mm_soft_emb_norm",
            )?,
            projector,
            token_embedding: embedding,
        };

        let language_prefix = "model.vlm.model.language_model";
        let mut language_layers = Vec::with_capacity(config.language.depth);
        for (layer, layer_type) in config.language.layer_types.iter().copied().enumerate() {
            let prefix = format!("{language_prefix}.layers.{layer}");
            language_layers.push(LanguageLayerWeights {
                layer_type,
                input_norm: take_rms(&mut tensors, &format!("{prefix}.input_layernorm"))?,
                attention: take_attention(&mut tensors, &prefix)?,
                post_attention_norm: take_rms(
                    &mut tensors,
                    &format!("{prefix}.post_attention_layernorm"),
                )?,
                pre_feedforward_norm: take_rms(
                    &mut tensors,
                    &format!("{prefix}.pre_feedforward_layernorm"),
                )?,
                mlp: take_mlp(&mut tensors, &prefix)?,
                post_feedforward_norm: take_rms(
                    &mut tensors,
                    &format!("{prefix}.post_feedforward_layernorm"),
                )?,
            });
        }
        let language_final_norm = take_rms(&mut tensors, &format!("{language_prefix}.norm"))?;

        let action_prefix = "model.action_expert";
        let mut action_layers = Vec::with_capacity(config.action_expert.depth);
        for (layer, layer_type) in config.action_expert.layer_types.iter().copied().enumerate() {
            let prefix = format!("{action_prefix}.layers.{layer}");
            take_ignored_rms(
                &mut tensors,
                &format!("{prefix}.input_layernorm"),
                config.action_expert.width,
            )?;
            take_ignored_rms(
                &mut tensors,
                &format!("{prefix}.pre_feedforward_layernorm"),
                config.action_expert.width,
            )?;
            action_layers.push(ActionLayerWeights {
                layer_type,
                input_modulator: take_linear(
                    &mut tensors,
                    &format!("{action_prefix}.input_time_modulators.{layer}"),
                    true,
                )?,
                attention: take_attention(&mut tensors, &prefix)?,
                post_attention_norm: take_rms(
                    &mut tensors,
                    &format!("{prefix}.post_attention_layernorm"),
                )?,
                mlp_modulator: take_linear(
                    &mut tensors,
                    &format!("{action_prefix}.mlp_time_modulators.{layer}"),
                    true,
                )?,
                mlp: take_mlp(&mut tensors, &prefix)?,
                post_feedforward_norm: take_rms(
                    &mut tensors,
                    &format!("{prefix}.post_feedforward_layernorm"),
                )?,
            });
        }
        take_ignored_rms(
            &mut tensors,
            &format!("{action_prefix}.norm"),
            config.action_expert.width,
        )?;

        let weights = Self {
            vision,
            language_layers,
            language_final_norm,
            action_layers,
            action_final_modulator: take_linear(
                &mut tensors,
                &format!("{action_prefix}.final_time_modulator"),
                true,
            )?,
            action_in: take_linear(&mut tensors, "model.action_in_proj", true)?,
            action_out: take_linear(&mut tensors, "model.action_out_proj", true)?,
            time_mlp_in: take_linear(&mut tensors, "model.time_mlp_in", true)?,
            time_mlp_out: take_linear(&mut tensors, "model.time_mlp_out", true)?,
        };

        if !tensors.is_empty() {
            let mut names = tensors.into_keys().collect::<Vec<_>>();
            names.sort();
            return Err(Error::Other(format!(
                "DM05 checkpoint has {} unowned tensors; first: {:?}",
                names.len(),
                names.iter().take(8).collect::<Vec<_>>()
            )));
        }
        validate_shapes(config, &weights)?;
        Ok(weights)
    }
}

fn take_attention(
    tensors: &mut HashMap<String, Tensor>,
    layer: &str,
) -> Result<GemmaAttentionWeights> {
    let prefix = format!("{layer}.self_attn");
    Ok(GemmaAttentionWeights {
        q: take_linear(tensors, &format!("{prefix}.q_proj"), false)?,
        k: take_linear(tensors, &format!("{prefix}.k_proj"), false)?,
        v: take_linear(tensors, &format!("{prefix}.v_proj"), false)?,
        output: take_linear(tensors, &format!("{prefix}.o_proj"), false)?,
        q_norm: take_rms(tensors, &format!("{prefix}.q_norm"))?,
        k_norm: take_rms(tensors, &format!("{prefix}.k_norm"))?,
    })
}

fn take_mlp(tensors: &mut HashMap<String, Tensor>, layer: &str) -> Result<GemmaMlpWeights> {
    let prefix = format!("{layer}.mlp");
    Ok(GemmaMlpWeights {
        gate: take_linear(tensors, &format!("{prefix}.gate_proj"), false)?,
        up: take_linear(tensors, &format!("{prefix}.up_proj"), false)?,
        down: take_linear(tensors, &format!("{prefix}.down_proj"), false)?,
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

fn take_rms(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<GemmaRmsWeights> {
    Ok(GemmaRmsWeights {
        raw_weight: take(tensors, &format!("{prefix}.weight"))?,
    })
}

fn take_ignored_rms(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    width: usize,
) -> Result<()> {
    let name = format!("{prefix}.weight");
    let tensor = take(tensors, &name)?;
    expect_shape(&name, &tensor, &[width])
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
        .ok_or_else(|| Error::Other(format!("missing DM05 weight `{name}`")))
}

fn require_all_bf16(tensors: &HashMap<String, Tensor>) -> Result<()> {
    if let Some((name, tensor)) = tensors
        .iter()
        .find(|(_, tensor)| tensor.dtype() != DType::BF16)
    {
        return Err(Error::Other(format!(
            "DM05 checkpoint tensor `{name}` must be BF16, got {}",
            tensor.dtype()
        )));
    }
    Ok(())
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if tensor.dtype() != DType::BF16 || dims.len() != 2 {
        return Err(Error::Other(format!(
            "DM05 linear weight must be BF16 2D, got {} {dims:?}",
            tensor.dtype()
        )));
    }
    let rows = dims[0];
    let cols = dims[1];
    let source = tensor.as_bf16()?;
    let mut transposed = vec![half::bf16::ZERO; source.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = source[row * cols + col];
        }
    }
    Tensor::from_bf16(vec![cols, rows], &transposed)
}

fn expect_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "DM05 weight `{name}` expected BF16 {expected:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn validate_linear(
    label: &str,
    weights: &LinearWeights,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<()> {
    expect_shape(
        &format!("{label}.weight"),
        &weights.weight,
        &[input, output],
    )?;
    match (&weights.bias, bias) {
        (Some(value), true) => expect_shape(&format!("{label}.bias"), value, &[output]),
        (None, false) => Ok(()),
        _ => Err(Error::Other(format!("DM05 {label} bias contract mismatch"))),
    }
}

fn validate_rms(label: &str, weights: &GemmaRmsWeights, width: usize) -> Result<()> {
    expect_shape(label, &weights.raw_weight, &[width])
}

fn validate_attention(
    label: &str,
    config: &Dm05TextConfig,
    weights: &GemmaAttentionWeights,
) -> Result<()> {
    let query = config.num_heads * config.head_dim;
    let kv = config.num_kv_heads * config.head_dim;
    validate_linear(
        &format!("{label}.q"),
        &weights.q,
        config.width,
        query,
        false,
    )?;
    validate_linear(&format!("{label}.k"), &weights.k, config.width, kv, false)?;
    validate_linear(&format!("{label}.v"), &weights.v, config.width, kv, false)?;
    validate_linear(
        &format!("{label}.output"),
        &weights.output,
        query,
        config.width,
        false,
    )?;
    validate_rms(&format!("{label}.q_norm"), &weights.q_norm, config.head_dim)?;
    validate_rms(&format!("{label}.k_norm"), &weights.k_norm, config.head_dim)
}

fn validate_mlp(label: &str, config: &Dm05TextConfig, weights: &GemmaMlpWeights) -> Result<()> {
    validate_linear(
        &format!("{label}.gate"),
        &weights.gate,
        config.width,
        config.mlp_dim,
        false,
    )?;
    validate_linear(
        &format!("{label}.up"),
        &weights.up,
        config.width,
        config.mlp_dim,
        false,
    )?;
    validate_linear(
        &format!("{label}.down"),
        &weights.down,
        config.mlp_dim,
        config.width,
        false,
    )
}

fn validate_shapes(config: &Dm05Config, weights: &Dm05Weights) -> Result<()> {
    let vision = &config.vision;
    validate_linear(
        "vision.patch",
        &weights.vision.patch_embedding,
        3 * vision.patch_size * vision.patch_size,
        vision.width,
        true,
    )?;
    expect_shape(
        "vision.position_embedding",
        &weights.vision.position_embedding,
        &[config.patches_per_view(), vision.width],
    )?;
    if weights.vision.blocks.len() != vision.depth {
        return Err(Error::Other("DM05 vision depth mismatch".into()));
    }
    for (index, block) in weights.vision.blocks.iter().enumerate() {
        for (suffix, norm) in [("norm1", &block.norm1), ("norm2", &block.norm2)] {
            expect_shape(
                &format!("vision.{index}.{suffix}.weight"),
                &norm.weight,
                &[vision.width],
            )?;
            expect_shape(
                &format!("vision.{index}.{suffix}.bias"),
                &norm.bias,
                &[vision.width],
            )?;
        }
        for (suffix, linear, input, output) in [
            ("q", &block.q, vision.width, vision.width),
            ("k", &block.k, vision.width, vision.width),
            ("v", &block.v, vision.width, vision.width),
            ("output", &block.output, vision.width, vision.width),
            ("fc1", &block.fc1, vision.width, vision.mlp_dim),
            ("fc2", &block.fc2, vision.mlp_dim, vision.width),
        ] {
            validate_linear(
                &format!("vision.{index}.{suffix}"),
                linear,
                input,
                output,
                true,
            )?;
        }
    }
    expect_shape(
        "vision.post_norm.weight",
        &weights.vision.post_layer_norm.weight,
        &[vision.width],
    )?;
    expect_shape(
        "vision.post_norm.bias",
        &weights.vision.post_layer_norm.bias,
        &[vision.width],
    )?;
    validate_rms(
        "vision.projector_norm",
        &weights.vision.projector_norm,
        vision.width,
    )?;
    expect_shape(
        "vision.projector",
        &weights.vision.projector,
        &[vision.width, config.language.width],
    )?;
    expect_shape(
        "vision.token_embedding",
        &weights.vision.token_embedding,
        &[config.language.vocab_size, config.language.width],
    )?;

    if weights.language_layers.len() != config.language.depth
        || weights.action_layers.len() != config.action_expert.depth
    {
        return Err(Error::Other("DM05 transformer depth mismatch".into()));
    }
    for (index, layer) in weights.language_layers.iter().enumerate() {
        if layer.layer_type != config.language.layer_types[index] {
            return Err(Error::Other(format!(
                "DM05 language layer {index} type mismatch"
            )));
        }
        validate_rms(
            &format!("language.{index}.input"),
            &layer.input_norm,
            config.language.width,
        )?;
        validate_attention(
            &format!("language.{index}.attention"),
            &config.language,
            &layer.attention,
        )?;
        validate_rms(
            &format!("language.{index}.post_attention"),
            &layer.post_attention_norm,
            config.language.width,
        )?;
        validate_rms(
            &format!("language.{index}.pre_ff"),
            &layer.pre_feedforward_norm,
            config.language.width,
        )?;
        validate_mlp(
            &format!("language.{index}.mlp"),
            &config.language,
            &layer.mlp,
        )?;
        validate_rms(
            &format!("language.{index}.post_ff"),
            &layer.post_feedforward_norm,
            config.language.width,
        )?;
    }
    validate_rms(
        "language.final_norm",
        &weights.language_final_norm,
        config.language.width,
    )?;

    for (index, layer) in weights.action_layers.iter().enumerate() {
        if layer.layer_type != config.action_expert.layer_types[index] {
            return Err(Error::Other(format!(
                "DM05 action layer {index} type mismatch"
            )));
        }
        validate_linear(
            &format!("action.{index}.input_modulator"),
            &layer.input_modulator,
            config.action_expert.width,
            3 * config.action_expert.width,
            true,
        )?;
        validate_attention(
            &format!("action.{index}.attention"),
            &config.action_expert,
            &layer.attention,
        )?;
        validate_rms(
            &format!("action.{index}.post_attention"),
            &layer.post_attention_norm,
            config.action_expert.width,
        )?;
        validate_linear(
            &format!("action.{index}.mlp_modulator"),
            &layer.mlp_modulator,
            config.action_expert.width,
            3 * config.action_expert.width,
            true,
        )?;
        validate_mlp(
            &format!("action.{index}.mlp"),
            &config.action_expert,
            &layer.mlp,
        )?;
        validate_rms(
            &format!("action.{index}.post_ff"),
            &layer.post_feedforward_norm,
            config.action_expert.width,
        )?;
    }
    validate_linear(
        "action.final_modulator",
        &weights.action_final_modulator,
        config.action_expert.width,
        3 * config.action_expert.width,
        true,
    )?;
    validate_linear(
        "action.in",
        &weights.action_in,
        config.action_dim,
        config.action_expert.width,
        true,
    )?;
    validate_linear(
        "action.out",
        &weights.action_out,
        config.action_expert.width,
        config.action_dim,
        true,
    )?;
    validate_linear(
        "time.in",
        &weights.time_mlp_in,
        config.action_expert.width,
        config.action_expert.width,
        true,
    )?;
    validate_linear(
        "time.out",
        &weights.time_mlp_out,
        config.action_expert.width,
        config.action_expert.width,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn transpose_bf16_is_physical() {
        let source = Tensor::from_bf16(
            vec![2, 3],
            &[
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
                bf16::from_f32(5.0),
                bf16::from_f32(6.0),
            ],
        )
        .unwrap();
        let transposed = transpose_2d(&source).unwrap();
        assert_eq!(transposed.shape().dims(), [3, 2]);
        let values = transposed
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(values, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn direct_projector_matrix_is_not_transposed() {
        let projector = Tensor::from_bf16(
            vec![2, 3],
            &[
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
                bf16::from_f32(5.0),
                bf16::from_f32(6.0),
            ],
        )
        .unwrap();
        expect_shape("projector", &projector, &[2, 3]).unwrap();
        assert_eq!(projector.as_bf16().unwrap()[1].to_f32(), 2.0);
    }

    #[test]
    fn tied_embedding_check_is_raw_bf16_not_float_tolerance() {
        let left = Tensor::from_bf16(vec![2], &[bf16::from_bits(1), bf16::from_bits(2)]).unwrap();
        let right = Tensor::from_bf16(vec![2], &[bf16::from_bits(1), bf16::from_bits(3)]).unwrap();
        assert_ne!(left.as_bf16().unwrap(), right.as_bf16().unwrap());
    }
}
