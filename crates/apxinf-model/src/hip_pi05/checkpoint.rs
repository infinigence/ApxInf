//! A LeRobot/OpenPI π0.5 SafeTensors checkpoint, read into host tensors.
//!
//! Owned by `hip_pi05` (the CUDA implementation has its own reader in
//! `pi05/weights/host.rs`). Checkpoints store Hugging Face `[out, in]`
//! projections; every matrix is transposed here to the row-major `[in, out]`
//! the GEMMs consume, and Gemma's learned RMSNorm offsets become `1 + w`.
//!
//! This port then folds each language layer's two RMSNorm scales into the
//! weights that consume the normalized activations (Q/K/V, gate/up), so the
//! runtime norms multiply by one. That is this port's choice — it is what the
//! accuracy results were measured with — and can change here without
//! touching any other implementation.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};
use half::bf16;

use super::config::{Config, GemmaConfig};

const ROOT: &str = "paligemma_with_expert";

/// Row-major `[in, out]` projection and its optional bias.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct VisionBlock {
    pub norm1: LayerNorm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
}

pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct LanguageLayer {
    /// The RMSNorm scale after folding: ones.
    pub input_norm_scale: Tensor,
    pub attention: Attention,
    pub post_attention_norm_scale: Tensor,
    pub mlp: Mlp,
}

pub struct ActionLayer {
    /// Conditioning → `[3 * width]` scale/shift/gate, per norm.
    pub input_modulation: Linear,
    pub attention: Attention,
    pub post_attention_modulation: Linear,
    pub mlp: Mlp,
}

pub struct Checkpoint {
    /// Flattened patch convolution, `[3 * patch², vision_width]`.
    pub patch_embedding: Linear,
    pub position_embedding: Tensor,
    pub vision_blocks: Vec<VisionBlock>,
    pub vision_post_norm: LayerNorm,
    pub multimodal_projector: Linear,
    /// Tied PaliGemma embedding, `[vocab, language_width]`.
    pub token_embedding: Tensor,
    pub language_layers: Vec<LanguageLayer>,
    pub action_layers: Vec<ActionLayer>,
    pub action_final_modulation: Linear,
    pub action_in: Linear,
    pub action_out: Linear,
    pub time_mlp_in: Linear,
    pub time_mlp_out: Linear,
}

impl Checkpoint {
    /// Load a single-file or sharded SafeTensors checkpoint.
    pub fn from_safetensors(config: &Config, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load π0.5 SafeTensors: {error}")))?;
        Self::from_map(config, tensors)
    }

    /// Take every tensor inference needs from a checkpoint map and check its
    /// shape. Tensors inference does not use (the action LM head) may remain.
    pub fn from_map(config: &Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        config.validate()?;
        strip_lerobot_wrapper(&mut tensors);
        let t = &mut tensors;

        let vision = format!("{ROOT}.paligemma.model.vision_tower.vision_model");
        let mut vision_blocks = Vec::with_capacity(config.vision_depth);
        for layer in 0..config.vision_depth {
            let p = format!("{vision}.encoder.layers.{layer}");
            vision_blocks.push(VisionBlock {
                norm1: layer_norm(t, &format!("{p}.layer_norm1"))?,
                q: linear(t, &format!("{p}.self_attn.q_proj"), true)?,
                k: linear(t, &format!("{p}.self_attn.k_proj"), true)?,
                v: linear(t, &format!("{p}.self_attn.v_proj"), true)?,
                output: linear(t, &format!("{p}.self_attn.out_proj"), true)?,
                norm2: layer_norm(t, &format!("{p}.layer_norm2"))?,
                fc1: linear(t, &format!("{p}.mlp.fc1"), true)?,
                fc2: linear(t, &format!("{p}.mlp.fc2"), true)?,
            });
        }

        // The patch convolution, flattened channel-major to [vision_width, 3·p²].
        let name = format!("{vision}.embeddings.patch_embedding.weight");
        let patch = take(t, &name)?;
        if patch.numel() != config.vision_width * config.patch_width() {
            return Err(Error::Other(format!(
                "{name}: expected {} elements, got {}",
                config.vision_width * config.patch_width(),
                patch.numel()
            )));
        }
        let patch = patch.reshape(vec![config.vision_width, config.patch_width()])?;
        let patch_embedding = Linear {
            weight: transpose(&patch)?,
            bias: Some(take(
                t,
                &format!("{vision}.embeddings.patch_embedding.bias"),
            )?),
        };

        let language = format!("{ROOT}.paligemma.model.language_model");
        let mut language_layers = Vec::with_capacity(config.language.depth);
        for layer in 0..config.language.depth {
            let p = format!("{language}.layers.{layer}");
            let attention_scale = add_one(take(t, &format!("{p}.input_layernorm.weight"))?)?;
            let mlp_scale = add_one(take(t, &format!("{p}.post_attention_layernorm.weight"))?)?;
            let mut attention = attention(t, &p)?;
            for projection in [&mut attention.q, &mut attention.k, &mut attention.v] {
                fold_input_scale(projection, &attention_scale)?;
            }
            let mut mlp = mlp(t, &p)?;
            for projection in [&mut mlp.gate, &mut mlp.up] {
                fold_input_scale(projection, &mlp_scale)?;
            }
            language_layers.push(LanguageLayer {
                input_norm_scale: ones(config.language.width)?,
                attention,
                post_attention_norm_scale: ones(config.language.width)?,
                mlp,
            });
        }

        let action = format!("{ROOT}.gemma_expert.model");
        let mut action_layers = Vec::with_capacity(config.action_expert.depth);
        for layer in 0..config.action_expert.depth {
            let p = format!("{action}.layers.{layer}");
            action_layers.push(ActionLayer {
                input_modulation: linear(t, &format!("{p}.input_layernorm.dense"), true)?,
                attention: attention(t, &p)?,
                post_attention_modulation: linear(
                    t,
                    &format!("{p}.post_attention_layernorm.dense"),
                    true,
                )?,
                mlp: mlp(t, &p)?,
            });
        }

        let checkpoint = Self {
            patch_embedding,
            position_embedding: take(t, &format!("{vision}.embeddings.position_embedding.weight"))?,
            vision_blocks,
            vision_post_norm: layer_norm(t, &format!("{vision}.post_layernorm"))?,
            multimodal_projector: linear(
                t,
                &format!("{ROOT}.paligemma.model.multi_modal_projector.linear"),
                true,
            )?,
            token_embedding: take_any(
                t,
                &[
                    format!("{language}.embed_tokens.weight"),
                    format!("{ROOT}.paligemma.lm_head.weight"),
                ],
            )?,
            language_layers,
            action_layers,
            action_final_modulation: linear(t, &format!("{action}.norm.dense"), true)?,
            action_in: linear(t, "action_in_proj", true)?,
            action_out: linear(t, "action_out_proj", true)?,
            time_mlp_in: linear(t, "time_mlp_in", true)?,
            time_mlp_out: linear(t, "time_mlp_out", true)?,
        };
        checkpoint.validate(config)?;
        Ok(checkpoint)
    }

    /// Deterministic random weights at final orientation, for benchmarks that
    /// measure the engine without a checkpoint. Small values keep deep BF16
    /// stacks finite; norm scales are ones, LayerNorm is identity.
    pub fn synthetic(config: &Config, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = Lcg(seed ^ 0x9E37_79B9_7F4A_7C15);
        let rng = &mut rng;
        let (lang, act) = (&config.language, &config.action_expert);
        let vw = config.vision_width;
        let vision_blocks = (0..config.vision_depth)
            .map(|_| -> Result<VisionBlock> {
                Ok(VisionBlock {
                    norm1: identity_layer_norm(vw)?,
                    q: random_linear(rng, vw, vw, true)?,
                    k: random_linear(rng, vw, vw, true)?,
                    v: random_linear(rng, vw, vw, true)?,
                    output: random_linear(rng, vw, vw, true)?,
                    norm2: identity_layer_norm(vw)?,
                    fc1: random_linear(rng, vw, config.vision_mlp_dim, true)?,
                    fc2: random_linear(rng, config.vision_mlp_dim, vw, true)?,
                })
            })
            .collect::<Result<_>>()?;
        let patch_embedding = random_linear(rng, config.patch_width(), vw, true)?;
        let position_embedding = random(rng, vec![config.patches_per_view(), vw])?;
        let multimodal_projector = random_linear(rng, vw, lang.width, true)?;
        let token_embedding = random(rng, vec![config.vocab_size, lang.width])?;
        let language_layers = (0..lang.depth)
            .map(|_| -> Result<LanguageLayer> {
                Ok(LanguageLayer {
                    input_norm_scale: ones(lang.width)?,
                    attention: random_attention(rng, lang)?,
                    post_attention_norm_scale: ones(lang.width)?,
                    mlp: random_mlp(rng, lang)?,
                })
            })
            .collect::<Result<_>>()?;
        let action_layers = (0..act.depth)
            .map(|_| -> Result<ActionLayer> {
                Ok(ActionLayer {
                    input_modulation: random_linear(rng, act.width, 3 * act.width, true)?,
                    attention: random_attention(rng, act)?,
                    post_attention_modulation: random_linear(rng, act.width, 3 * act.width, true)?,
                    mlp: random_mlp(rng, act)?,
                })
            })
            .collect::<Result<_>>()?;
        let checkpoint = Self {
            patch_embedding,
            position_embedding,
            vision_blocks,
            vision_post_norm: identity_layer_norm(vw)?,
            multimodal_projector,
            token_embedding,
            language_layers,
            action_layers,
            action_final_modulation: random_linear(rng, act.width, 3 * act.width, true)?,
            action_in: random_linear(rng, config.action_dim, act.width, true)?,
            action_out: random_linear(rng, act.width, config.action_dim, true)?,
            time_mlp_in: random_linear(rng, act.width, act.width, true)?,
            time_mlp_out: random_linear(rng, act.width, act.width, true)?,
        };
        checkpoint.validate(config)?;
        Ok(checkpoint)
    }

    fn validate(&self, config: &Config) -> Result<()> {
        let (lang, act) = (&config.language, &config.action_expert);
        let vw = config.vision_width;
        shape(
            "token_embedding",
            &self.token_embedding,
            &[config.vocab_size, lang.width],
        )?;
        shape(
            "position_embedding",
            &self.position_embedding,
            &[config.patches_per_view(), vw],
        )?;
        linear_shape(
            "patch_embedding",
            &self.patch_embedding,
            config.patch_width(),
            vw,
        )?;
        linear_shape(
            "multimodal_projector",
            &self.multimodal_projector,
            vw,
            lang.width,
        )?;
        linear_shape(
            "action_in_proj",
            &self.action_in,
            config.action_dim,
            act.width,
        )?;
        linear_shape(
            "action_out_proj",
            &self.action_out,
            act.width,
            config.action_dim,
        )?;
        linear_shape("time_mlp_in", &self.time_mlp_in, act.width, act.width)?;
        linear_shape("time_mlp_out", &self.time_mlp_out, act.width, act.width)?;
        linear_shape(
            "action_final_norm",
            &self.action_final_modulation,
            act.width,
            3 * act.width,
        )?;
        count("vision", self.vision_blocks.len(), config.vision_depth)?;
        for (i, block) in self.vision_blocks.iter().enumerate() {
            for (part, l, input, output) in [
                ("q", &block.q, vw, vw),
                ("k", &block.k, vw, vw),
                ("v", &block.v, vw, vw),
                ("output", &block.output, vw, vw),
                ("fc1", &block.fc1, vw, config.vision_mlp_dim),
                ("fc2", &block.fc2, config.vision_mlp_dim, vw),
            ] {
                linear_shape(&format!("vision.{i}.{part}"), l, input, output)?;
            }
        }
        count("language", self.language_layers.len(), lang.depth)?;
        for (i, layer) in self.language_layers.iter().enumerate() {
            gemma_shapes(&format!("language.{i}"), lang, &layer.attention, &layer.mlp)?;
        }
        count("action", self.action_layers.len(), act.depth)?;
        for (i, layer) in self.action_layers.iter().enumerate() {
            gemma_shapes(&format!("action.{i}"), act, &layer.attention, &layer.mlp)?;
            linear_shape(
                &format!("action.{i}.input_norm"),
                &layer.input_modulation,
                act.width,
                3 * act.width,
            )?;
            linear_shape(
                &format!("action.{i}.post_attention_norm"),
                &layer.post_attention_modulation,
                act.width,
                3 * act.width,
            )?;
        }
        Ok(())
    }
}

fn gemma_shapes(name: &str, g: &GemmaConfig, attention: &Attention, mlp: &Mlp) -> Result<()> {
    let (q, kv) = (g.num_heads * g.head_dim, g.num_kv_heads * g.head_dim);
    linear_shape(&format!("{name}.q"), &attention.q, g.width, q)?;
    linear_shape(&format!("{name}.k"), &attention.k, g.width, kv)?;
    linear_shape(&format!("{name}.v"), &attention.v, g.width, kv)?;
    linear_shape(&format!("{name}.output"), &attention.output, q, g.width)?;
    linear_shape(&format!("{name}.gate"), &mlp.gate, g.width, g.mlp_dim)?;
    linear_shape(&format!("{name}.up"), &mlp.up, g.width, g.mlp_dim)?;
    linear_shape(&format!("{name}.down"), &mlp.down, g.mlp_dim, g.width)
}

fn count(name: &str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "π0.5 {name}: expected {expected} layers, got {got}"
        )))
    }
}

fn linear_shape(name: &str, linear: &Linear, input: usize, output: usize) -> Result<()> {
    shape(name, &linear.weight, &[input, output])?;
    match &linear.bias {
        Some(bias) => shape(&format!("{name}.bias"), bias, &[output]),
        None => Ok(()),
    }
}

fn shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.shape().dims() == expected {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "π0.5 {name}: expected shape {expected:?}, got {:?}",
            tensor.shape().dims()
        )))
    }
}

fn attention(t: &mut HashMap<String, Tensor>, layer: &str) -> Result<Attention> {
    let p = format!("{layer}.self_attn");
    Ok(Attention {
        q: linear(t, &format!("{p}.q_proj"), false)?,
        k: linear(t, &format!("{p}.k_proj"), false)?,
        v: linear(t, &format!("{p}.v_proj"), false)?,
        output: linear(t, &format!("{p}.o_proj"), false)?,
    })
}

fn mlp(t: &mut HashMap<String, Tensor>, layer: &str) -> Result<Mlp> {
    let p = format!("{layer}.mlp");
    Ok(Mlp {
        gate: linear(t, &format!("{p}.gate_proj"), false)?,
        up: linear(t, &format!("{p}.up_proj"), false)?,
        down: linear(t, &format!("{p}.down_proj"), false)?,
    })
}

fn layer_norm(t: &mut HashMap<String, Tensor>, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: take(t, &format!("{prefix}.weight"))?,
        bias: take(t, &format!("{prefix}.bias"))?,
    })
}

fn linear(t: &mut HashMap<String, Tensor>, prefix: &str, has_bias: bool) -> Result<Linear> {
    Ok(Linear {
        weight: transpose(&take(t, &format!("{prefix}.weight"))?)?,
        bias: has_bias
            .then(|| take(t, &format!("{prefix}.bias")))
            .transpose()?,
    })
}

fn take(t: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    t.remove(name)
        .ok_or_else(|| Error::Other(format!("missing π0.5 weight `{name}`")))
}

fn take_any(t: &mut HashMap<String, Tensor>, names: &[String]) -> Result<Tensor> {
    names.iter().find_map(|name| t.remove(name)).ok_or_else(|| {
        Error::Other(format!(
            "missing π0.5 weight (accepted aliases: {})",
            names.join(", ")
        ))
    })
}

/// LeRobot sometimes wraps every key in `model.`; strip it if, and only if,
/// no key already uses the canonical root.
fn strip_lerobot_wrapper(t: &mut HashMap<String, Tensor>) {
    let canonical = format!("{ROOT}.");
    let wrapped = format!("model.{ROOT}.");
    if t.keys().any(|k| k.starts_with(&canonical)) || !t.keys().any(|k| k.starts_with(&wrapped)) {
        return;
    }
    *t = std::mem::take(t)
        .into_iter()
        .map(|(k, v)| (k.strip_prefix("model.").unwrap_or(&k).to_owned(), v))
        .collect();
}

/// Physically transpose a 2-D tensor, keeping its float dtype.
fn transpose(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "π0.5 linear weight must be 2-D, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    fn flip<T: Copy + Default>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
        let mut dst = vec![T::default(); src.len()];
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        dst
    }
    let shape = vec![cols, rows];
    match tensor.dtype() {
        DType::F32 => Tensor::from_f32(shape, &flip(tensor.as_f32()?, rows, cols)),
        DType::F16 => Tensor::from_f16(shape, &flip(tensor.as_f16()?, rows, cols)),
        DType::BF16 => Tensor::from_bf16(shape, &flip(tensor.as_bf16()?, rows, cols)),
        dtype => Err(Error::Other(format!(
            "π0.5 on HIP reads F32/F16/BF16 checkpoints, got {dtype}"
        ))),
    }
}

/// Gemma's RMSNorm stores `w`; the scale applied is `1 + w`, in F32.
fn add_one(tensor: Tensor) -> Result<Tensor> {
    let values: Vec<f32> = tensor.to_f32_vec()?.into_iter().map(|w| w + 1.0).collect();
    Tensor::from_f32(tensor.shape().dims().to_vec(), &values)
}

fn ones(width: usize) -> Result<Tensor> {
    Tensor::from_f32(vec![width], &vec![1.0; width])
}

/// Scale row `i` of an `[in, out]` weight by `scale[i]`, in F32.
fn fold_input_scale(linear: &mut Linear, scale: &Tensor) -> Result<()> {
    let dims = linear.weight.shape().dims().to_vec();
    if dims.len() != 2 || scale.shape().dims() != [dims[0]] {
        return Err(Error::Other(format!(
            "π0.5 norm fold: weight {dims:?} and scale {:?} disagree",
            scale.shape().dims()
        )));
    }
    let mut weight = linear.weight.to_f32_vec()?;
    for (row, multiplier) in weight.chunks_exact_mut(dims[1]).zip(scale.to_f32_vec()?) {
        row.iter_mut().for_each(|value| *value *= multiplier);
    }
    linear.weight = Tensor::from_f32(dims, &weight)?;
    Ok(())
}

/// Seeded LCG in [-0.02, 0.02), for synthetic weights only.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.0 >> 33) as u32 as f32 / (1u32 << 31) as f32;
        (unit * 2.0 - 1.0) * 0.02
    }
}

fn random(rng: &mut Lcg, shape: Vec<usize>) -> Result<Tensor> {
    let values: Vec<bf16> = (0..shape.iter().product())
        .map(|_| bf16::from_f32(rng.next()))
        .collect();
    Tensor::from_bf16(shape, &values)
}

fn random_linear(rng: &mut Lcg, input: usize, output: usize, bias: bool) -> Result<Linear> {
    Ok(Linear {
        weight: random(rng, vec![input, output])?,
        bias: bias.then(|| random(rng, vec![output])).transpose()?,
    })
}

fn random_attention(rng: &mut Lcg, g: &GemmaConfig) -> Result<Attention> {
    let (q, kv) = (g.num_heads * g.head_dim, g.num_kv_heads * g.head_dim);
    Ok(Attention {
        q: random_linear(rng, g.width, q, false)?,
        k: random_linear(rng, g.width, kv, false)?,
        v: random_linear(rng, g.width, kv, false)?,
        output: random_linear(rng, q, g.width, false)?,
    })
}

fn random_mlp(rng: &mut Lcg, g: &GemmaConfig) -> Result<Mlp> {
    Ok(Mlp {
        gate: random_linear(rng, g.width, g.mlp_dim, false)?,
        up: random_linear(rng, g.width, g.mlp_dim, false)?,
        down: random_linear(rng, g.mlp_dim, g.width, false)?,
    })
}

fn identity_layer_norm(width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: Tensor::from_f32(vec![width], &vec![1.0; width])?,
        bias: Tensor::from_f32(vec![width], &vec![0.0; width])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_is_physical_and_keeps_dtype() {
        let t =
            Tensor::from_bf16(vec![2, 3], &[1.0, 2., 3., 4., 5., 6.].map(bf16::from_f32)).unwrap();
        let flipped = transpose(&t).unwrap();
        assert_eq!(flipped.dtype(), DType::BF16);
        assert_eq!(flipped.shape().dims(), &[3, 2]);
        assert_eq!(flipped.to_f32_vec().unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn gemma_norm_offset_is_applied_once_and_folded_per_input_row() {
        let scale = add_one(Tensor::from_f32(vec![2], &[0.5, -0.25]).unwrap()).unwrap();
        assert_eq!(scale.to_f32_vec().unwrap(), vec![1.5, 0.75]);
        let mut linear = Linear {
            weight: Tensor::from_f32(vec![2, 2], &[1., 2., 3., 4.]).unwrap(),
            bias: None,
        };
        fold_input_scale(&mut linear, &scale).unwrap();
        assert_eq!(
            linear.weight.to_f32_vec().unwrap(),
            vec![1.5, 3.0, 2.25, 3.0]
        );
    }

    #[test]
    fn missing_tensor_names_the_key() {
        let err = Checkpoint::from_map(&Config::default(), HashMap::new())
            .err()
            .unwrap();
        assert!(err.to_string().contains("missing π0.5 weight"), "{err}");
    }

    #[test]
    fn synthetic_weights_are_valid_and_deterministic() {
        let config = Config {
            vision_depth: 1,
            language: GemmaConfig {
                depth: 1,
                ..GemmaConfig::GEMMA_2B
            },
            action_expert: GemmaConfig {
                depth: 1,
                ..GemmaConfig::GEMMA_300M
            },
            vocab_size: 64,
            ..Config::default()
        };
        let a = Checkpoint::synthetic(&config, 7).unwrap();
        let b = Checkpoint::synthetic(&config, 7).unwrap();
        assert_eq!(
            a.action_in.weight.to_f32_vec().unwrap(),
            b.action_in.weight.to_f32_vec().unwrap()
        );
    }
}
