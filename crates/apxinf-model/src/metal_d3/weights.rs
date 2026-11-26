//! d3-lite checkpoint loading: `language_model.*` BF16 backbone tensors, the
//! FP32 `readout.safetensors` matrix, and the `visual.*` vision tower, mapped
//! into the Metal port's own typed weight trees.
//!
//! Fixed transformations happen once at load, never per request:
//! - Qwen3.5 norms whose stored value is `w` are materialized as `1 + w`
//!   (input/post/final norms and the attention q/k norms); the GDN mixer's
//!   own `linear_attn.norm.weight` is already a direct RMSNorm weight and is
//!   left untouched, matching the reference loader.
//! - `linear_attn.conv1d.weight` is stored `[C, 1, S]` and transposed to the
//!   MLX conv1d `[C, S, 1]` layout.
//! - The vision patch embedding Conv3d weight `[C_out, C_in, T, P, P]` is
//!   reshaped to the `[C_out, C_in·T·P·P]` matrix its kernel = stride equals.
//!
//! The two trees are loaded by separate filtered reads so the text path's
//! loading contract (and its parity anchor) is unchanged.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};

use super::config::{Geometry, VisionContract, CODES};

#[derive(Clone)]
pub struct Linear {
    pub weight: Array,
}

/// A linear layer with the bias the vision tower stores for every projection.
#[derive(Clone)]
pub struct BiasedLinear {
    pub weight: Array,
    pub bias: Array,
}

pub struct VisionBlock {
    pub norm1_weight: Array,
    pub norm1_bias: Array,
    pub qkv: BiasedLinear,
    pub proj: BiasedLinear,
    pub norm2_weight: Array,
    pub norm2_bias: Array,
    pub fc1: BiasedLinear,
    pub fc2: BiasedLinear,
}

pub struct VisionMerger {
    pub norm_weight: Array,
    pub norm_bias: Array,
    pub fc1: BiasedLinear,
    pub fc2: BiasedLinear,
}

/// The 153 `visual.*` tensors, after the patch-embed reshape.
pub struct VisionWeights {
    /// `[embed_dim, in_channels·T·P·P]`, the Conv3d as its kernel=stride matrix.
    pub patch_weight: Array,
    pub patch_bias: Array,
    /// Learned `[num_position_embeddings, embed_dim]` table (48 x 48 grid).
    pub pos_embed: Array,
    pub blocks: Vec<VisionBlock>,
    pub merger: VisionMerger,
}

pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct GatedDelta {
    pub qkv: Linear,
    pub z: Linear,
    pub b: Linear,
    pub a: Linear,
    pub out: Linear,
    /// Depthwise causal kernel `[conv_dim, conv_kernel, 1]`, BF16.
    pub conv: Array,
    pub a_log: Array,
    pub dt_bias: Array,
    /// Gated RMSNorm weight `[v_dim]`, stored directly (not `1 + w`).
    pub norm: Array,
}

pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: Array,
    pub k_norm: Array,
}

pub enum Mixer {
    GatedDelta(GatedDelta),
    Attention(Attention),
}

pub struct Layer {
    pub input_norm: Array,
    pub post_norm: Array,
    pub mixer: Mixer,
    pub mlp: Mlp,
}

pub struct Weights {
    pub embed: Array,
    pub layers: Vec<Layer>,
    pub norm: Array,
    /// FP32 readout `[CODES, hidden]`, applied at the last position.
    pub readout: Array,
}

struct Take {
    map: HashMap<String, Array>,
    what: &'static str,
}

impl Take {
    fn array(&mut self, name: &str, dtype: MlxDType, shape: &[usize]) -> Result<Array> {
        let array = self
            .map
            .remove(name)
            .ok_or_else(|| Error::Other(format!("{} is missing `{name}`", self.what)))?;
        if array.dtype() != dtype || array.shape() != shape {
            return Err(Error::Other(format!(
                "{} `{name}`: expected {dtype:?} {shape:?}, got {:?} {:?}",
                self.what,
                array.dtype(),
                array.shape()
            )));
        }
        Ok(array)
    }

    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Array> {
        self.array(name, MlxDType::BF16, shape)
    }

    /// Qwen3.5 norms stored as `w`, applied by the model as `1 + w`.
    fn shifted(&mut self, name: &str, shape: &[usize]) -> Result<Array> {
        let weight = self.bf16(name, shape)?;
        let one = Array::scalar(weight.stream(), 1.0, MlxDType::BF16)?;
        weight.add(&one)
    }

    fn linear(&mut self, name: &str, output: usize, input: usize) -> Result<Linear> {
        Ok(Linear {
            weight: self.bf16(&format!("{name}.weight"), &[output, input])?,
        })
    }

    fn biased(&mut self, name: &str, output: usize, input: usize) -> Result<BiasedLinear> {
        Ok(BiasedLinear {
            weight: self.bf16(&format!("{name}.weight"), &[output, input])?,
            bias: self.bf16(&format!("{name}.bias"), &[output])?,
        })
    }

    fn finish(self) -> Result<()> {
        let mut extra: Vec<_> = self.map.into_keys().collect();
        if extra.is_empty() {
            return Ok(());
        }
        extra.sort();
        extra.truncate(8);
        Err(Error::Other(format!(
            "{} has unrecognized tensors: {}",
            self.what,
            extra.join(", ")
        )))
    }
}

impl Weights {
    /// Loads every text tensor of the d3-lite package. The caller supplies a
    /// validated [`Geometry`]; shapes below are derived from it so a geometry
    /// change cannot silently reuse the wrong layout.
    pub fn load(geometry: Geometry, stream: &Stream, checkpoint: &Path) -> Result<Self> {
        let mut files =
            apxinf_loader::safetensors::safetensors_files(checkpoint).map_err(Error::Other)?;
        let readout_file = checkpoint.join("readout.safetensors");
        if !readout_file.is_file() {
            return Err(Error::Other(format!(
                "d3-lite checkpoint is missing {}",
                readout_file.display()
            )));
        }
        files.push(readout_file);
        // This loader reads only `language_model.*`; `VisionWeights` reads `visual.*`.
        let map = apxinf_mlx::load_safetensors(stream, &files, |name| {
            name.starts_with("language_model.") || name == "weight"
        })?;
        let mut take = Take {
            map,
            what: "d3-lite Metal checkpoint",
        };
        let hidden = geometry.hidden;
        let inter = geometry.intermediate;
        let value_dim = geometry.value_dim();
        let conv_dim = geometry.conv_dim();
        let q_width = geometry.q_width();
        let kv_width = geometry.kv_width();
        let embed = take.bf16(
            "language_model.embed_tokens.weight",
            &[geometry.vocab, hidden],
        )?;
        let mut layers = Vec::with_capacity(geometry.layers);
        for index in 0..geometry.layers {
            let prefix = format!("language_model.layers.{index}");
            let mixer = if geometry.is_attention(index) {
                let attention = format!("{prefix}.self_attn");
                Mixer::Attention(Attention {
                    q: take.linear(&format!("{attention}.q_proj"), q_width, hidden)?,
                    k: take.linear(&format!("{attention}.k_proj"), kv_width, hidden)?,
                    v: take.linear(&format!("{attention}.v_proj"), kv_width, hidden)?,
                    o: take.linear(
                        &format!("{attention}.o_proj"),
                        hidden,
                        geometry.q_heads * geometry.head_dim,
                    )?,
                    q_norm: take
                        .shifted(&format!("{attention}.q_norm.weight"), &[geometry.head_dim])?,
                    k_norm: take
                        .shifted(&format!("{attention}.k_norm.weight"), &[geometry.head_dim])?,
                })
            } else {
                let linear = format!("{prefix}.linear_attn");
                let conv_stored = take.bf16(
                    &format!("{linear}.conv1d.weight"),
                    // `[C, 1, S]` in the checkpoint.
                    &[conv_dim, 1, geometry.conv_kernel],
                )?;
                Mixer::GatedDelta(GatedDelta {
                    qkv: take.linear(&format!("{linear}.in_proj_qkv"), conv_dim, hidden)?,
                    z: take.linear(&format!("{linear}.in_proj_z"), value_dim, hidden)?,
                    b: take.linear(&format!("{linear}.in_proj_b"), geometry.v_heads, hidden)?,
                    a: take.linear(&format!("{linear}.in_proj_a"), geometry.v_heads, hidden)?,
                    out: take.linear(&format!("{linear}.out_proj"), hidden, value_dim)?,
                    conv: conv_stored.transpose(&[0, 2, 1])?,
                    a_log: take.bf16(&format!("{linear}.A_log"), &[geometry.v_heads])?,
                    dt_bias: take.bf16(&format!("{linear}.dt_bias"), &[geometry.v_heads])?,
                    norm: take.bf16(&format!("{linear}.norm.weight"), &[geometry.v_dim])?,
                })
            };
            let mlp_prefix = format!("{prefix}.mlp");
            let mlp = Mlp {
                gate: take.linear(&format!("{mlp_prefix}.gate_proj"), inter, hidden)?,
                up: take.linear(&format!("{mlp_prefix}.up_proj"), inter, hidden)?,
                down: take.linear(&format!("{mlp_prefix}.down_proj"), hidden, inter)?,
            };
            layers.push(Layer {
                input_norm: take.shifted(&format!("{prefix}.input_layernorm.weight"), &[hidden])?,
                post_norm: take.shifted(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[hidden],
                )?,
                mixer,
                mlp,
            });
        }
        let norm = take.shifted("language_model.norm.weight", &[hidden])?;
        let readout = take.array("weight", MlxDType::F32, &[CODES, hidden])?;
        take.finish()?;
        Ok(Self {
            embed,
            layers,
            norm,
            readout,
        })
    }
}

impl VisionWeights {
    /// Loads and validates every `visual.*` tensor of the d3-lite package.
    /// The filter reads the vision tower only, so the text loader's file
    /// contract is untouched.
    pub fn load(vision: VisionContract, stream: &Stream, checkpoint: &Path) -> Result<Self> {
        let files =
            apxinf_loader::safetensors::safetensors_files(checkpoint).map_err(Error::Other)?;
        let map = apxinf_mlx::load_safetensors(stream, &files, |name| name.starts_with("visual."))?;
        let mut take = Take {
            map,
            what: "d3-lite Metal vision checkpoint",
        };
        let hidden = vision.hidden;
        let patch_dim = 3 * vision.temporal_patch * vision.patch * vision.patch;
        let patch_stored = take.bf16(
            "visual.patch_embed.proj.weight",
            &[hidden, 3, vision.temporal_patch, vision.patch, vision.patch],
        )?;
        let patch_weight = patch_stored.reshape(&[hidden, patch_dim])?;
        let patch_bias = take.bf16("visual.patch_embed.proj.bias", &[hidden])?;
        let pos_embed = take.bf16(
            "visual.pos_embed.weight",
            &[vision.num_position_embeddings, hidden],
        )?;
        debug_assert_eq!(vision.heads * vision.head_dim, hidden);
        let mut blocks = Vec::with_capacity(vision.depth);
        for index in 0..vision.depth {
            let prefix = format!("visual.blocks.{index}");
            blocks.push(VisionBlock {
                norm1_weight: take.bf16(&format!("{prefix}.norm1.weight"), &[hidden])?,
                norm1_bias: take.bf16(&format!("{prefix}.norm1.bias"), &[hidden])?,
                qkv: take.biased(&format!("{prefix}.attn.qkv"), 3 * hidden, hidden)?,
                proj: take.biased(&format!("{prefix}.attn.proj"), hidden, hidden)?,
                norm2_weight: take.bf16(&format!("{prefix}.norm2.weight"), &[hidden])?,
                norm2_bias: take.bf16(&format!("{prefix}.norm2.bias"), &[hidden])?,
                fc1: take.biased(
                    &format!("{prefix}.mlp.linear_fc1"),
                    vision.intermediate,
                    hidden,
                )?,
                fc2: take.biased(
                    &format!("{prefix}.mlp.linear_fc2"),
                    hidden,
                    vision.intermediate,
                )?,
            });
        }
        let merged = hidden * vision.spatial_merge * vision.spatial_merge;
        let merger = VisionMerger {
            norm_weight: take.bf16("visual.merger.norm.weight", &[hidden])?,
            norm_bias: take.bf16("visual.merger.norm.bias", &[hidden])?,
            fc1: take.biased("visual.merger.linear_fc1", merged, merged)?,
            fc2: take.biased("visual.merger.linear_fc2", vision.out_hidden, merged)?,
        };
        take.finish()?;
        Ok(Self {
            patch_weight,
            patch_bias,
            pos_embed,
            blocks,
            merger,
        })
    }
}
