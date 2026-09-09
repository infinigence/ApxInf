//! Device-resident BF16 weights for the Qwen-Drive native executor.
//!
//! Built once at load from the checkpoint maps: HF `[out, in]` projections
//! are transposed to `[in, out]` for the row-major GEMM path (the scaffold's
//! transpose), per-layer Q/K/V and gate/up projections are concatenated into
//! fused weights (a canonical rewrite: identical dot products, one GEMM), and
//! the GDN in_proj_qkv/z/b/a projections are fused into one GEMM whose column
//! slices the preparation kernels read strided. Every transformation runs
//! once at load, never on the hot path.

use std::collections::HashMap;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::config::QwenDriveConfig;
use super::weights::{transpose_2d, QwenDriveExpertWeights, QwenDriveVlmWeights};

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| Error::Other(format!("qwen_drive device weights: missing {name}")))
}

fn take_transposed(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    transpose_2d(&take(map, name)?)
}

/// Host-side column concat of equally tall `[in, out_i]` matrices into
/// `[in, sum(out_i)]`. BF16 only (all checkpoint projections are bf16).
fn concat_columns(tensors: &[&Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(Error::Other("qwen_drive concat: no tensors".into()));
    }
    let rows = tensors[0].shape().dims()[0];
    let mut cols = 0usize;
    for tensor in tensors {
        let dims = tensor.shape().dims();
        if dims.len() != 2 || dims[0] != rows || tensor.dtype() != DType::BF16 {
            return Err(Error::Other(
                "qwen_drive concat: expected equally tall BF16 matrices".into(),
            ));
        }
        cols += dims[1];
    }
    let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        let mut offset = 0usize;
        for tensor in tensors {
            let width = tensor.shape().dims()[1];
            let data = tensor.as_bf16()?;
            out[r * cols + offset..r * cols + offset + width]
                .copy_from_slice(&data[r * width..(r + 1) * width]);
            offset += width;
        }
    }
    Tensor::from_bf16(vec![rows, cols], &out)
}

/// Exact bf16 -> f32 widening on the host (for fp32-consumed constants).
fn widen_to_f32(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let values = tensor.to_f32_vec()?;
    Tensor::from_f32(dims, &values)
}

/// f32 -> bf16 narrowing on the host (round-to-nearest-even, the reference's
/// `.to(bfloat16)` cast). The checkpoint stores the GDN gated-RMSNorm weight
/// (`linear_attn.norm.weight`) in f32; the gated RMS kernel expects bf16.
fn narrow_to_bf16(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let values = tensor.to_f32_vec()?;
    let narrowed: Vec<half::bf16> = values.iter().map(|value| half::bf16::from_f32(*value)).collect();
    Tensor::from_bf16(dims, &narrowed)
}

fn up(backend: &dyn Backend, tensor: &Tensor) -> Result<Tensor> {
    backend.to_device(tensor)
}

fn up_mlp(backend: &dyn Backend, mlp: &super::weights::ExpertMlp) -> Result<DeviceMlp> {
    Ok(DeviceMlp {
        fc1_w: up(backend, &mlp.fc1_w)?,
        fc1_b: up(backend, &mlp.fc1_b)?,
        fc2_w: up(backend, &mlp.fc2_w)?,
        fc2_b: up(backend, &mlp.fc2_b)?,
    })
}

/// One Linear -> SiLU -> Linear block on device.
pub struct DeviceMlp {
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

pub struct FullAttentionLayerWeights {
    pub input_norm: Tensor,
    /// Fused `[hidden, q_heads*2*head_dim + 2*kv_heads*head_dim]` (q|gate per
    /// head first, then k heads, then v heads), transposed at load.
    pub qkv_w: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub o_w: Tensor,
    pub post_norm: Tensor,
    /// Fused `[hidden, 2*intermediate]` (gate rows first).
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
}

pub struct GdnLayerWeights {
    pub input_norm: Tensor,
    /// Fused `[hidden, conv_dim + value_dim + 2*num_v_heads]`
    /// (in_proj_qkv | in_proj_z | in_proj_b | in_proj_a).
    pub qkvzba_w: Tensor,
    /// `[conv_dim, kernel_size]` (squeezed depthwise conv weight).
    pub conv_w: Tensor,
    /// `[num_v_heads]` fp32.
    pub dt_bias: Tensor,
    /// `[num_v_heads]` fp32.
    pub a_log: Tensor,
    /// `[head_v_dim]` gated RMSNorm weight (plain semantics).
    pub gated_norm: Tensor,
    pub out_w: Tensor,
    pub post_norm: Tensor,
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
}

pub enum MixerWeights {
    FullAttention(FullAttentionLayerWeights),
    Gdn(GdnLayerWeights),
}

pub struct VisionBlockWeights {
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    pub qkv_w: Tensor,
    pub qkv_b: Tensor,
    pub proj_w: Tensor,
    pub proj_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

pub struct VisionDeviceWeights {
    /// `[3 * temporal * patch * patch, hidden]` flattened Conv3d, transposed.
    pub patch_w: Tensor,
    pub patch_b: Tensor,
    /// `[num_position_embeddings, hidden]` learned table (bilinear source).
    pub pos_embed: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    pub merger_norm_w: Tensor,
    pub merger_norm_b: Tensor,
    pub merger_fc1_w: Tensor,
    pub merger_fc1_b: Tensor,
    pub merger_fc2_w: Tensor,
    pub merger_fc2_b: Tensor,
}

pub struct ExpertLayerDeviceWeights {
    pub input_norm: Tensor,
    pub qkv_w: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub o_w: Tensor,
    pub post_norm: Tensor,
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
    pub modulation_w: Tensor,
    pub modulation_b: Tensor,
}

pub struct ExpertDeviceWeights {
    pub trajectory_proj_w: Tensor,
    pub trajectory_proj_b: Tensor,
    pub fourier: DeviceMlp,
    pub waypoint_embed: Tensor,
    pub time_mlp: DeviceMlp,
    pub nav_mlp: DeviceMlp,
    pub ego_mlp: DeviceMlp,
    pub history_encoder: DeviceMlp,
    pub history_velocity_encoder: DeviceMlp,
    pub history_acceleration_encoder: DeviceMlp,
    pub query_fusion: DeviceMlp,
    pub layers: Vec<ExpertLayerDeviceWeights>,
    pub final_norm: Tensor,
    pub out_proj_w: Tensor,
    pub out_proj_b: Tensor,
    /// bf16 logspace frequency table (rounding is semantic), built at load.
    pub fourier_freqs: Tensor,
}

/// torch.logspace(0, log10(max_frequency), steps, dtype=bf16) rebuilt in the
/// compute dtype exactly like the reference encoder.
fn fourier_freq_table(num_features: usize, max_frequency: f32) -> Result<Tensor> {
    if num_features == 0 {
        return Err(Error::Other("qwen_drive: empty Fourier table".into()));
    }
    let log_max = (max_frequency as f64).log10() as f32;
    let step = if num_features > 1 {
        log_max / (num_features - 1) as f32
    } else {
        0.0
    };
    let values: Vec<half::bf16> = (0..num_features)
        .map(|i| half::bf16::from_f32(10f32.powf(step * i as f32)))
        .collect();
    Tensor::from_bf16(vec![num_features], &values)
}

pub struct QwenDriveDeviceWeights {
    /// `[vocab, hidden]`; doubles as the tied lm_head via a transposed GEMM.
    pub embed_tokens: Tensor,
    pub layers: Vec<MixerWeights>,
    pub final_norm: Tensor,
    pub vision: VisionDeviceWeights,
    pub expert: Option<ExpertDeviceWeights>,
}

impl QwenDriveDeviceWeights {
    pub fn from_maps(
        config: &QwenDriveConfig,
        vlm: QwenDriveVlmWeights,
        expert: Option<QwenDriveExpertWeights>,
        backend: &dyn Backend,
    ) -> Result<Self> {
        let mut language = vlm.language;
        let mut visual = vlm.visual;
        let text = &config.text;
        let hidden = text.hidden_size;
        let kv_width = text.n_kv_heads * text.head_dim;
        let q_width = text.n_heads * 2 * text.head_dim;

        let mut layers = Vec::with_capacity(text.n_layers);
        for index in 0..text.n_layers {
            let p = format!("model.language_model.layers.{index}");
            let input_norm = up(backend, &take(&mut language, &format!("{p}.input_layernorm.weight"))?)?;
            let post_norm = up(backend, &take(&mut language, &format!("{p}.post_attention_layernorm.weight"))?)?;
            let gate = take_transposed(&mut language, &format!("{p}.mlp.gate_proj.weight"))?;
            let up_w = take_transposed(&mut language, &format!("{p}.mlp.up_proj.weight"))?;
            let gate_up_w = up(backend, &concat_columns(&[&gate, &up_w])?)?;
            let down_w = up(backend, &take_transposed(&mut language, &format!("{p}.mlp.down_proj.weight"))?)?;
            if text.is_full_attention(index) {
                let q = take_transposed(&mut language, &format!("{p}.self_attn.q_proj.weight"))?;
                let k = take_transposed(&mut language, &format!("{p}.self_attn.k_proj.weight"))?;
                let v = take_transposed(&mut language, &format!("{p}.self_attn.v_proj.weight"))?;
                let qkv_w = concat_columns(&[&q, &k, &v])?;
                let dims = qkv_w.shape().dims().to_vec();
                if dims != [hidden, q_width + 2 * kv_width] {
                    return Err(Error::Other(format!(
                        "qwen_drive: fused qkv width mismatch at layer {index}: {dims:?}"
                    )));
                }
                layers.push(MixerWeights::FullAttention(FullAttentionLayerWeights {
                    input_norm,
                    qkv_w: up(backend, &qkv_w)?,
                    q_norm: up(backend, &take(&mut language, &format!("{p}.self_attn.q_norm.weight"))?)?,
                    k_norm: up(backend, &take(&mut language, &format!("{p}.self_attn.k_norm.weight"))?)?,
                    o_w: up(backend, &take_transposed(&mut language, &format!("{p}.self_attn.o_proj.weight"))?)?,
                    post_norm,
                    gate_up_w,
                    down_w,
                }));
            } else {
                let in_qkv = take_transposed(&mut language, &format!("{p}.linear_attn.in_proj_qkv.weight"))?;
                let in_z = take_transposed(&mut language, &format!("{p}.linear_attn.in_proj_z.weight"))?;
                let in_b = take_transposed(&mut language, &format!("{p}.linear_attn.in_proj_b.weight"))?;
                let in_a = take_transposed(&mut language, &format!("{p}.linear_attn.in_proj_a.weight"))?;
                let conv = take(&mut language, &format!("{p}.linear_attn.conv1d.weight"))?;
                let conv_dims = conv.shape().dims().to_vec();
                let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
                    + text.linear_num_value_heads * text.linear_value_head_dim;
                let kernel = text.linear_conv_kernel_dim;
                if conv_dims != [conv_dim, 1, kernel] {
                    return Err(Error::Other(format!(
                        "qwen_drive: conv1d weight at layer {index} has shape {conv_dims:?}, expected [{conv_dim}, 1, {kernel}]"
                    )));
                }
                layers.push(MixerWeights::Gdn(GdnLayerWeights {
                    input_norm,
                    qkvzba_w: up(backend, &concat_columns(&[&in_qkv, &in_z, &in_b, &in_a])?)?,
                    conv_w: up(backend, &conv.reshape(vec![conv_dim, kernel])?)?,
                    dt_bias: up(backend, &widen_to_f32(&take(&mut language, &format!("{p}.linear_attn.dt_bias"))?)?)?,
                    a_log: up(backend, &widen_to_f32(&take(&mut language, &format!("{p}.linear_attn.A_log"))?)?)?,
                    gated_norm: up(backend, &narrow_to_bf16(&take(&mut language, &format!("{p}.linear_attn.norm.weight"))?)?)?,
                    out_w: up(backend, &take_transposed(&mut language, &format!("{p}.linear_attn.out_proj.weight"))?)?,
                    post_norm,
                    gate_up_w,
                    down_w,
                }));
            }
        }
        let embed_tokens = up(backend, &take(&mut language, "model.language_model.embed_tokens.weight")?)?;
        let final_norm = up(backend, &take(&mut language, "model.language_model.norm.weight")?)?;
        if !language.is_empty() {
            let mut names: Vec<String> = language.keys().cloned().collect();
            names.sort_unstable();
            // MTP/auxiliary tensors are allowed but must be named, not silently
            // dropped: keep the check informational-strict like the scaffold.
            let mtp_only = names.iter().all(|name| name.starts_with("model.language_model.mtp"));
            if !mtp_only {
                return Err(Error::Other(format!(
                    "qwen_drive device weights: {} unconsumed language tensors, first: {}",
                    names.len(),
                    names[0]
                )));
            }
        }

        // Vision tower.
        let vision_cfg = &config.vision;
        let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
        let patch = take(&mut visual, "model.visual.patch_embed.proj.weight")?;
        let patch_dims = patch.shape().dims().to_vec();
        let patch_vec = vision_cfg.in_channels * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size * vision_cfg.patch_size;
        if patch_dims.len() != 5 || patch_dims[0] != vision_cfg.hidden_size
            || patch_dims.iter().product::<usize>() != vision_cfg.hidden_size * patch_vec
        {
            return Err(Error::Other(format!(
                "qwen_drive: patch_embed weight has shape {patch_dims:?}"
            )));
        }
        let patch_w = transpose_2d(&patch.reshape(vec![vision_cfg.hidden_size, patch_vec])?)?;
        let mut blocks = Vec::with_capacity(vision_cfg.depth);
        for index in 0..vision_cfg.depth {
            let p = format!("model.visual.blocks.{index}");
            blocks.push(VisionBlockWeights {
                norm1_w: up(backend, &take(&mut visual, &format!("{p}.norm1.weight"))?)?,
                norm1_b: up(backend, &take(&mut visual, &format!("{p}.norm1.bias"))?)?,
                qkv_w: up(backend, &take_transposed(&mut visual, &format!("{p}.attn.qkv.weight"))?)?,
                qkv_b: up(backend, &take(&mut visual, &format!("{p}.attn.qkv.bias"))?)?,
                proj_w: up(backend, &take_transposed(&mut visual, &format!("{p}.attn.proj.weight"))?)?,
                proj_b: up(backend, &take(&mut visual, &format!("{p}.attn.proj.bias"))?)?,
                norm2_w: up(backend, &take(&mut visual, &format!("{p}.norm2.weight"))?)?,
                norm2_b: up(backend, &take(&mut visual, &format!("{p}.norm2.bias"))?)?,
                fc1_w: up(backend, &take_transposed(&mut visual, &format!("{p}.mlp.linear_fc1.weight"))?)?,
                fc1_b: up(backend, &take(&mut visual, &format!("{p}.mlp.linear_fc1.bias"))?)?,
                fc2_w: up(backend, &take_transposed(&mut visual, &format!("{p}.mlp.linear_fc2.weight"))?)?,
                fc2_b: up(backend, &take(&mut visual, &format!("{p}.mlp.linear_fc2.bias"))?)?,
            });
        }
        let merger_fc1 = take_transposed(&mut visual, "model.visual.merger.linear_fc1.weight")?;
        let merger_fc2 = take_transposed(&mut visual, "model.visual.merger.linear_fc2.weight")?;
        let expected_fc1 = vision_cfg.hidden_size * merge_sq;
        if merger_fc1.shape().dims() != [expected_fc1, expected_fc1] {
            return Err(Error::Other("qwen_drive: merger fc1 shape mismatch".into()));
        }
        let vision = VisionDeviceWeights {
            patch_w: up(backend, &patch_w)?,
            patch_b: up(backend, &take(&mut visual, "model.visual.patch_embed.proj.bias")?)?,
            pos_embed: up(backend, &take(&mut visual, "model.visual.pos_embed.weight")?)?,
            blocks,
            merger_norm_w: up(backend, &take(&mut visual, "model.visual.merger.norm.weight")?)?,
            merger_norm_b: up(backend, &take(&mut visual, "model.visual.merger.norm.bias")?)?,
            merger_fc1_w: up(backend, &merger_fc1)?,
            merger_fc1_b: up(backend, &take(&mut visual, "model.visual.merger.linear_fc1.bias")?)?,
            merger_fc2_w: up(backend, &merger_fc2)?,
            merger_fc2_b: up(backend, &take(&mut visual, "model.visual.merger.linear_fc2.bias")?)?,
        };
        if !visual.is_empty() {
            let mut names: Vec<String> = visual.keys().cloned().collect();
            names.sort_unstable();
            return Err(Error::Other(format!(
                "qwen_drive device weights: {} unconsumed visual tensors, first: {}",
                names.len(),
                names[0]
            )));
        }

        let expert = expert
            .map(|expert| Self::upload_expert(config, expert, backend))
            .transpose()?;
        Ok(Self { embed_tokens, layers, final_norm, vision, expert })
    }

    fn upload_expert(
        config: &QwenDriveConfig,
        weights: QwenDriveExpertWeights,
        backend: &dyn Backend,
    ) -> Result<ExpertDeviceWeights> {
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(ExpertLayerDeviceWeights {
                input_norm: up(backend, &layer.input_layernorm)?,
                qkv_w: up(backend, &layer.qkv_w)?,
                q_norm: up(backend, &layer.q_norm)?,
                k_norm: up(backend, &layer.k_norm)?,
                o_w: up(backend, &layer.o_w)?,
                post_norm: up(backend, &layer.post_attention_layernorm)?,
                gate_up_w: up(backend, &layer.gate_up_w)?,
                down_w: up(backend, &layer.down_w)?,
                modulation_w: up(backend, &layer.modulation_w)?,
                modulation_b: up(backend, &layer.modulation_b)?,
            });
        }
        Ok(ExpertDeviceWeights {
            trajectory_proj_w: up(backend, &weights.trajectory_proj_w)?,
            trajectory_proj_b: up(backend, &weights.trajectory_proj_b)?,
            fourier: up_mlp(backend, &weights.fourier_encoder)?,
            waypoint_embed: up(backend, &weights.waypoint_embed)?,
            time_mlp: up_mlp(backend, &weights.time_mlp)?,
            nav_mlp: up_mlp(backend, &weights.nav_mlp)?,
            ego_mlp: up_mlp(backend, &weights.ego_mlp)?,
            history_encoder: up_mlp(backend, &weights.history_encoder)?,
            history_velocity_encoder: up_mlp(backend, &weights.history_velocity_encoder)?,
            history_acceleration_encoder: up_mlp(backend, &weights.history_acceleration_encoder)?,
            query_fusion: up_mlp(backend, &weights.query_fusion)?,
            layers,
            final_norm: up(backend, &weights.final_layernorm)?,
            out_proj_w: up(backend, &weights.out_proj_w)?,
            out_proj_b: up(backend, &weights.out_proj_b)?,
            fourier_freqs: up(
                backend,
                &fourier_freq_table(
                    config.expert.fourier_num_features,
                    config.expert.fourier_max_frequency,
                )?,
            )?,
        })
    }
}
