//! Device-resident BF16 weights for the Qwen-Drive native executor.
//!
//! Built once at load from the checkpoint maps: most HF `[out, in]` projections
//! are transposed to `[in, out]` for the row-major GEMM path (the scaffold's
//! transpose), per-layer Q/K/V and gate/up projections are concatenated into
//! fused weights. GDN input projections retain their original [out,in] layout
//! and separate GEMMs: their BF16 reductions depend on physical geometry.
//! Every checkpoint transformation runs
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

// FIX (implement_r3 / synthesis_r3): the bf16-loaded reference holds A_log bf16-rounded
// and upcasts in-formula (modeling_qwen3_5.py:522); native kept the F32 checkpoint value
// -- a <=2^-9 decay-rate divergence and a guaranteed exact-match blocker. Permanent: do
// NOT revert.
fn bf16_grid_round_f32(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let values: Vec<f32> = tensor
        .to_f32_vec()?
        .iter()
        .map(|&value| half::bf16::from_f32(value).to_f32())
        .collect();
    Tensor::from_f32(dims, &values)
}

/// f32 -> bf16 narrowing on the host (round-to-nearest-even, the reference's
/// `.to(bfloat16)` cast). The checkpoint stores the GDN gated-RMSNorm weight
/// (`linear_attn.norm.weight`) in f32; the gated RMS kernel expects bf16.
fn narrow_to_bf16(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let values = tensor.to_f32_vec()?;
    let narrowed: Vec<half::bf16> = values
        .iter()
        .map(|value| half::bf16::from_f32(*value))
        .collect();
    Tensor::from_bf16(dims, &narrowed)
}

fn up(backend: &dyn Backend, tensor: &Tensor) -> Result<Tensor> {
    backend.to_device(tensor)
}

fn up_mlp(
    backend: &dyn Backend,
    mlp: &super::weights::ExpertMlp,
    checkpoint_layout: bool,
) -> Result<DeviceMlp> {
    let fc1 = if checkpoint_layout {
        transpose_2d(&mlp.fc1_w)?
    } else {
        mlp.fc1_w.clone()
    };
    let fc2 = if checkpoint_layout {
        transpose_2d(&mlp.fc2_w)?
    } else {
        mlp.fc2_w.clone()
    };
    Ok(DeviceMlp {
        fc1_w: up(backend, &fc1)?,
        fc1_b: up(backend, &mlp.fc1_b)?,
        fc2_w: up(backend, &fc2)?,
        fc2_b: up(backend, &mlp.fc2_b)?,
        checkpoint_layout,
    })
}

/// One Linear -> SiLU -> Linear block on device.
pub struct DeviceMlp {
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
    /// Both weights retain checkpoint `[out,in]` layout for single-row MLPs.
    pub checkpoint_layout: bool,
}

pub struct FullAttentionLayerWeights {
    pub input_norm: Tensor,
    /// Separate checkpoint [out,hidden] projections; Q contains q|gate per head.
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
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
    /// Input projections preserve reference `[out, hidden]` storage and GEMM shape.
    pub qkv_w: Tensor,
    pub z_w: Tensor,
    pub b_w: Tensor,
    pub a_w: Tensor,
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
    /// Checkpoint [6*hidden,hidden] layout for the biased matrix-vector product.
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
    if num_features == 0 || !max_frequency.is_finite() || max_frequency <= 0.0 {
        return Err(Error::Other(
            "qwen_drive: invalid Fourier table geometry".into(),
        ));
    }
    // Torch CUDA BF16 logspace rounds the endpoint, step and each arithmetic
    // operation to BF16, and constructs the second half backwards from the end.
    // Computing a float logspace and casting only its output changes frequencies.
    let round = |value: f32| half::bf16::from_f32(value).to_f32();
    let log_max = round((max_frequency as f64).log10() as f32);
    let step = if num_features > 1 {
        round(log_max / round((num_features - 1) as f32))
    } else {
        0.0
    };
    let values: Vec<half::bf16> = (0..num_features)
        .map(|i| {
            let exponent = if num_features == 1 {
                0.0
            } else if i < num_features / 2 {
                round(step * round(i as f32))
            } else {
                round(log_max - round(step * round((num_features - i - 1) as f32)))
            };
            half::bf16::from_f32(10f32.powf(exponent))
        })
        .collect();
    Tensor::from_bf16(vec![num_features], &values)
}

#[cfg(test)]
mod frequency_tests {
    use super::fourier_freq_table;

    #[test]
    fn bf16_logspace_matches_frozen_cuda_reference() {
        let expected = [
            1.0, 1.203125, 1.4453125, 1.7421875, 2.09375, 2.515625, 3.015625, 3.65625, 4.375,
            5.28125, 6.375, 7.625, 9.125, 11.125, 13.3125, 15.9375,
        ];
        assert_eq!(
            fourier_freq_table(16, 16.0).unwrap().to_f32_vec().unwrap(),
            expected
        );
        assert_eq!(
            fourier_freq_table(1, 16.0).unwrap().to_f32_vec().unwrap(),
            [1.0]
        );
        assert!(fourier_freq_table(0, 16.0).is_err());
        assert!(fourier_freq_table(16, 0.0).is_err());
    }
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
        // FIX (implement_r3 / synthesis_r3): confirmation-line accumulators for the A_log
        // bf16-grid rounding below; max_delta over the PRE-round values proves the fix
        // load-bearing (>0 expected). Permanent.
        let mut a_log_max_delta = 0.0f32;
        let mut a_log_layer0_first4: Vec<f32> = Vec::new();
        for index in 0..text.n_layers {
            let p = format!("model.language_model.layers.{index}");
            let input_norm = up(
                backend,
                &take(&mut language, &format!("{p}.input_layernorm.weight"))?,
            )?;
            let post_norm = up(
                backend,
                &take(
                    &mut language,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
            )?;
            let gate = take_transposed(&mut language, &format!("{p}.mlp.gate_proj.weight"))?;
            let up_w = take_transposed(&mut language, &format!("{p}.mlp.up_proj.weight"))?;
            let gate_up_w = up(backend, &concat_columns(&[&gate, &up_w])?)?;
            let down_w = up(
                backend,
                &take(&mut language, &format!("{p}.mlp.down_proj.weight"))?,
            )?;
            if text.is_full_attention(index) {
                let q = take(&mut language, &format!("{p}.self_attn.q_proj.weight"))?;
                let k = take(&mut language, &format!("{p}.self_attn.k_proj.weight"))?;
                let v = take(&mut language, &format!("{p}.self_attn.v_proj.weight"))?;
                if q.shape().dims() != [q_width, hidden]
                    || k.shape().dims() != [kv_width, hidden]
                    || v.shape().dims() != [kv_width, hidden]
                {
                    return Err(Error::Other(format!(
                        "qwen_drive: q/k/v projection shape mismatch at layer {index}"
                    )));
                }
                layers.push(MixerWeights::FullAttention(FullAttentionLayerWeights {
                    input_norm,
                    q_w: up(backend, &q)?,
                    k_w: up(backend, &k)?,
                    v_w: up(backend, &v)?,
                    q_norm: up(
                        backend,
                        &take(&mut language, &format!("{p}.self_attn.q_norm.weight"))?,
                    )?,
                    k_norm: up(
                        backend,
                        &take(&mut language, &format!("{p}.self_attn.k_norm.weight"))?,
                    )?,
                    o_w: up(
                        backend,
                        &take(&mut language, &format!("{p}.self_attn.o_proj.weight"))?,
                    )?,
                    post_norm,
                    gate_up_w,
                    down_w,
                }));
            } else {
                let in_qkv = take(
                    &mut language,
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                )?;
                let in_z = take(&mut language, &format!("{p}.linear_attn.in_proj_z.weight"))?;
                let in_b = take(&mut language, &format!("{p}.linear_attn.in_proj_b.weight"))?;
                let in_a = take(&mut language, &format!("{p}.linear_attn.in_proj_a.weight"))?;
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
                // FIX (implement_r3 / synthesis_r3): bf16-grid-round A_log at load (see
                // bf16_grid_round_f32); the pre-round max-delta feeds the confirmation line.
                let a_log_raw =
                    widen_to_f32(&take(&mut language, &format!("{p}.linear_attn.A_log"))?)?;
                let a_log_pre = a_log_raw.to_f32_vec()?;
                a_log_max_delta = a_log_pre.iter().fold(a_log_max_delta, |m, &v| {
                    m.max((v - half::bf16::from_f32(v).to_f32()).abs())
                });
                if index == 0 {
                    a_log_layer0_first4 = a_log_pre[..a_log_pre.len().min(4)].to_vec();
                }
                layers.push(MixerWeights::Gdn(GdnLayerWeights {
                    input_norm,
                    qkv_w: up(backend, &in_qkv)?,
                    z_w: up(backend, &in_z)?,
                    b_w: up(backend, &in_b)?,
                    a_w: up(backend, &in_a)?,
                    conv_w: up(backend, &conv.reshape(vec![conv_dim, kernel])?)?,
                    dt_bias: up(
                        backend,
                        &widen_to_f32(&take(&mut language, &format!("{p}.linear_attn.dt_bias"))?)?,
                    )?,
                    a_log: up(backend, &bf16_grid_round_f32(&a_log_raw)?)?,
                    gated_norm: up(
                        backend,
                        &narrow_to_bf16(&take(
                            &mut language,
                            &format!("{p}.linear_attn.norm.weight"),
                        )?)?,
                    )?,
                    out_w: up(
                        backend,
                        &take(&mut language, &format!("{p}.linear_attn.out_proj.weight"))?,
                    )?,
                    post_norm,
                    gate_up_w,
                    down_w,
                }));
            }
        }
        // FIX (implement_r3 / synthesis_r3): load-time confirmation line for the A_log
        // bf16-grid rounding (lands in the captured load section; expected max_delta>0).
        eprintln!(
            "[qwen_drive] a_log_bf16_round max_delta={:.6} layer0_first4={:?}",
            a_log_max_delta, a_log_layer0_first4
        );
        let embed_tokens = up(
            backend,
            &take(&mut language, "model.language_model.embed_tokens.weight")?,
        )?;
        let final_norm = up(
            backend,
            &take(&mut language, "model.language_model.norm.weight")?,
        )?;
        if !language.is_empty() {
            let mut names: Vec<String> = language.keys().cloned().collect();
            names.sort_unstable();
            // MTP/auxiliary tensors are allowed but must be named, not silently
            // dropped: keep the check informational-strict like the scaffold.
            let mtp_only = names
                .iter()
                .all(|name| name.starts_with("model.language_model.mtp"));
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
        let patch_vec = vision_cfg.in_channels
            * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size
            * vision_cfg.patch_size;
        if patch_dims.len() != 5
            || patch_dims[0] != vision_cfg.hidden_size
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
                qkv_w: up(
                    backend,
                    &take_transposed(&mut visual, &format!("{p}.attn.qkv.weight"))?,
                )?,
                qkv_b: up(backend, &take(&mut visual, &format!("{p}.attn.qkv.bias"))?)?,
                proj_w: up(
                    backend,
                    &take_transposed(&mut visual, &format!("{p}.attn.proj.weight"))?,
                )?,
                proj_b: up(backend, &take(&mut visual, &format!("{p}.attn.proj.bias"))?)?,
                norm2_w: up(backend, &take(&mut visual, &format!("{p}.norm2.weight"))?)?,
                norm2_b: up(backend, &take(&mut visual, &format!("{p}.norm2.bias"))?)?,
                fc1_w: up(
                    backend,
                    &take_transposed(&mut visual, &format!("{p}.mlp.linear_fc1.weight"))?,
                )?,
                fc1_b: up(
                    backend,
                    &take(&mut visual, &format!("{p}.mlp.linear_fc1.bias"))?,
                )?,
                fc2_w: up(
                    backend,
                    &take_transposed(&mut visual, &format!("{p}.mlp.linear_fc2.weight"))?,
                )?,
                fc2_b: up(
                    backend,
                    &take(&mut visual, &format!("{p}.mlp.linear_fc2.bias"))?,
                )?,
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
            patch_b: up(
                backend,
                &take(&mut visual, "model.visual.patch_embed.proj.bias")?,
            )?,
            pos_embed: up(
                backend,
                &take(&mut visual, "model.visual.pos_embed.weight")?,
            )?,
            blocks,
            merger_norm_w: up(
                backend,
                &take(&mut visual, "model.visual.merger.norm.weight")?,
            )?,
            merger_norm_b: up(
                backend,
                &take(&mut visual, "model.visual.merger.norm.bias")?,
            )?,
            merger_fc1_w: up(backend, &merger_fc1)?,
            merger_fc1_b: up(
                backend,
                &take(&mut visual, "model.visual.merger.linear_fc1.bias")?,
            )?,
            merger_fc2_w: up(backend, &merger_fc2)?,
            merger_fc2_b: up(
                backend,
                &take(&mut visual, "model.visual.merger.linear_fc2.bias")?,
            )?,
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
        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            vision,
            expert,
        })
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
                modulation_w: up(backend, &transpose_2d(&layer.modulation_w)?)?,
                modulation_b: up(backend, &layer.modulation_b)?,
            });
        }
        Ok(ExpertDeviceWeights {
            trajectory_proj_w: up(backend, &weights.trajectory_proj_w)?,
            trajectory_proj_b: up(backend, &weights.trajectory_proj_b)?,
            fourier: up_mlp(backend, &weights.fourier_encoder, false)?,
            waypoint_embed: up(backend, &weights.waypoint_embed)?,
            time_mlp: up_mlp(backend, &weights.time_mlp, true)?,
            nav_mlp: up_mlp(backend, &weights.nav_mlp, true)?,
            ego_mlp: up_mlp(backend, &weights.ego_mlp, true)?,
            history_encoder: up_mlp(backend, &weights.history_encoder, true)?,
            history_velocity_encoder: up_mlp(backend, &weights.history_velocity_encoder, true)?,
            history_acceleration_encoder: up_mlp(
                backend,
                &weights.history_acceleration_encoder,
                true,
            )?,
            query_fusion: up_mlp(backend, &weights.query_fusion, false)?,
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
