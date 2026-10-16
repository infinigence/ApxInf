//! BF16 device layout for the official d3-mini checkpoint.

use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::CudaContext;

use crate::d3::backend::upload;
use crate::d3::{D3CheckpointWeights, D3PackageConfig, D3ReadoutWeights};

use super::host::transpose_2d;

pub(crate) struct FullAttentionWeights {
    pub input_norm: Tensor,
    pub q_w: Tensor,
    pub k_w: Tensor,
    pub v_w: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub o_w: Tensor,
    pub post_norm: Tensor,
    /// `[hidden, 2 * intermediate]` for fused SwiGLU.
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
}

pub(crate) struct GdnWeights {
    pub input_norm: Tensor,
    pub qkv_w: Tensor,
    pub z_w: Tensor,
    pub b_w: Tensor,
    pub a_w: Tensor,
    pub conv_w: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub gated_norm: Tensor,
    pub out_w: Tensor,
    pub post_norm: Tensor,
    pub gate_up_w: Tensor,
    pub down_w: Tensor,
}

pub(crate) enum MixerWeights {
    FullAttention(FullAttentionWeights),
    Gdn(GdnWeights),
}

pub(crate) struct VisionBlockWeights {
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

pub(crate) struct VisionWeights {
    pub patch_w: Tensor,
    pub patch_b: Tensor,
    pub pos_embed: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    pub merger_norm_w: Tensor,
    pub merger_norm_b: Tensor,
    pub merger_fc1_w: Tensor,
    pub merger_fc1_b: Tensor,
    pub merger_fc2_w: Tensor,
    pub merger_fc2_b: Tensor,
}

pub(crate) struct D3DeviceWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<MixerWeights>,
    pub final_norm: Tensor,
    pub vision: VisionWeights,
    /// FP32 `[hidden, number_of_codes]`.
    pub readout: Tensor,
}

impl D3DeviceWeights {
    pub(crate) fn from_host(
        config: &D3PackageConfig,
        checkpoint: D3CheckpointWeights,
        readout: D3ReadoutWeights,
        ctx: &CudaContext,
    ) -> Result<Self> {
        let mut language = checkpoint.language;
        let mut visual = checkpoint.visual;
        let text = &config.model.text_config;
        let hidden = text.hidden_size;
        let inter = text.intermediate_size;
        let kv_width = text.num_key_value_heads * text.head_dim;
        let q_width = text.num_attention_heads * 2 * text.head_dim;
        let gdn_qkv = 2 * text.linear_num_key_heads * text.linear_key_head_dim
            + text.linear_num_value_heads * text.linear_value_head_dim;
        let gdn_z = text.linear_num_value_heads * text.linear_value_head_dim;
        let gdn_ab = text.linear_num_value_heads;

        expect_shape(
            language.get("language_model.embed_tokens.weight"),
            "language_model.embed_tokens.weight",
            &[text.vocab_size, hidden],
            DType::BF16,
        )?;
        expect_shape(
            language.get("language_model.norm.weight"),
            "language_model.norm.weight",
            &[hidden],
            DType::BF16,
        )?;

        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for index in 0..text.num_hidden_layers {
            let p = format!("language_model.layers.{index}");
            let input_norm = take_shape(
                &mut language,
                &format!("{p}.input_layernorm.weight"),
                &[hidden],
                DType::BF16,
            )?;
            let post_norm = take_shape(
                &mut language,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
                DType::BF16,
            )?;
            let gate = take_shape(
                &mut language,
                &format!("{p}.mlp.gate_proj.weight"),
                &[inter, hidden],
                DType::BF16,
            )?;
            let up = take_shape(
                &mut language,
                &format!("{p}.mlp.up_proj.weight"),
                &[inter, hidden],
                DType::BF16,
            )?;
            let gate_up_w = upload(ctx, &transpose_2d(&concat_rows_bf16(&[&gate, &up])?)?)?;
            let down_w = upload(
                ctx,
                &transpose_2d(&take_shape(
                    &mut language,
                    &format!("{p}.mlp.down_proj.weight"),
                    &[hidden, inter],
                    DType::BF16,
                )?)?,
            )?;

            if text.layer_types[index] == "full_attention" {
                let q = take_shape(
                    &mut language,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[q_width, hidden],
                    DType::BF16,
                )?;
                let k = take_shape(
                    &mut language,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[kv_width, hidden],
                    DType::BF16,
                )?;
                let v = take_shape(
                    &mut language,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv_width, hidden],
                    DType::BF16,
                )?;
                layers.push(MixerWeights::FullAttention(FullAttentionWeights {
                    input_norm: upload(ctx, &input_norm)?,
                    q_w: upload(ctx, &transpose_2d(&q)?)?,
                    k_w: upload(ctx, &transpose_2d(&k)?)?,
                    v_w: upload(ctx, &transpose_2d(&v)?)?,
                    q_norm: upload(
                        ctx,
                        &take_shape(
                            &mut language,
                            &format!("{p}.self_attn.q_norm.weight"),
                            &[text.head_dim],
                            DType::BF16,
                        )?,
                    )?,
                    k_norm: upload(
                        ctx,
                        &take_shape(
                            &mut language,
                            &format!("{p}.self_attn.k_norm.weight"),
                            &[text.head_dim],
                            DType::BF16,
                        )?,
                    )?,
                    o_w: upload(
                        ctx,
                        &transpose_2d(&take_shape(
                            &mut language,
                            &format!("{p}.self_attn.o_proj.weight"),
                            &[hidden, text.num_attention_heads * text.head_dim],
                            DType::BF16,
                        )?)?,
                    )?,
                    post_norm: upload(ctx, &post_norm)?,
                    gate_up_w,
                    down_w,
                }));
            } else {
                let qkv = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.in_proj_qkv.weight"),
                    &[gdn_qkv, hidden],
                    DType::BF16,
                )?;
                let z = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.in_proj_z.weight"),
                    &[gdn_z, hidden],
                    DType::BF16,
                )?;
                let b = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.in_proj_b.weight"),
                    &[gdn_ab, hidden],
                    DType::BF16,
                )?;
                let a = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.in_proj_a.weight"),
                    &[gdn_ab, hidden],
                    DType::BF16,
                )?;
                let conv_dim = gdn_qkv;
                let conv = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.conv1d.weight"),
                    &[conv_dim, 1, text.linear_conv_kernel_dim],
                    DType::BF16,
                )?;
                let dt_bias = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.dt_bias"),
                    &[text.linear_num_value_heads],
                    DType::BF16,
                )?;
                let a_log = take_shape(
                    &mut language,
                    &format!("{p}.linear_attn.A_log"),
                    &[text.linear_num_value_heads],
                    DType::BF16,
                )?;
                layers.push(MixerWeights::Gdn(GdnWeights {
                    input_norm: upload(ctx, &input_norm)?,
                    qkv_w: upload(ctx, &transpose_2d(&qkv)?)?,
                    z_w: upload(ctx, &transpose_2d(&z)?)?,
                    b_w: upload(ctx, &transpose_2d(&b)?)?,
                    a_w: upload(ctx, &transpose_2d(&a)?)?,
                    conv_w: upload(
                        ctx,
                        &conv.reshape(vec![conv_dim, text.linear_conv_kernel_dim])?,
                    )?,
                    dt_bias: upload(ctx, &dt_bias)?,
                    a_log: upload(ctx, &a_log)?,
                    gated_norm: upload(
                        ctx,
                        &take_shape(
                            &mut language,
                            &format!("{p}.linear_attn.norm.weight"),
                            &[text.linear_value_head_dim],
                            DType::BF16,
                        )?,
                    )?,
                    out_w: upload(
                        ctx,
                        &transpose_2d(&take_shape(
                            &mut language,
                            &format!("{p}.linear_attn.out_proj.weight"),
                            &[hidden, gdn_z],
                            DType::BF16,
                        )?)?,
                    )?,
                    post_norm: upload(ctx, &post_norm)?,
                    gate_up_w,
                    down_w,
                }));
            }
        }

        let embed_tokens = upload(
            ctx,
            &take(&mut language, "language_model.embed_tokens.weight")?,
        )?;
        let final_norm = upload(ctx, &take(&mut language, "language_model.norm.weight")?)?;
        ensure_empty("language", &language)?;

        let vision_cfg = &config.model.vision_config;
        let patch_vec =
            3 * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size;
        let patch = take_shape(
            &mut visual,
            "visual.patch_embed.proj.weight",
            &[
                vision_cfg.hidden_size,
                3,
                vision_cfg.temporal_patch_size,
                vision_cfg.patch_size,
                vision_cfg.patch_size,
            ],
            DType::BF16,
        )?;
        let patch_w = transpose_2d(&patch.reshape(vec![vision_cfg.hidden_size, patch_vec])?)?;
        let mut blocks = Vec::with_capacity(vision_cfg.depth);
        for index in 0..vision_cfg.depth {
            let p = format!("visual.blocks.{index}");
            let h = vision_cfg.hidden_size;
            let i = vision_cfg.intermediate_size;
            blocks.push(VisionBlockWeights {
                norm1_w: upload(
                    ctx,
                    &take_shape(&mut visual, &format!("{p}.norm1.weight"), &[h], DType::BF16)?,
                )?,
                norm1_b: upload(
                    ctx,
                    &take_shape(&mut visual, &format!("{p}.norm1.bias"), &[h], DType::BF16)?,
                )?,
                qkv_w: upload(
                    ctx,
                    &transpose_2d(&take_shape(
                        &mut visual,
                        &format!("{p}.attn.qkv.weight"),
                        &[3 * h, h],
                        DType::BF16,
                    )?)?,
                )?,
                qkv_b: upload(
                    ctx,
                    &take_shape(
                        &mut visual,
                        &format!("{p}.attn.qkv.bias"),
                        &[3 * h],
                        DType::BF16,
                    )?,
                )?,
                proj_w: upload(
                    ctx,
                    &transpose_2d(&take_shape(
                        &mut visual,
                        &format!("{p}.attn.proj.weight"),
                        &[h, h],
                        DType::BF16,
                    )?)?,
                )?,
                proj_b: upload(
                    ctx,
                    &take_shape(
                        &mut visual,
                        &format!("{p}.attn.proj.bias"),
                        &[h],
                        DType::BF16,
                    )?,
                )?,
                norm2_w: upload(
                    ctx,
                    &take_shape(&mut visual, &format!("{p}.norm2.weight"), &[h], DType::BF16)?,
                )?,
                norm2_b: upload(
                    ctx,
                    &take_shape(&mut visual, &format!("{p}.norm2.bias"), &[h], DType::BF16)?,
                )?,
                fc1_w: upload(
                    ctx,
                    &transpose_2d(&take_shape(
                        &mut visual,
                        &format!("{p}.mlp.linear_fc1.weight"),
                        &[i, h],
                        DType::BF16,
                    )?)?,
                )?,
                fc1_b: upload(
                    ctx,
                    &take_shape(
                        &mut visual,
                        &format!("{p}.mlp.linear_fc1.bias"),
                        &[i],
                        DType::BF16,
                    )?,
                )?,
                fc2_w: upload(
                    ctx,
                    &transpose_2d(&take_shape(
                        &mut visual,
                        &format!("{p}.mlp.linear_fc2.weight"),
                        &[h, i],
                        DType::BF16,
                    )?)?,
                )?,
                fc2_b: upload(
                    ctx,
                    &take_shape(
                        &mut visual,
                        &format!("{p}.mlp.linear_fc2.bias"),
                        &[h],
                        DType::BF16,
                    )?,
                )?,
            });
        }
        let merged =
            vision_cfg.hidden_size * vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
        let vision = VisionWeights {
            patch_w: upload(ctx, &patch_w)?,
            patch_b: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.patch_embed.proj.bias",
                    &[vision_cfg.hidden_size],
                    DType::BF16,
                )?,
            )?,
            pos_embed: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.pos_embed.weight",
                    &[vision_cfg.num_position_embeddings, vision_cfg.hidden_size],
                    DType::BF16,
                )?,
            )?,
            blocks,
            merger_norm_w: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.merger.norm.weight",
                    &[vision_cfg.hidden_size],
                    DType::BF16,
                )?,
            )?,
            merger_norm_b: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.merger.norm.bias",
                    &[vision_cfg.hidden_size],
                    DType::BF16,
                )?,
            )?,
            merger_fc1_w: upload(
                ctx,
                &transpose_2d(&take_shape(
                    &mut visual,
                    "visual.merger.linear_fc1.weight",
                    &[merged, merged],
                    DType::BF16,
                )?)?,
            )?,
            merger_fc1_b: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.merger.linear_fc1.bias",
                    &[merged],
                    DType::BF16,
                )?,
            )?,
            merger_fc2_w: upload(
                ctx,
                &transpose_2d(&take_shape(
                    &mut visual,
                    "visual.merger.linear_fc2.weight",
                    &[vision_cfg.out_hidden_size, merged],
                    DType::BF16,
                )?)?,
            )?,
            merger_fc2_b: upload(
                ctx,
                &take_shape(
                    &mut visual,
                    "visual.merger.linear_fc2.bias",
                    &[vision_cfg.out_hidden_size],
                    DType::BF16,
                )?,
            )?,
        };
        ensure_empty("visual", &visual)?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            vision,
            readout: upload(ctx, &transpose_2d(&readout.weight)?)?,
        })
    }
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| Error::Other(format!("d3 weights: missing {name}")))
}

fn take_shape(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let tensor = take(map, name)?;
    expect_shape(Some(&tensor), name, shape, dtype)?;
    Ok(tensor)
}

fn expect_shape(tensor: Option<&Tensor>, name: &str, shape: &[usize], dtype: DType) -> Result<()> {
    let tensor = tensor.ok_or_else(|| Error::Other(format!("d3 weights: missing {name}")))?;
    if tensor.shape().dims() != shape || tensor.dtype() != dtype {
        return Err(Error::Other(format!(
            "d3 weights: {name} must be {dtype} {shape:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn ensure_empty(namespace: &str, map: &HashMap<String, Tensor>) -> Result<()> {
    if let Some(first) = map.keys().min() {
        return Err(Error::Other(format!(
            "d3 weights: {} unconsumed {namespace} tensors, first: {first}",
            map.len()
        )));
    }
    Ok(())
}

fn concat_rows_bf16(parts: &[&Tensor]) -> Result<Tensor> {
    let cols = parts
        .first()
        .and_then(|tensor| tensor.shape().dims().get(1).copied())
        .ok_or_else(|| Error::Other("d3 weights: packed projection requires 2D parts".into()))?;
    let mut rows = 0usize;
    let mut values = Vec::new();
    for tensor in parts {
        let dims = tensor.shape().dims();
        if dims.len() != 2 || dims[1] != cols || tensor.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "d3 weights: packed BF16 projection expected [out, {cols}], got {} {dims:?}",
                tensor.dtype()
            )));
        }
        rows += dims[0];
        values.extend_from_slice(tensor.as_bf16()?);
    }
    Tensor::from_bf16(vec![rows, cols], &values)
}
