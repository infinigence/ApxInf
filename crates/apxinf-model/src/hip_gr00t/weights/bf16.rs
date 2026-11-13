//! GR00T N1.7 weights on a HIP device, in BF16.
//!
//! Built from this port's [`super::checkpoint`] host tensors. Every matrix
//! keeps the row-major `[in, out]` the checkpoint reader produced, so the
//! native GEMM consumes it directly.
//!
//! Unlike `gr00t/`, there is no `DeviceLinearWeights` trait and no precision
//! dispatch: gfx1151 has no FP8 tensor cores, so BF16 is the only path and the
//! concrete types are written out. A self-attending DiT block's q/k/v are
//! packed along N into one `[k, 3n]` matrix (`qkv_fused`), displacing rather
//! than duplicating the three separate ones, so one GEMM and an in-place
//! strided attention read replace three GEMMs and a split. Cross-attending
//! blocks keep separate projections: their Q reads the 1536-wide state while
//! K and V read the 2048-wide backbone rows, so they cannot share a GEMM.

use apxinf_core::{Device, DType, Error, Result, Shape, Tensor};
use apxinf_hip::HipContext;

use super::host::action_head as host_action;
use super::host::text as host_text;
use super::host::vision as host_vision;
use super::host::Gr00tWeights;
use crate::hip_gr00t::config::{Gr00tConfig, Qwen3VLConfig};

/// A row-major `[in, out]` projection and its bias.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// Affine normalization parameters.
pub struct Norm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub struct TextLayer {
    pub attn_norm: Tensor,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub ffn_norm: Tensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct VisionBlock {
    pub norm1: Norm,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: Norm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct Merger {
    pub norm: Norm,
    pub fc1: Linear,
    pub fc2: Linear,
}

/// One embodiment's state/action adapters, already selected out of the
/// checkpoint's `[embodiment, in, out]` category tensors.
pub struct Embodiment {
    pub state_encoder_in: Linear,
    pub state_encoder_out: Linear,
    pub action_encoder_in: Linear,
    pub action_encoder_time: Linear,
    pub action_encoder_out: Linear,
    pub action_decoder_in: Linear,
    pub action_decoder_out: Linear,
}

/// One of the action head's VL self-attention blocks -- a diffusers
/// `BasicTransformerBlock`: LayerNorm, non-causal self-attention with biases
/// everywhere, LayerNorm, then a plain (not gated) GELU feed-forward.
pub struct VlBlock {
    pub attn_norm: Norm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub ffn_norm: Norm,
    pub ffn_in: Linear,
    pub ffn_out: Linear,
}

pub struct DitBlock {
    pub adaptive_norm: Linear,
    /// q/k/v concatenated along N, built only in fused mode and only for
    /// self-attending blocks (whose three projections share the same K).
    pub qkv_fused: Option<Linear>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub ffn_in: Linear,
    pub ffn_out: Linear,
}

pub struct HipGr00tWeights {
    // Vision tower.
    pub patch_embed: Linear,
    // The 48x48 learned position table stays on the host: the tower needs it
    // bilinearly interpolated to the actual grid, and `vision::prepare` does
    // that once per shape with HF's BF16 rounding points.
    pub vision_blocks: Vec<VisionBlock>,
    pub merger: Merger,
    pub deepstack_mergers: Vec<Merger>,
    // Text stack, truncated to `select_layer`.
    pub token_embedding: Tensor,
    pub text_layers: Vec<TextLayer>,
    pub text_output_norm: Tensor,
    // Action head.
    pub backbone_layer_norm: Option<Norm>,
    pub vl_self_attention: Vec<VlBlock>,
    pub action_position_embedding: Option<Tensor>,
    pub timestep_in: Linear,
    pub timestep_out: Linear,
    pub dit_blocks: Vec<DitBlock>,
    pub output_modulation: Linear,
    pub output_projection: Linear,
}

impl HipGr00tWeights {
    /// Upload every weight the BF16 inference path reads.
    ///
    /// `embodiment_id` selects one row out of the checkpoint's per-embodiment
    /// adapters; the fixture and the NVIDIA reference both use `libero_sim`.
    pub fn from_checkpoint(
        weights: &Gr00tWeights,
        config: &Gr00tConfig,
        backbone: &Qwen3VLConfig,
        embodiment_id: usize,
        ctx: &HipContext,
    ) -> Result<(Self, Embodiment)> {
        let vision = &weights.backbone_vision;
        let text = &weights.backbone_text;
        let action = &weights.action_head;
        if text.layers.len() != config.select_layer {
            return Err(Error::Other(format!(
                "GR00T text stack has {} layers but config selects {}",
                text.layers.len(),
                config.select_layer
            )));
        }
        if action.dit_blocks.len() != config.diffusion.num_layers {
            return Err(Error::Other(format!(
                "GR00T checkpoint has {} DiT blocks but config declares {}",
                action.dit_blocks.len(),
                config.diffusion.num_layers
            )));
        }
        if vision.blocks.len() != backbone.vision.depth {
            return Err(Error::Other(format!(
                "GR00T vision tower has {} blocks but the Cosmos config declares {}",
                vision.blocks.len(),
                backbone.vision.depth
            )));
        }
        let declared = config
            .vl_self_attention
            .as_ref()
            .map_or(0, |value| value.num_layers);
        if action.vl_self_attention.len() != declared {
            return Err(Error::Other(format!(
                "GR00T checkpoint has {} VL self-attention blocks but the config declares {declared}",
                action.vl_self_attention.len(),
            )));
        }
        let device = Self {
            patch_embed: linear_with_bias(&vision.patch_embed_weight, &vision.patch_embed_bias, ctx)?,
            vision_blocks: vision
                .blocks
                .iter()
                .map(|block| vision_block(block, ctx))
                .collect::<Result<_>>()?,
            merger: merger(&vision.merger, ctx)?,
            deepstack_mergers: vision
                .deepstack_mergers
                .iter()
                .map(|value| merger(value, ctx))
                .collect::<Result<_>>()?,
            token_embedding: to_device(&text.token_embedding, ctx)?,
            text_layers: text
                .layers
                .iter()
                .map(|layer| text_layer(layer, ctx))
                .collect::<Result<_>>()?,
            text_output_norm: to_device(&text.output_norm_weight, ctx)?,
            backbone_layer_norm: action
                .backbone_layer_norm
                .as_ref()
                .map(|value| norm(&value.weight, Some(&value.bias), ctx))
                .transpose()?,
            vl_self_attention: action
                .vl_self_attention
                .iter()
                .map(|block| vl_block(block, ctx))
                .collect::<Result<_>>()?,
            // The checkpoint stores `max_seq_len` rows; only the first
            // `action_horizon` are ever read, so the rest never reach the
            // device.
            action_position_embedding: action
                .position_embedding
                .as_ref()
                .map(|value| first_rows(value, config.action_horizon, ctx))
                .transpose()?,
            timestep_in: dense(&action.timestep_input, ctx)?,
            timestep_out: dense(&action.timestep_output, ctx)?,
            dit_blocks: action
                .dit_blocks
                .iter()
                .map(|block| dit_block(block, ctx))
                .collect::<Result<_>>()?,
            output_modulation: dense(&action.output_modulation, ctx)?,
            output_projection: dense(&action.output_projection, ctx)?,
        };
        let embodiment = select_embodiment(action, embodiment_id, config, ctx)?;
        Ok((device, embodiment))
    }
}

fn text_layer(layer: &host_text::Qwen3VLLayer, ctx: &HipContext) -> Result<TextLayer> {
    Ok(TextLayer {
        attn_norm: to_device(&layer.attn_norm_weight, ctx)?,
        q: linear(&layer.wq, ctx)?,
        k: linear(&layer.wk, ctx)?,
        v: linear(&layer.wv, ctx)?,
        output: linear(&layer.wo, ctx)?,
        q_norm: to_device(&layer.q_norm_weight, ctx)?,
        k_norm: to_device(&layer.k_norm_weight, ctx)?,
        ffn_norm: to_device(&layer.ffn_norm_weight, ctx)?,
        gate: linear(&layer.w_gate, ctx)?,
        up: linear(&layer.w_up, ctx)?,
        down: linear(&layer.w_down, ctx)?,
    })
}

fn vision_block(block: &host_vision::Qwen3VLVisionBlock, ctx: &HipContext) -> Result<VisionBlock> {
    Ok(VisionBlock {
        norm1: norm(&block.norm1_w, Some(&block.norm1_b), ctx)?,
        qkv: linear_with_bias(&block.qkv_w, &block.qkv_b, ctx)?,
        proj: linear_with_bias(&block.proj_w, &block.proj_b, ctx)?,
        norm2: norm(&block.norm2_w, Some(&block.norm2_b), ctx)?,
        fc1: linear_with_bias(&block.fc1_w, &block.fc1_b, ctx)?,
        fc2: linear_with_bias(&block.fc2_w, &block.fc2_b, ctx)?,
    })
}

fn merger(value: &host_vision::Qwen3VLMerger, ctx: &HipContext) -> Result<Merger> {
    Ok(Merger {
        norm: norm(&value.norm_w, Some(&value.norm_b), ctx)?,
        fc1: linear_with_bias(&value.fc1_w, &value.fc1_b, ctx)?,
        fc2: linear_with_bias(&value.fc2_w, &value.fc2_b, ctx)?,
    })
}

fn vl_block(
    block: &host_action::Gr00tVlSelfAttentionBlockWeights,
    ctx: &HipContext,
) -> Result<VlBlock> {
    Ok(VlBlock {
        attn_norm: norm(
            &block.attention_norm.weight,
            Some(&block.attention_norm.bias),
            ctx,
        )?,
        q: dense(&block.attention.query, ctx)?,
        k: dense(&block.attention.key, ctx)?,
        v: dense(&block.attention.value, ctx)?,
        output: dense(&block.attention.output, ctx)?,
        ffn_norm: norm(
            &block.feed_forward_norm.weight,
            Some(&block.feed_forward_norm.bias),
            ctx,
        )?,
        ffn_in: dense(&block.feed_forward.input, ctx)?,
        ffn_out: dense(&block.feed_forward.output, ctx)?,
    })
}

fn dit_block(block: &host_action::Gr00tDitBlockWeights, ctx: &HipContext) -> Result<DitBlock> {
    let qkv_fused = fused_qkv(block, ctx)?;
    // In fused mode a self-attending block reads only the packed matrix, so
    // its separate q/k/v never reach the device. Empty placeholders keep the
    // struct non-Optional; an accidental use fails on a shape or device check
    // rather than quietly reading something stale.
    let (q, k, v) = if qkv_fused.is_some() {
        (empty_linear()?, empty_linear()?, empty_linear()?)
    } else {
        (
            dense(&block.attention.query, ctx)?,
            dense(&block.attention.key, ctx)?,
            dense(&block.attention.value, ctx)?,
        )
    };
    Ok(DitBlock {
        adaptive_norm: dense(&block.adaptive_norm, ctx)?,
        qkv_fused,
        q,
        k,
        v,
        output: dense(&block.attention.output, ctx)?,
        ffn_in: dense(&block.feed_forward.input, ctx)?,
        ffn_out: dense(&block.feed_forward.output, ctx)?,
    })
}

/// Concatenate a DiT block's q/k/v weights along the output dimension.
///
/// Only self-attending blocks qualify: a cross-attending block projects K/V
/// from the 2048-wide backbone rows while Q comes from the 1536-wide DiT
/// state, so the three matrices do not share K and cannot be one GEMM.
/// Packing happens on the host, before anything is uploaded, so the fused
/// path reads exactly the same BF16 bytes as the three separate projections.
fn fused_qkv(
    block: &host_action::Gr00tDitBlockWeights,
    ctx: &HipContext,
) -> Result<Option<Linear>> {
    if crate::hip_gr00t::model::blocks::qkv::mode() != crate::hip_gr00t::model::blocks::qkv::QkvMode::Fused {
        return Ok(None);
    }
    let (q, k, v) = (
        &block.attention.query,
        &block.attention.key,
        &block.attention.value,
    );
    let shape = q.weight.shape().dims();
    if shape.len() != 2
        || k.weight.shape().dims() != shape
        || v.weight.shape().dims() != shape
    {
        // Cross-attention block: its K/V use a different input width.
        return Ok(None);
    }
    let (kin, n) = (shape[0], shape[1]);
    let element = DType::BF16.size_in_bytes();
    fn cpu(tensor: &Tensor) -> Result<&[u8]> {
        tensor.storage().as_cpu().ok_or_else(|| {
            Error::Other("apxinf: q/k/v packing needs host-side tensors".into())
        })
    }
    let mut weight_bytes = Vec::with_capacity(kin * 3 * n * element);
    for row in 0..kin {
        for source in [&q.weight, &k.weight, &v.weight] {
            let bytes = cpu(source)?;
            let start = row * n * element;
            weight_bytes.extend_from_slice(&bytes[start..start + n * element]);
        }
    }
    let mut bias_bytes = Vec::with_capacity(3 * n * element);
    for source in [&q.bias, &k.bias, &v.bias] {
        bias_bytes.extend_from_slice(cpu(source)?);
    }
    let weight = Tensor::from_raw(
        Shape::new(vec![kin, 3 * n]),
        DType::BF16,
        Device::Cpu,
        weight_bytes,
    )?;
    let bias = Tensor::from_raw(
        Shape::new(vec![3 * n]),
        DType::BF16,
        Device::Cpu,
        bias_bytes,
    )?;
    Ok(Some(Linear {
        weight: to_device(&weight, ctx)?,
        bias: Some(to_device(&bias, ctx)?),
    }))
}

/// A `[0, 0]` placeholder for a projection that was packed into `qkv_fused`
/// and is never read. Any accidental device-side use fails on a shape or
/// device check rather than quietly reading something stale.
fn empty_linear() -> Result<Linear> {
    Ok(Linear {
        weight: Tensor::from_raw(Shape::new(vec![0, 0]), DType::BF16, Device::Cpu, Vec::new())?,
        bias: None,
    })
}

/// Select one embodiment's adapters. The checkpoint stores them as
/// `[embodiment, in, out]`; slicing happens on the host so only the selected
/// matrices reach the device.
///
/// The returned tensors own their device allocations, so dropping an
/// `Embodiment` frees them -- switching embodiment at runtime costs one upload
/// and does not accumulate.
pub fn select_embodiment(
    action: &host_action::Gr00tActionHeadWeights,
    id: usize,
    config: &Gr00tConfig,
    ctx: &HipContext,
) -> Result<Embodiment> {
    if id >= config.max_num_embodiments {
        return Err(Error::Other(format!(
            "GR00T embodiment {id} is outside 0..{}",
            config.max_num_embodiments
        )));
    }
    Ok(Embodiment {
        state_encoder_in: category(&action.state_encoder.input, id, ctx)?,
        state_encoder_out: category(&action.state_encoder.output, id, ctx)?,
        action_encoder_in: category(&action.action_encoder_input, id, ctx)?,
        action_encoder_time: category(&action.action_encoder_time, id, ctx)?,
        action_encoder_out: category(&action.action_encoder_output, id, ctx)?,
        action_decoder_in: category(&action.action_decoder.input, id, ctx)?,
        action_decoder_out: category(&action.action_decoder.output, id, ctx)?,
    })
}

/// One embodiment's slice of a `[embodiment, in, out]` weight and
/// `[embodiment, out]` bias.
fn category(
    weights: &host_action::Gr00tCategoryLinearWeights,
    id: usize,
    ctx: &HipContext,
) -> Result<Linear> {
    let dims = weights.weight.shape().dims();
    if dims.len() != 3 {
        return Err(Error::Other(format!(
            "GR00T category weight must be [embodiment, in, out], got {dims:?}"
        )));
    }
    let (embodiments, rows, columns) = (dims[0], dims[1], dims[2]);
    if id >= embodiments {
        return Err(Error::Other(format!(
            "GR00T category weight holds {embodiments} embodiments, asked for {id}"
        )));
    }
    let stride = rows * columns;
    let source = weights.weight.as_bf16()?;
    let weight = Tensor::from_bf16(
        vec![rows, columns],
        &source[id * stride..(id + 1) * stride],
    )?;
    let bias_dims = weights.bias.shape().dims();
    if bias_dims.len() != 2 || bias_dims[0] != embodiments || bias_dims[1] != columns {
        return Err(Error::Other(format!(
            "GR00T category bias must be [{embodiments}, {columns}], got {bias_dims:?}"
        )));
    }
    let bias_source = weights.bias.as_bf16()?;
    let bias = Tensor::from_bf16(
        vec![columns],
        &bias_source[id * columns..(id + 1) * columns],
    )?;
    Ok(Linear {
        weight: to_device(&weight, ctx)?,
        bias: Some(to_device(&bias, ctx)?),
    })
}

/// Upload the leading `rows` of a host matrix.
fn first_rows(tensor: &Tensor, rows: usize, ctx: &HipContext) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 || dims[0] < rows {
        return Err(Error::Other(format!(
            "GR00T expected at least {rows} rows of a matrix, got {dims:?}"
        )));
    }
    let columns = dims[1];
    let source = tensor.as_bf16()?;
    let sliced = Tensor::from_bf16(vec![rows, columns], &source[..rows * columns])?;
    to_device(&sliced, ctx)
}

fn dense(weights: &host_action::Gr00tLinearWeights, ctx: &HipContext) -> Result<Linear> {
    linear_with_bias(&weights.weight, &weights.bias, ctx)
}

fn linear(weight: &Tensor, ctx: &HipContext) -> Result<Linear> {
    Ok(Linear {
        weight: to_device(weight, ctx)?,
        bias: None,
    })
}

fn linear_with_bias(weight: &Tensor, bias: &Tensor, ctx: &HipContext) -> Result<Linear> {
    Ok(Linear {
        weight: to_device(weight, ctx)?,
        bias: Some(to_device(bias, ctx)?),
    })
}

fn norm(weight: &Tensor, bias: Option<&Tensor>, ctx: &HipContext) -> Result<Norm> {
    Ok(Norm {
        weight: to_device(weight, ctx)?,
        bias: bias.map(|value| to_device(value, ctx)).transpose()?,
    })
}

/// Move one host tensor to the device, rejecting anything but BF16: the whole
/// inference path is BF16 and a silent dtype surprise here would surface as a
/// wrong result much later.
fn to_device(tensor: &Tensor, ctx: &HipContext) -> Result<Tensor> {
    if tensor.dtype() != apxinf_core::DType::BF16 {
        return Err(Error::UnsupportedDType {
            got: tensor.dtype(),
            allowed: "bf16",
        });
    }
    ctx.to_device(tensor)
}
