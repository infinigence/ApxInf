//! Native-BF16 π0.5 transformer-layer execution.

use super::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{activation, attention, embedding, fused, gemm, norm, rope};

use super::{
    Bf16DeviceActionLayer, Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Bf16LinearWeights,
    GemmaVariantConfig,
};

pub struct Bf16LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Bf16ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16LanguageLayerOutput> {
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let qkv = rope::split_qkv_apply_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
    )?;
    let tokens = input.shape().dims()[0];
    if !compute_tail {
        return Ok(Bf16LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.key_2d(tokens, config.head_dim)?,
            value: qkv.value_2d(tokens, config.head_dim)?,
        });
    }
    let attention = attention::mqa_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let activated = match gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )? {
        Some(value) => value,
        None => {
            let gate_up = gemm::bf16(ctx, &fused.normalized, &weights.gate_up.weight)?;
            activation::geglu_bf16(ctx, &gate_up)?
        }
    };
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    let hidden =
        fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Bf16LanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_style: &Tensor,
    mlp_style: &Tensor,
    next_norm_style: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => norm::adaptive_rms_bf16(ctx, input, attention_style, rms_eps)?,
    };
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let q = rope::apply_q_write_kv_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
        position_offset,
    )?;
    let attention = attention::mqa_bf16(
        ctx,
        &q,
        prefix_k,
        prefix_v,
        position_offset + input.shape().dims()[0],
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let projected = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        input,
        attention_style,
        mlp_style,
        rms_eps,
    )?;
    let activated = match gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )? {
        Some(value) => value,
        None => {
            let gate_up = gemm::bf16(ctx, &fused.normalized, &weights.gate_up.weight)?;
            activation::geglu_bf16(ctx, &gate_up)?
        }
    };
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        &fused.hidden,
        mlp_style,
        next_norm_style,
        rms_eps,
    )?;
    Ok(Bf16ActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

pub fn vision_patch_embed_bf16(
    ctx: &Context,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm::bf16(ctx, patches, &weights.weight)?;
    embedding::add_position_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_bf16(
    ctx: &Context,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = norm::layer_bf16(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let qkv =
        attention::split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = attention::mha_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = gemm::bf16(ctx, &fused.normalized, &weights.fc1.weight)?;
    let activation = activation::bias_gelu_bf16(ctx, &activation, weights.fc1.bias.as_ref())?;
    let projection = gemm::bf16(ctx, &activation, &weights.fc2.weight)?;
    fused::bias_residual_bf16(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

trait QkvViews {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
}

impl QkvViews for rope::QkvTensors {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.k.reshape(vec![tokens, head_dim])
    }

    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.v.reshape(vec![tokens, head_dim])
    }
}
