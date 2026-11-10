//! W8A8 INT8 π0.5 transformer-layer execution.

use super::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{activation, attention, embedding, fused, norm, rope};

use super::{
    GemmaVariantConfig, Int8DeviceActionLayer, Int8DeviceLanguageLayer, Int8DeviceVisionBlock,
    Int8LinearWeights,
};

pub struct Int8LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Int8ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_int8(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Int8LanguageLayerOutput> {
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = weights.qkv.gemm(ctx, &normalized)?;
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
        return Ok(Int8LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.key_2d(tokens, config.head_dim)?,
            value: qkv.value_2d(tokens, config.head_dim)?,
        });
    }
    let attention = attention::mqa_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = weights.output.gemm(ctx, &attention)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = weights.gate_up.gemm(ctx, &fused.normalized)?;
    let activated = activation::geglu_bf16(ctx, &gate_up)?;
    let projected = weights.down.gemm(ctx, &activated)?;
    let hidden =
        fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Int8LanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_int8(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DeviceActionLayer,
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
) -> Result<Int8ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => norm::adaptive_rms_bf16(ctx, input, attention_style, rms_eps)?,
    };
    let qkv = weights.qkv.gemm(ctx, &normalized)?;
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
    let projected = weights.output.gemm(ctx, &attention)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        input,
        attention_style,
        mlp_style,
        rms_eps,
    )?;
    let gate_up = weights.gate_up.gemm(ctx, &fused.normalized)?;
    let activated = activation::geglu_bf16(ctx, &gate_up)?;
    let projected = weights.down.gemm(ctx, &activated)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        &fused.hidden,
        mlp_style,
        next_norm_style,
        rms_eps,
    )?;
    Ok(Int8ActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

pub fn vision_patch_embed_int8(
    ctx: &Context,
    weights: &Int8LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = weights.gemm(ctx, patches)?;
    embedding::add_position_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_int8(
    ctx: &Context,
    weights: &Int8DeviceVisionBlock,
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
    let qkv = weights.qkv.gemm(ctx, &normalized)?;
    let qkv =
        attention::split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = attention::mha_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = weights.output.gemm(ctx, &attention)?;
    let fused = fused::bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = weights.fc1.gemm(ctx, &fused.normalized)?;
    let activation = activation::bias_gelu_bf16(ctx, &activation, weights.fc1.bias.as_ref())?;
    let projection = weights.fc2.gemm(ctx, &activation)?;
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
