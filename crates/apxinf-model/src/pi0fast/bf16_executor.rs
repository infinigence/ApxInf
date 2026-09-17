//! Native-BF16 π0-FAST transformer-layer execution.

use super::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{activation, attention, embedding, fused, gemm, norm, rope};

use super::{
    Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Pi0FastLanguageConfig,
    VisionPatchEmbeddingF32,
};

pub struct Bf16LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_bf16(
    ctx: &Context,
    config: Pi0FastLanguageConfig,
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
    let activated = gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )?;
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    let hidden =
        fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Bf16LanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

/// SigLIP patch embedding: FP32 projection, FP32 bias/position add, BF16 output.
///
/// PaliGemma's `SiglipVisionEmbeddings` upcasts the pixels to the projection
/// dtype, convolves in FP32, adds the learned position embedding in FP32, and
/// only then lets `SiglipVisionTransformer` cast into the BF16 encoder. Keeping
/// that split matters numerically: doing the projection in BF16 changes the
/// autoregressive action tokens.
pub fn vision_patch_embed_f32_bf16(
    ctx: &Context,
    weights: &VisionPatchEmbeddingF32,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm::matmul(ctx, patches, &weights.weight)?;
    embedding::add_position_f32_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        &weights.position,
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

/// Project one activation row through `weight`.
///
/// This is the single seam between the decode layer and its projection
/// backend, and it is deliberately a plain `gemm::bf16` call.
///
/// A dedicated matrix-vector kernel was tried here and removed. At one token
/// every projection is a pure streaming read of the weight, and cuBLAS only
/// reaches 100-130 GB/s on some of those shapes against the ~178 GB/s the
/// memory system sustains, so a purpose-built kernel was worth up to 1.8x on
/// those reads. It was dropped because the decode loop has no budget for the
/// rounding difference: the kernel sums `K` in a different order than cuBLAS,
/// and argmax over a 257152-wide vocabulary turns one-ulp ties into different
/// action tokens.
///
/// Even restricting the kernel to the shapes where it matched cuBLAS to the
/// last BF16 step was not enough. Audited on the shipped checkpoint, `qkv`,
/// `gate_up` and `down` had zero elements beyond one ulp out of 32.3M, yet
/// still differed by exactly one ulp on roughly 4800 elements per frame, and
/// the LIBERO-10 gate moved from 7/10 to 3/8. See
/// `devlocal/pi0-fast/reports/18-decode-gemv-dispatch.md` for the
/// measurements; the short version is that this path has to stay bit-exact
/// with the configuration the model was validated against.
fn projection(ctx: &Context, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    gemm::bf16(ctx, input, weight)
}

/// One cached (single-token) language layer evaluated through the decode
/// projection seam so a decode-specific backend has a place to live.
///
/// This is `language_layer_cached_bf16` with the same norms, the same RoPE,
/// the same fused residual paths and the same `gelu(gate) * up` epilogue, but
/// every projection goes through [`projection`]. Keeping the GeGLU as a matmul
/// followed by `activation::geglu_bf16` instead of the fused epilogue costs a
/// few tens of kilobytes of traffic at one token, against a hundred megabytes
/// of weights.
#[allow(clippy::too_many_arguments)]
pub fn language_layer_cached_decode_bf16(
    ctx: &Context,
    config: Pi0FastLanguageConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = projection(ctx, &normalized, &weights.qkv.weight)?;
    let q = rope::apply_q_write_kv_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        key_cache,
        value_cache,
        position_offset,
    )?;
    let attention = attention::mqa_bf16(ctx, &q, key_cache, value_cache, position_offset + tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = projection(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = projection(ctx, &fused.normalized, &weights.gate_up.weight)?;
    let activated = activation::geglu_bf16(ctx, &gate_up)?;
    let projected = projection(ctx, &activated, &weights.down.weight)?;
    fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)
}

/// The LM head uses the same decode-tuned projection; at one token it is the
/// widest projection in the model and dominates the per-step weight read.
pub(super) fn decode_projection(ctx: &Context, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    projection(ctx, input, weight)
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_cached_bf16(
    ctx: &Context,
    config: Pi0FastLanguageConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
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
        key_cache,
        value_cache,
        position_offset,
    )?;
    let attention =
        attention::mqa_bf16(ctx, &q, key_cache, value_cache, position_offset + tokens)?.reshape(
            vec![tokens, config.num_heads * config.head_dim],
        )?;
    let projected = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let activated = gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )?;
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)
}
