//! Native-BF16 π0-FAST transformer-layer execution.

use super::backend::{ops, Context};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use ops::{embedding, gemm, pointwise};

use super::{
    Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Pi0FastLanguageConfig, VisionPatchEmbeddingF32,
};

pub struct Bf16LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub(super) fn plain_gemm(ctx: &Context, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let a = input.shape().dims();
    let b = weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "PI0-FAST GEMM shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    gemm::gemm(ctx, gemm::GemmArgs::new(input, weight, &mut output))?;
    Ok(output)
}

fn bias_activation(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

fn geglu(ctx: &Context, gate_up: &Tensor) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other(
            "PI0-FAST GeGLU requires [rows,2*cols] input".into(),
        ));
    }
    let mut output =
        ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), gate_up.dtype())?;
    let args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, gate_up, &mut output);
    ops::pointwise(ctx, args)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let tokens =
        qkv.shape().dims().first().copied().ok_or_else(|| {
            Error::Other("PI0-FAST packed QKV must have a token dimension".into())
        })?;
    let mut query =
        ctx.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), qkv.dtype())?;
    let kv_shape = Shape::new(vec![tokens, kv_heads, head_dim]);
    let mut key = ctx.allocate_output(kv_shape.clone(), qkv.dtype())?;
    let mut value = ctx.allocate_output(kv_shape, qkv.dtype())?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut query,
            k: &mut key,
            v: &mut value,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok((query, key, value))
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope_into_cache(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    key_cache: &Tensor,
    value_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    let tokens =
        qkv.shape().dims().first().copied().ok_or_else(|| {
            Error::Other("PI0-FAST packed QKV must have a token dimension".into())
        })?;
    let mut query =
        ctx.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), qkv.dtype())?;
    let mut key = key_cache.clone();
    let mut value = value_cache.clone();
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut query,
            k: &mut key,
            v: &mut value,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: output_offset,
        },
    )?;
    Ok(query)
}

fn geglu_gemm(ctx: &Context, input: &Tensor, packed_weight: &Tensor) -> Result<Tensor> {
    let a = input.shape().dims();
    let b = packed_weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || b[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "PI0-FAST GeGLU GEMM shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1] / 2]), DType::BF16)?;
    let gemm = gemm::GemmArgs::new(input, packed_weight, &mut output);
    gemm::gemm_geglu(ctx, gemm::GemmGegluArgs { gemm })?;
    Ok(output)
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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::rms(input, &weights.input_norm_scale, &normalized, rms_eps),
    )?;
    let qkv = plain_gemm(ctx, &normalized, &weights.qkv.weight)?;
    let (query, key, value) = split_qkv_rope(
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
            key: key.reshape(vec![tokens, config.head_dim])?,
            value: value.reshape(vec![tokens, config.head_dim])?,
        });
    }
    let attention = super::attention_mqa(ctx, &query, &key, &value, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = plain_gemm(ctx, &attention, &weights.output.weight)?;
    let residual = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input: &projected,
            bias: weights.output.bias.as_ref(),
            residual: input,
            gate: None,
            hidden: &residual,
            normalized: Some(&normalized),
            kind: ops::NormKind::Rms,
            affine: ops::Affine::Weight {
                weight: &weights.post_attention_norm_scale,
                bias: None,
                offset: false,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            bias_order: ops::BiasOrder::SingleRound,
            eps: rms_eps,
        },
    )?;
    let activated = geglu_gemm(ctx, &normalized, &weights.gate_up.weight)?;
    let projected = plain_gemm(ctx, &activated, &weights.down.weight)?;
    let hidden = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::residual_add(
        ctx,
        ops::ResidualAddArgs {
            input: &projected,
            bias: weights.down.bias.as_ref(),
            residual: &residual,
            gate: None,
            hidden: &hidden,
            bias_order: ops::BiasOrder::SingleRound,
        },
    )?;
    Ok(Bf16LanguageLayerOutput {
        hidden,
        key: key.reshape(vec![tokens, config.head_dim])?,
        value: value.reshape(vec![tokens, config.head_dim])?,
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
    embedding::add_position_projected(
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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::layer(
            input,
            &weights.norm1.weight,
            &weights.norm1.bias,
            &normalized,
            layer_norm_eps,
        ),
    )?;
    let qkv = plain_gemm(ctx, &normalized, &weights.qkv.weight)?;
    let qkv = super::split_qkv_bias(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = super::attention_mha(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = plain_gemm(ctx, &attention, &weights.output.weight)?;
    let hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input: &projection,
            bias: weights.output.bias.as_ref(),
            residual: input,
            gate: None,
            hidden: &hidden,
            normalized: Some(&normalized),
            kind: ops::NormKind::Layer,
            affine: ops::Affine::Weight {
                weight: &weights.norm2.weight,
                bias: Some(&weights.norm2.bias),
                offset: false,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            bias_order: ops::BiasOrder::SingleRound,
            eps: layer_norm_eps,
        },
    )?;
    let activation = plain_gemm(ctx, &normalized, &weights.fc1.weight)?;
    let activation = bias_activation(
        ctx,
        &activation,
        weights.fc1.bias.as_ref(),
        pointwise::PointwiseActivation::Gelu,
    )?;
    let projection = plain_gemm(ctx, &activation, &weights.fc2.weight)?;
    let output = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ops::residual_add(
        ctx,
        ops::ResidualAddArgs {
            input: &projection,
            bias: weights.fc2.bias.as_ref(),
            residual: &hidden,
            gate: None,
            hidden: &output,
            bias_order: ops::BiasOrder::SingleRound,
        },
    )?;
    Ok(output)
}

/// Project one activation row through `weight`.
///
/// This is the single seam between the decode layer and its projection
/// backend, and it deliberately uses the formal GEMM semantic.
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
    plain_gemm(ctx, input, weight)
}

/// One cached (single-token) language layer evaluated through the decode
/// projection seam so a decode-specific backend has a place to live.
///
/// This is `language_layer_cached_bf16` with the same norms, the same RoPE,
/// the same fused residual paths and the same `gelu(gate) * up` epilogue, but
/// every projection goes through [`projection`]. Keeping the GeGLU as a matmul
/// followed by `pointwise::geglu` instead of the fused epilogue costs a
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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::rms(input, &weights.input_norm_scale, &normalized, rms_eps),
    )?;
    let qkv = projection(ctx, &normalized, &weights.qkv.weight)?;
    let q = split_qkv_rope_into_cache(
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
        super::attention_mqa(ctx, &q, key_cache, value_cache, position_offset + tokens)?
            .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = projection(ctx, &attention, &weights.output.weight)?;
    let residual = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input: &projected,
            bias: weights.output.bias.as_ref(),
            residual: input,
            gate: None,
            hidden: &residual,
            normalized: Some(&normalized),
            kind: ops::NormKind::Rms,
            affine: ops::Affine::Weight {
                weight: &weights.post_attention_norm_scale,
                bias: None,
                offset: false,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            bias_order: ops::BiasOrder::SingleRound,
            eps: rms_eps,
        },
    )?;
    let gate_up = projection(ctx, &normalized, &weights.gate_up.weight)?;
    let activated = geglu(ctx, &gate_up)?;
    let projected = projection(ctx, &activated, &weights.down.weight)?;
    let hidden = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::residual_add(
        ctx,
        ops::ResidualAddArgs {
            input: &projected,
            bias: weights.down.bias.as_ref(),
            residual: &residual,
            gate: None,
            hidden: &hidden,
            bias_order: ops::BiasOrder::SingleRound,
        },
    )?;
    Ok(hidden)
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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::rms(input, &weights.input_norm_scale, &normalized, rms_eps),
    )?;
    let qkv = plain_gemm(ctx, &normalized, &weights.qkv.weight)?;
    let q = split_qkv_rope_into_cache(
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
        super::attention_mqa(ctx, &q, key_cache, value_cache, position_offset + tokens)?
            .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = plain_gemm(ctx, &attention, &weights.output.weight)?;
    let residual = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input: &projected,
            bias: weights.output.bias.as_ref(),
            residual: input,
            gate: None,
            hidden: &residual,
            normalized: Some(&normalized),
            kind: ops::NormKind::Rms,
            affine: ops::Affine::Weight {
                weight: &weights.post_attention_norm_scale,
                bias: None,
                offset: false,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            bias_order: ops::BiasOrder::SingleRound,
            eps: rms_eps,
        },
    )?;
    let activated = geglu_gemm(ctx, &normalized, &weights.gate_up.weight)?;
    let projected = plain_gemm(ctx, &activated, &weights.down.weight)?;
    let hidden = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::residual_add(
        ctx,
        ops::ResidualAddArgs {
            input: &projected,
            bias: weights.down.bias.as_ref(),
            residual: &residual,
            gate: None,
            hidden: &hidden,
            bias_order: ops::BiasOrder::SingleRound,
        },
    )?;
    Ok(hidden)
}
