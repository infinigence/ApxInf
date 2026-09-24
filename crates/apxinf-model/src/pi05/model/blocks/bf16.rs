//! Native-BF16 π0.5 transformer-layer computation.

use crate::pi05::backend::{
    ops::{
        attention as l3_attention, concat_rows as l3_concat_rows, gather as l3_gather,
        adaptive_rms_norm as l3_adaptive_rms_norm, bias_residual as l3_bias_residual,
        bias_residual_layer_norm as l3_bias_residual_layer_norm,
        bias_residual_rms_norm as l3_bias_residual_rms_norm, gemm as l3_gemm,
        gemm_geglu as l3_gemm_geglu, layer_norm as l3_layer_norm,
        pointwise as l3_pointwise, rms_norm as l3_rms_norm,
        reserve_prefix as l3_reserve_prefix, rope as l3_rope, AttentionArgs, GatherArgs,
        AdaptiveRmsNormArgs, BiasResidualArgs, BiasResidualLayerNormArgs,
        BiasResidualRmsNormArgs, GatherPatchGeometry, GatherSemantic, GemmArgs, GemmGegluArgs,
        LayerNormArgs, PointwiseActivation, PointwiseArgs, PointwiseSemantic, RmsNormArgs,
        RopeArgs, RopeSemantic, WeightVersion,
    },
    Context, DeviceBuffer as CudaBuffer,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::L3Policies;
use crate::pi05::{
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

struct QkvTensors {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct ResidualNormTensors {
    hidden: Tensor,
    normalized: Tensor,
}

fn rms_bf16(ctx: &Context, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    l3_rms_norm(ctx, RmsNormArgs::new(input, weight, &mut output, eps))?;
    Ok(output)
}

fn layer_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    l3_layer_norm(
        ctx,
        LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

fn adaptive_rms_bf16(
    ctx: &Context,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 adaptive RMS input has no columns".into()))?;
    let style = bf16_vector_prefix(style, 2 * cols)?;
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    l3_adaptive_rms_norm(
        ctx,
        AdaptiveRmsNormArgs::new(input, &style, &mut output, eps),
    )?;
    Ok(output)
}

fn bias_activation_bf16(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    l3_pointwise(ctx, args)?;
    Ok(output)
}

fn euler_update_bf16(
    ctx: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(state.shape().clone(), DType::BF16)?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, state, &mut output);
    args.secondary = Some(velocity);
    args.dt = dt;
    l3_pointwise(ctx, args)?;
    Ok(output)
}

fn add_position_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut args = GatherArgs::new(GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(position);
    args.tokens_per_view = tokens_per_view;
    l3_gather(ctx, args)?;
    Ok(output)
}

fn gemm_bf16(
    ctx: &Context,
    policies: &L3Policies,
    activation: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
    {
        return Err(Error::Other(format!(
            "PI0.5 BF16 GEMM shape/dtype mismatch: {} {a:?} @ {} {b:?}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    crate::pi05::model::calibration::observe_bf16_activation(activation, weight)?;
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    let mut args = GemmArgs::new(activation, weight, &mut output);
    args.policy = policies.gemm.clone();
    l3_gemm(ctx, args)?;
    Ok(output)
}

fn gemm_geglu_bf16(
    ctx: &Context,
    policies: &L3Policies,
    activation: &Tensor,
    canonical_weight: &Tensor,
) -> Result<Tensor> {
    let a = activation.shape().dims();
    let b = canonical_weight.shape().dims();
    if activation.dtype() != DType::BF16
        || canonical_weight.dtype() != DType::BF16
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || b[1] % 2 != 0
    {
        return Err(Error::Other(format!(
            "PI0.5 BF16 GEMM+GeGLU shape/dtype mismatch: {} {a:?} @ {} {b:?}",
            activation.dtype(),
            canonical_weight.dtype()
        )));
    }
    crate::pi05::model::calibration::observe_bf16_activation(
        activation,
        canonical_weight,
    )?;
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1] / 2]), DType::BF16)?;
    let mut gemm = GemmArgs::new(activation, canonical_weight, &mut output)
        .with_immutable_weight(WeightVersion::new(1));
    gemm.policy = policies.gemm.clone();
    l3_gemm_geglu(ctx, GemmGegluArgs { gemm })?;
    Ok(output)
}

fn mqa_bf16(
    ctx: &Context,
    policies: &L3Policies,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q = query.shape().dims();
    let k = key.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || q.len() != 3
        || value.shape() != key.shape()
        || !matches!(k.len(), 2 | 3)
        || k[k.len() - 1] != q[2]
    {
        return Err(Error::Other("PI0.5 BF16 MQA shape/dtype mismatch".into()));
    }
    let kv_heads = if k.len() == 2 { 1 } else { k[1] };
    let available_tokens = k[0];
    if key_tokens == 0 || key_tokens > available_tokens || q[1] % kv_heads != 0 {
        return Err(Error::Other("PI0.5 BF16 MQA token/head mismatch".into()));
    }
    let kv_elements = key_tokens
        .checked_mul(kv_heads)
        .and_then(|count| count.checked_mul(q[2]))
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA size overflow".into()))?;
    let kv_bytes = kv_elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA byte-size overflow".into()))?;
    let key = CudaBuffer::from_tensor(key)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(Error::Cuda)?;
    let query4 = query.reshape(Shape::new(vec![1, q[0], q[1], q[2]]))?;
    let mut output = ctx.allocate_output(query4.shape().clone(), DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key, &value, &mut output);
    args.policy = policies.attention.clone();
    l3_attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

fn mha_bf16(
    ctx: &Context,
    policies: &L3Policies,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = query.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || shape.len() != 3
        || key.shape() != query.shape()
        || value.shape() != query.shape()
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other("PI0.5 BF16 MHA shape/dtype mismatch".into()));
    }
    let batch = shape[0] / tokens_per_batch;
    let shape4 = Shape::new(vec![batch, tokens_per_batch, shape[1], shape[2]]);
    let query4 = query.reshape(shape4.clone())?;
    let key4 = key.reshape(shape4.clone())?;
    let value4 = value.reshape(shape4.clone())?;
    let mut output = ctx.allocate_output(shape4, DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key4, &value4, &mut output);
    args.policy = policies.attention.clone();
    l3_attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

fn split_qkv_bias_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    if qkv.dtype() != DType::BF16
        || shape.len() != 2
        || shape[1] != 3 * heads * head_dim
    {
        return Err(Error::Other(
            "PI0.5 BF16 vision QKV shape/dtype mismatch".into(),
        ));
    }
    let output_shape = Shape::new(vec![shape[0], heads, head_dim]);
    let mut q = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(output_shape, DType::BF16)?;
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 1.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let kv_shape = Shape::new(vec![shape[0], kv_heads, head_dim]);
    let mut k = ctx.allocate_output(kv_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(kv_shape, DType::BF16)?;
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope_into_cache_bf16(
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
    cache_offset: usize,
) -> Result<Tensor> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 cached rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let mut key_cache = key_cache.clone();
    let mut value_cache = value_cache.clone();
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut key_cache,
            v: &mut value_cache,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: cache_offset,
        },
    )?;
    Ok(q)
}

fn bf16_vector_prefix(tensor: &Tensor, elements: usize) -> Result<Tensor> {
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 vector size overflow".into()))?;
    CudaBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(0, bytes))
        .and_then(|buffer| buffer.as_tensor(Shape::new(vec![elements]), DType::BF16))
        .map_err(Error::Cuda)
}

fn bias_residual_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual(
        ctx,
        BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

fn bias_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual_rms_norm(
        ctx,
        BiasResidualRmsNormArgs::new(
            projection,
            bias,
            residual,
            weight,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_bf16(
    ctx: &Context,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual_layer_norm(
        ctx,
        BiasResidualLayerNormArgs::new(
            projection,
            projection_bias,
            residual,
            norm_weight,
            norm_bias,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

fn adaptive_gate_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let cols = projection
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 AdaRMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    crate::pi05::backend::ops::ada_gate_residual_rms_norm(
        ctx,
        crate::pi05::backend::ops::AdaGateResidualRmsNormArgs::new(
            projection,
            residual,
            &norm_style,
            gate_style,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
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
    language_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        input,
        compute_tail,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn language_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16LanguageLayerOutput> {
    let normalized = rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = gemm_bf16(ctx, policies, &normalized, &weights.qkv.weight)?;
    let qkv = split_qkv_rope_bf16(
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
    let attention = mqa_bf16(ctx, policies, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;
    let fused = bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let activated = gemm_geglu_bf16(ctx, policies, &fused.normalized, &weights.gate_up.weight)?;
    let projected = gemm_bf16(ctx, policies, &activated, &weights.down.weight)?;
    let hidden =
        bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
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
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16ActionLayerOutput> {
    action_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        input,
        attention_normalized,
        attention_modulation,
        mlp_modulation,
        next_norm_modulation,
        prefix_k,
        prefix_v,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn action_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => adaptive_rms_bf16(ctx, input, attention_modulation, rms_eps)?,
    };
    let qkv = gemm_bf16(ctx, policies, &normalized, &weights.qkv.weight)?;
    let q = split_qkv_rope_into_cache_bf16(
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
    let attention = mqa_bf16(
        ctx,
        policies,
        &q,
        prefix_k,
        prefix_v,
        position_offset + input.shape().dims()[0],
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let projected = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;
    let fused = adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        input,
        attention_modulation,
        mlp_modulation,
        rms_eps,
    )?;
    let activated = gemm_geglu_bf16(ctx, policies, &fused.normalized, &weights.gate_up.weight)?;
    let projected = gemm_bf16(ctx, policies, &activated, &weights.down.weight)?;
    let fused = adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        &fused.hidden,
        mlp_modulation,
        next_norm_modulation,
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
    vision_patch_embed_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        weights,
        position_embedding,
        patches,
        patches_per_view,
    )
}

fn vision_patch_embed_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm_bf16(ctx, policies, patches, &weights.weight)?;
    add_position_bf16(
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
    vision_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        weights,
        input,
        patches_per_view,
        heads,
        head_dim,
        layer_norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = layer_bf16(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = gemm_bf16(ctx, policies, &normalized, &weights.qkv.weight)?;
    let qkv = split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = mha_bf16(ctx, policies, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;
    let fused = bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = gemm_bf16(ctx, policies, &fused.normalized, &weights.fc1.weight)?;
    let activation = bias_activation_bf16(
        ctx,
        &activation,
        weights.fc1.bias.as_ref(),
        PointwiseActivation::Gelu,
    )?;
    let projection = gemm_bf16(ctx, policies, &activation, &weights.fc2.weight)?;
    bias_residual_bf16(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

trait QkvViews {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
}

impl QkvViews for QkvTensors {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.k.reshape(vec![tokens, head_dim])
    }

    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.v.reshape(vec![tokens, head_dim])
    }
}

// Precision-specific backbone operations share this file with their layers.
pub(in crate::pi05::model) mod backbone {
    use super::*;
    use crate::pi05::backend::{Context, DeviceBuffer as CudaBuffer};
    use crate::pi05::weights::*;
    use crate::pi05::Pi05Config;
    use apxinf_core::{DType, Error, Result, Tensor};
    use std::sync::Arc;
    pub struct Bf16PrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Bf16StepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub struct Bf16Blocks {
        pub(in crate::pi05::model) backend: Arc<Context>,
        pub(in crate::pi05::model) config: Arc<Pi05Config>,
        pub(in crate::pi05::model) weights: Arc<Bf16Weights>,
        pub(in crate::pi05::model) policies: L3Policies,
    }
    impl Bf16Blocks {
        pub(in crate::pi05) fn new(
            backend: Arc<Context>,
            config: Arc<Pi05Config>,
            weights: Arc<Bf16Weights>,
            policies: L3Policies,
        ) -> Result<Self> {
            config.validate()?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 BF16 device weight depth mismatch".into(),
                ));
            }
            Ok(Self {
                backend,
                config,
                weights,
                policies,
            })
        }

        fn ctx(&self) -> &Context {
            &self.backend
        }

        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            if patches.dtype() != DType::BF16 {
                return Err(Error::DTypeMismatch {
                    expected: DType::BF16,
                    got: patches.dtype(),
                });
            }
            let mut hidden = vision_patch_embed_bf16_with_policies(
                self.ctx(),
                &self.policies,
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
            )?;
            for layer in &self.weights.vision_layers {
                hidden = vision_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    layer,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = layer_bf16(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
            )?;
            let projected = gemm_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.multimodal_projector.weight,
            )?;
            bias_activation_bf16(
                self.ctx(),
                &projected,
                self.weights.multimodal_projector.bias.as_ref(),
                PointwiseActivation::None,
            )
        }

        pub fn embed_prefix(
            &self,
            vision_tokens: &Tensor,
            token_ids: &CudaBuffer,
            token_count: usize,
        ) -> Result<Tensor> {
            if token_count == 0 || token_count > self.config.max_token_len {
                return Err(Error::Other(format!(
                    "π0.5 token count must be in 1..={}, got {token_count}",
                    self.config.max_token_len
                )));
            }
            let width = self.weights.token_embedding.shape().dims()[1];
            let mut language = self
                .ctx()
                .allocate_output(Shape::new(vec![token_count, width]), DType::BF16)?;
            let mut args = GatherArgs::new(
                GatherSemantic::EmbeddingLookup,
                &self.weights.token_embedding,
                &mut language,
            );
            args.ids = Some(token_ids);
            args.vocab_size = self.weights.token_embedding.shape().dims()[0];
            l3_gather(self.ctx(), args)?;
            l3_concat_rows(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Bf16PrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                let output = language_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.language,
                    layer,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(l3_reserve_prefix(self.ctx(), &output.key, cache_rows)?);
                values.push(l3_reserve_prefix(self.ctx(), &output.value, cache_rows)?);
            }
            Ok(Bf16PrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let hidden = gemm_bf16(
                self.ctx(),
                &self.policies,
                time_embedding,
                &self.weights.time_mlp_in.weight,
            )?;
            let hidden = bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                PointwiseActivation::Silu,
            )?;
            let output = gemm_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.time_mlp_out.weight,
            )?;
            bias_activation_bf16(
                self.ctx(),
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                PointwiseActivation::Silu,
            )
        }

        fn modulation(&self, conditioning: &Tensor, weights: &Bf16LinearWeights) -> Result<Tensor> {
            let projected = gemm_bf16(self.ctx(), &self.policies, conditioning, &weights.weight)?;
            let modulation = bias_activation_bf16(
                self.ctx(),
                &projected,
                weights.bias.as_ref(),
                PointwiseActivation::None,
            )?;
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(&self, time_embedding: &Tensor) -> Result<Bf16StepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Bf16StepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        pub(super) fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Bf16StepModulation>> {
            if time_embeddings.len() != self.config.num_flow_steps {
                return Err(Error::Other(format!(
                    "π0.5 expected {} timestep embeddings, got {}",
                    self.config.num_flow_steps,
                    time_embeddings.len()
                )));
            }
            time_embeddings
                .iter()
                .map(|embedding| self.prepare_step_modulation(embedding))
                .collect()
        }

        fn denoise_step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Bf16StepModulation,
            prefix: &Bf16PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 BF16 prefix/modulation depth mismatch".into(),
                ));
            }
            let hidden = gemm_bf16(
                self.ctx(),
                &self.policies,
                state,
                &self.weights.action_in.weight,
            )?;
            let mut hidden = bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.action_in.bias.as_ref(),
                PointwiseActivation::None,
            )?;
            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_norm_modulation = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention[index + 1]
                } else {
                    &modulation.final_norm
                };
                let output = action_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.action_expert,
                    layer,
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention[index],
                    &modulation.mlp[index],
                    next_norm_modulation,
                    &prefix.keys[index],
                    &prefix.values[index],
                    prefix.tokens,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                attention_normalized = Some(output.next_normalized);
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("π0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = gemm_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.action_out.weight,
            )?;
            let velocity = bias_activation_bf16(
                self.ctx(),
                &velocity,
                self.weights.action_out.bias.as_ref(),
                PointwiseActivation::None,
            )?;
            euler_update_bf16(self.ctx(), state, &velocity, dt)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Bf16PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }
    }

    impl super::super::Blocks for Bf16Blocks {
        type Prefix = Bf16PrefixKvCache;
        type StepModulation = Bf16StepModulation;
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor> {
            let _ = native;
            self.encode_vision(patches)
        }
        fn embed_prefix(&self, vision: &Tensor, ids: &CudaBuffer, count: usize) -> Result<Tensor> {
            self.embed_prefix(vision, ids, count)
        }
        fn prefix(&self, input: &Tensor) -> Result<Self::Prefix> {
            self.prefix_forward(input)
        }
        fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>> {
            self.prepare_all_modulation(embeddings)
        }
        fn eager_modulation(
            &self,
            embeddings: &[Tensor],
        ) -> Result<Option<Vec<Self::StepModulation>>> {
            self.prepare_all_modulation(embeddings).map(Some)
        }
        fn step(
            &self,
            state: &Tensor,
            embedding: &Tensor,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step(state, embedding, prefix, dt)
        }
        fn step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Self::StepModulation,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step_with_modulation(state, modulation, prefix, dt)
        }
    }
}

impl crate::pi05::model::PrepareBlocks for backbone::Bf16Blocks {
    fn backend(&self) -> &std::sync::Arc<crate::pi05::backend::Context> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::pi05::model::WorkspaceRequirements> {
        Ok(crate::pi05::model::WorkspaceRequirements {
            bytes: self.config.cuda_graph_workspace_bytes_bf16(tokens)?,
        })
    }
    fn raw_patch_dtype(&self) -> apxinf_core::DType {
        apxinf_core::DType::BF16
    }
    fn preprocess(
        &self,
        images: &crate::pi05::backend::DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        let mut patches = patches.clone();
        l3_gather(
            &self.backend,
            GatherArgs::rgb_to_patches(
                images,
                &mut patches,
                GatherPatchGeometry {
                    views: self.config.num_views,
                    image_size: self.config.image_size,
                    patch_size: self.config.patch_size,
                    nhwc: matches!(layout, crate::pi05::Pi05ImageLayout::Nhwc),
                },
            ),
        )
    }
}
