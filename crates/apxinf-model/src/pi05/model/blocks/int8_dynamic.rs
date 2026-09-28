//! Dynamic-activation INT8 π0.5 transformer-layer computation.

use crate::pi05::backend::{ops, Context, DeviceBuffer};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::pi05::{
    GemmaVariantConfig, Int8DynamicDeviceActionLayer, Int8DynamicDeviceLanguageLayer,
    Int8DynamicDeviceVisionBlock, Int8DynamicLinearWeights,
};

pub struct Int8DynamicLanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Int8DynamicActionLayerOutput {
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

pub(in crate::pi05) type Int8DynamicL3Policies = super::L3Policies;

fn l3_rms_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    ops::rms_norm(
        ctx,
        ops::RmsNormArgs::new(input, weight, &mut output, eps),
    )?;
    Ok(output)
}

fn l3_layer_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    ops::layer_norm(
        ctx,
        ops::LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

fn bf16_vector_prefix(tensor: &Tensor, elements: usize) -> Result<Tensor> {
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 INT8 BF16 vector size overflow".into()))?;
    DeviceBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(0, bytes))
        .and_then(|buffer| buffer.as_tensor(Shape::new(vec![elements]), DType::BF16))
        .map_err(Error::Cuda)
}

fn l3_adaptive_rms_bf16(
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
        .ok_or_else(|| Error::Other("PI0.5 INT8 adaptive RMS input has no columns".into()))?;
    let style = bf16_vector_prefix(style, 2 * cols)?;
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    ops::adaptive_rms_norm(
        ctx,
        ops::AdaptiveRmsNormArgs::new(input, &style, &mut output, eps),
    )?;
    Ok(output)
}

fn l3_bias_activation_bf16(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

fn l3_geglu_bf16(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    let shape = input.shape().dims();
    if input.dtype() != DType::BF16 || shape.len() != 2 || shape[1] % 2 != 0 {
        return Err(Error::Other(
            "PI0.5 INT8 GeGLU expects BF16 [rows,2*cols]".into(),
        ));
    }
    let mut output =
        ctx.allocate_output(Shape::new(vec![shape[0], shape[1] / 2]), DType::BF16)?;
    let args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, input, &mut output);
    ops::pointwise(ctx, args)?;
    Ok(output)
}

fn l3_euler_update_bf16(
    ctx: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(state.shape().clone(), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut output);
    args.secondary = Some(velocity);
    args.dt = dt;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

fn l3_mqa_bf16(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
    policy: &ops::AttentionPolicy,
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
        return Err(Error::Other(
            "PI0.5 INT8 BF16 MQA shape/dtype mismatch".into(),
        ));
    }
    let kv_heads = if k.len() == 2 { 1 } else { k[1] };
    if key_tokens == 0 || key_tokens > k[0] || q[1] % kv_heads != 0 {
        return Err(Error::Other(
            "PI0.5 INT8 BF16 MQA token/head mismatch".into(),
        ));
    }
    let kv_bytes = key_tokens
        .checked_mul(kv_heads)
        .and_then(|count| count.checked_mul(q[2]))
        .and_then(|count| count.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("PI0.5 INT8 BF16 MQA size overflow".into()))?;
    let key = DeviceBuffer::from_tensor(key)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(Error::Cuda)?;
    let value = DeviceBuffer::from_tensor(value)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(Error::Cuda)?;
    let query_shape = Shape::new(vec![1, q[0], q[1], q[2]]);
    let query = query.reshape(query_shape.clone())?;
    let mut output = ctx.allocate_output(query_shape, DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&query, &key, &value, &mut output);
    args.policy = policy.clone();
    ops::attention(ctx, args)?;
    output.reshape(Shape::new(vec![q[0], q[1], q[2]]))
}

fn l3_mha_bf16(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    tokens_per_batch: usize,
    policy: &ops::AttentionPolicy,
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
        return Err(Error::Other(
            "PI0.5 INT8 BF16 MHA shape/dtype mismatch".into(),
        ));
    }
    let batch = shape[0] / tokens_per_batch;
    let shape4 = Shape::new(vec![batch, tokens_per_batch, shape[1], shape[2]]);
    let query4 = query.reshape(shape4.clone())?;
    let key4 = key.reshape(shape4.clone())?;
    let value4 = value.reshape(shape4.clone())?;
    let mut output = ctx.allocate_output(shape4, DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&query4, &key4, &value4, &mut output);
    args.policy = policy.clone();
    ops::attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

fn l3_split_qkv_bias_bf16(
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
            "PI0.5 INT8 vision QKV shape/dtype mismatch".into(),
        ));
    }
    let output_shape = Shape::new(vec![shape[0], heads, head_dim]);
    let mut q = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(output_shape, DType::BF16)?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvBias,
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
fn l3_split_qkv_rope_bf16(
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
        .ok_or_else(|| Error::Other("PI0.5 INT8 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 INT8 rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let kv_shape = Shape::new(vec![shape[0], kv_heads, head_dim]);
    let mut k = ctx.allocate_output(kv_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(kv_shape, DType::BF16)?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
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
fn l3_split_qkv_rope_into_cache_bf16(
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
        .ok_or_else(|| Error::Other("PI0.5 INT8 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 INT8 cached rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let mut key_cache = key_cache.clone();
    let mut value_cache = value_cache.clone();
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
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

fn l3_bias_residual_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ops::bias_residual(
        ctx,
        ops::BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

fn l3_bias_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ops::bias_residual_rms_norm(
        ctx,
        ops::BiasResidualRmsNormArgs::new(
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
fn l3_bias_residual_layer_bf16(
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
    ops::bias_residual_layer_norm(
        ctx,
        ops::BiasResidualLayerNormArgs::new(
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

fn l3_adaptive_gate_residual_rms_bf16(
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
        .ok_or_else(|| Error::Other("PI0.5 INT8 AdaRMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ops::ada_gate_residual_rms_norm(
        ctx,
        ops::AdaGateResidualRmsNormArgs::new(
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

fn l3_bias_position_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut args =
        ops::GatherArgs::new(ops::GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(position);
    args.tokens_per_view = tokens_per_view;
    ops::gather(ctx, args)?;
    Ok(output)
}

fn l3_embedding_lookup_bf16(
    ctx: &Context,
    table: &Tensor,
    ids: &DeviceBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let shape = table.shape().dims();
    if table.dtype() != DType::BF16 || shape.len() != 2 || tokens == 0 {
        return Err(Error::Other(
            "PI0.5 INT8 embedding expects BF16 [vocab,width]".into(),
        ));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![tokens, shape[1]]), DType::BF16)?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(ids);
    args.vocab_size = shape[0];
    ops::gather(ctx, args)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn language_layer_int8_dynamic_with_policies(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DynamicDeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
    policies: &Int8DynamicL3Policies,
) -> Result<Int8DynamicLanguageLayerOutput> {
    let normalized = l3_rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = weights
        .qkv
        .gemm_with_policies(ctx, &normalized, &policies.gemm)?;
    let qkv = l3_split_qkv_rope_bf16(
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
        return Ok(Int8DynamicLanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.key_2d(tokens, config.head_dim)?,
            value: qkv.value_2d(tokens, config.head_dim)?,
        });
    }
    let attention = l3_mqa_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, tokens, &policies.attention)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = weights
        .output
        .gemm_with_policies(ctx, &attention, &policies.gemm)?;
    let fused = l3_bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = weights.gate_up.gemm_with_policies(
        ctx,
        &fused.normalized,
        &policies.gemm,
    )?;
    let activated = l3_geglu_bf16(ctx, &gate_up)?;
    let projected = weights
        .down
        .gemm_with_policies(ctx, &activated, &policies.gemm)?;
    let hidden = l3_bias_residual_bf16(
        ctx,
        &projected,
        weights.down.bias.as_ref(),
        &fused.hidden,
    )?;
    Ok(Int8DynamicLanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_int8_dynamic(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DynamicDeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Int8DynamicLanguageLayerOutput> {
    language_layer_int8_dynamic_with_policies(
        ctx,
        config,
        weights,
        input,
        compute_tail,
        position_offset,
        rms_eps,
        rope_theta,
        &Int8DynamicL3Policies::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn action_layer_int8_dynamic_with_policies(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DynamicDeviceActionLayer,
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
    policies: &Int8DynamicL3Policies,
) -> Result<Int8DynamicActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => l3_adaptive_rms_bf16(
            ctx,
            input,
            attention_modulation,
            rms_eps,
        )?,
    };
    let qkv = weights
        .qkv
        .gemm_with_policies(ctx, &normalized, &policies.gemm)?;
    let q = l3_split_qkv_rope_into_cache_bf16(
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
    let attention = l3_mqa_bf16(
        ctx,
        &q,
        prefix_k,
        prefix_v,
        position_offset + input.shape().dims()[0],
        &policies.attention,
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let projected = weights
        .output
        .gemm_with_policies(ctx, &attention, &policies.gemm)?;
    let fused = l3_adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        input,
        attention_modulation,
        mlp_modulation,
        rms_eps,
    )?;
    let gate_up = weights.gate_up.gemm_with_policies(
        ctx,
        &fused.normalized,
        &policies.gemm,
    )?;
    let activated = l3_geglu_bf16(ctx, &gate_up)?;
    let projected = weights
        .down
        .gemm_with_policies(ctx, &activated, &policies.gemm)?;
    let fused = l3_adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        &fused.hidden,
        mlp_modulation,
        next_norm_modulation,
        rms_eps,
    )?;
    Ok(Int8DynamicActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_int8_dynamic(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Int8DynamicDeviceActionLayer,
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
) -> Result<Int8DynamicActionLayerOutput> {
    action_layer_int8_dynamic_with_policies(
        ctx,
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
        &Int8DynamicL3Policies::default(),
    )
}

fn vision_patch_embed_int8_dynamic_with_policies(
    ctx: &Context,
    weights: &Int8DynamicLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    policies: &Int8DynamicL3Policies,
) -> Result<Tensor> {
    let projection = weights.gemm_with_policies(ctx, patches, &policies.gemm)?;
    l3_bias_position_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

pub fn vision_patch_embed_int8_dynamic(
    ctx: &Context,
    weights: &Int8DynamicLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    vision_patch_embed_int8_dynamic_with_policies(
        ctx,
        weights,
        position_embedding,
        patches,
        patches_per_view,
        &Int8DynamicL3Policies::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_layer_int8_dynamic_with_policies(
    ctx: &Context,
    weights: &Int8DynamicDeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
    policies: &Int8DynamicL3Policies,
) -> Result<Tensor> {
    let normalized = l3_layer_bf16(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = weights
        .qkv
        .gemm_with_policies(ctx, &normalized, &policies.gemm)?;
    let qkv = l3_split_qkv_bias_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        heads,
        head_dim,
    )?;
    let attention = l3_mha_bf16(
        ctx,
        &qkv.q,
        &qkv.k,
        &qkv.v,
        patches_per_view,
        &policies.attention,
    )?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = weights
        .output
        .gemm_with_policies(ctx, &attention, &policies.gemm)?;
    let fused = l3_bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = weights.fc1.gemm_with_policies(
        ctx,
        &fused.normalized,
        &policies.gemm,
    )?;
    let activation = l3_bias_activation_bf16(
        ctx,
        &activation,
        weights.fc1.bias.as_ref(),
        ops::PointwiseActivation::Gelu,
    )?;
    let projection = weights
        .fc2
        .gemm_with_policies(ctx, &activation, &policies.gemm)?;
    l3_bias_residual_bf16(
        ctx,
        &projection,
        weights.fc2.bias.as_ref(),
        &fused.hidden,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_int8_dynamic(
    ctx: &Context,
    weights: &Int8DynamicDeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    vision_layer_int8_dynamic_with_policies(
        ctx,
        weights,
        input,
        patches_per_view,
        heads,
        head_dim,
        layer_norm_eps,
        &Int8DynamicL3Policies::default(),
    )
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
    pub struct Int8DynamicPrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Int8DynamicStepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub struct Int8DynamicBlocks {
        pub(in crate::pi05::model) backend: Arc<Context>,
        pub(in crate::pi05::model) config: Arc<Pi05Config>,
        pub(in crate::pi05::model) weights: Arc<Int8DynamicWeights>,
        pub(in crate::pi05::model) policies: Int8DynamicL3Policies,
    }
    impl Int8DynamicBlocks {
        pub(in crate::pi05) fn new(
            backend: Arc<Context>,
            config: Arc<Pi05Config>,
            weights: Arc<Int8DynamicWeights>,
            policies: Int8DynamicL3Policies,
        ) -> Result<Self> {
            config.validate()?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 INT8 device weight depth mismatch".into(),
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
            let mut hidden = vision_patch_embed_int8_dynamic_with_policies(
                self.ctx(),
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
                &self.policies,
            )?;
            for layer in &self.weights.vision_layers {
                hidden = vision_layer_int8_dynamic_with_policies(
                    self.ctx(),
                    layer,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                    &self.policies,
                )?;
            }
            let hidden = l3_layer_bf16(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
            )?;
            let projected = self
                .weights
                .multimodal_projector
                .gemm_with_policies(
                    self.ctx(),
                    &hidden,
                    &self.policies.gemm,
                )?;
            l3_bias_activation_bf16(
                self.ctx(),
                &projected,
                self.weights.multimodal_projector.bias.as_ref(),
                ops::PointwiseActivation::None,
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
            let language = l3_embedding_lookup_bf16(
                self.ctx(),
                &self.weights.token_embedding,
                token_ids,
                token_count,
            )?;
            ops::concat_rows(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Int8DynamicPrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                let output = language_layer_int8_dynamic_with_policies(
                    self.ctx(),
                    self.config.language,
                    layer,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                    &self.policies,
                )?;
                hidden = output.hidden;
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(ops::reserve_prefix(self.ctx(), &output.key, cache_rows)?);
                values.push(ops::reserve_prefix(self.ctx(), &output.value, cache_rows)?);
            }
            Ok(Int8DynamicPrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let hidden = self.weights.time_mlp_in.gemm_with_policies(
                self.ctx(),
                time_embedding,
                &self.policies.gemm,
            )?;
            let hidden = l3_bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )?;
            let output = self.weights.time_mlp_out.gemm_with_policies(
                self.ctx(),
                &hidden,
                &self.policies.gemm,
            )?;
            l3_bias_activation_bf16(
                self.ctx(),
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )
        }

        fn modulation(
            &self,
            conditioning: &Tensor,
            weights: &Int8DynamicLinearWeights,
        ) -> Result<Tensor> {
            let projected = weights.gemm_with_policies(
                self.ctx(),
                conditioning,
                &self.policies.gemm,
            )?;
            let modulation = l3_bias_activation_bf16(
                self.ctx(),
                &projected,
                weights.bias.as_ref(),
                ops::PointwiseActivation::None,
            )?;
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(
            &self,
            time_embedding: &Tensor,
        ) -> Result<Int8DynamicStepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Int8DynamicStepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Int8DynamicStepModulation>> {
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
            modulation: &Int8DynamicStepModulation,
            prefix: &Int8DynamicPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 INT8 prefix/modulation depth mismatch".into(),
                ));
            }
            let hidden = self.weights.action_in.gemm_with_policies(
                self.ctx(),
                state,
                &self.policies.gemm,
            )?;
            let mut hidden = l3_bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.action_in.bias.as_ref(),
                ops::PointwiseActivation::None,
            )?;
            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_norm_modulation = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention[index + 1]
                } else {
                    &modulation.final_norm
                };
                let output = action_layer_int8_dynamic_with_policies(
                    self.ctx(),
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
                    &self.policies,
                )?;
                hidden = output.hidden;
                attention_normalized = Some(output.next_normalized);
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("π0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = self.weights.action_out.gemm_with_policies(
                self.ctx(),
                &hidden,
                &self.policies.gemm,
            )?;
            let velocity = l3_bias_activation_bf16(
                self.ctx(),
                &velocity,
                self.weights.action_out.bias.as_ref(),
                ops::PointwiseActivation::None,
            )?;
            l3_euler_update_bf16(self.ctx(), state, &velocity, dt)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Int8DynamicPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }
    }

    impl super::super::Blocks for Int8DynamicBlocks {
        type Prefix = Int8DynamicPrefixKvCache;
        type StepModulation = Int8DynamicStepModulation;
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

impl crate::pi05::model::PrepareBlocks for backbone::Int8DynamicBlocks {
    fn backend(&self) -> &std::sync::Arc<crate::pi05::backend::Context> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::pi05::model::WorkspaceRequirements> {
        Ok(crate::pi05::model::WorkspaceRequirements {
            bytes: self
                .config
                .cuda_graph_workspace_bytes_int8_dynamic(tokens)?,
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
        let args = ops::GatherArgs::rgb_to_patches(
            images,
            &mut patches,
            ops::GatherPatchGeometry {
                views: self.config.num_views,
                image_size: self.config.image_size,
                patch_size: self.config.patch_size,
                nhwc: matches!(layout, crate::pi05::Pi05ImageLayout::Nhwc),
            },
        );
        ops::gather(&self.backend, args)
    }
}
