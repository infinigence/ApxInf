//! π0.5 FP8 CUDA transformer-layer computation.

use crate::pi05::backend::{ops, Context, DeviceBuffer};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::pi05::{
    Fp8StaticDeviceActionLayer, Fp8StaticDeviceLanguageLayer, Fp8StaticDeviceVisionBlock,
    GemmaVariantConfig,
};

use crate::pi05::{Fp8StaticTransformerLayerScales, Fp8StaticVisionLayerScales};

pub struct Fp8StaticLanguageLayerOutput {
    pub hidden: Tensor,
    /// Prefix K/V are retained per layer for the paired action expert.
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Fp8StaticActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

fn action_attention_args<'a>(
    query: &'a Tensor,
    key: &'a Tensor,
    value: &'a Tensor,
    output: &'a mut Tensor,
    valid_key_tokens: usize,
    policy: &ops::AttentionPolicy,
) -> ops::KvCacheAttentionArgs<'a> {
    let mut args = ops::KvCacheAttentionArgs::new(query, key, value, output).non_causal();
    args.valid_key_tokens = valid_key_tokens;
    args.policy = policy.clone();
    args
}

pub(in crate::pi05) type Fp8L3Policies = super::L3Policies;

struct QkvTensors {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

fn output(ctx: &Context, shape: impl Into<Vec<usize>>, dtype: DType) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(shape.into()), dtype)
}

fn fixed_quantize(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let mut out = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::QuantizationArgs::new(
        ops::QuantizationSemantic::FixedScaleE4m3,
        input,
        &mut out,
    );
    args.scale = scale;
    ops::quantization(ctx, args)?;
    Ok(out)
}

fn fp8_gemm(
    ctx: &Context,
    policies: &Fp8L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::pi05::Fp8StaticLinearWeights,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other("π0.5 FP8 GEMM shape mismatch".into()));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::F16)?;
    let mut args = ops::GemmArgs::new(input, &weights.weight, &mut out)
        .with_immutable_weight(ops::WeightVersion::new(1));
    args.quantization = ops::GemmQuantization::Fp8UnitScale;
    args.alpha = input_scale * weights.weight_scale;
    args.policy = policies.gemm.clone();
    ops::gemm(ctx, args)?;
    Ok(out)
}

fn fp8_gemm_bias(
    ctx: &Context,
    policies: &Fp8L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::pi05::Fp8StaticLinearWeights,
) -> Result<Tensor> {
    let bias = weights
        .bias
        .as_ref()
        .ok_or_else(|| Error::Other("π0.5 FP8 fused GEMM requires a bias".into()))?;
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::F16)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out)
        .with_immutable_weight(ops::WeightVersion::new(1));
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.policy = policies.gemm.clone();
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?;
    Ok(out)
}

fn fp8_gemm_bias_gelu_quant(
    ctx: &Context,
    policies: &Fp8L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    output_scale: f32,
) -> Result<Tensor> {
    let bias = weights
        .bias
        .as_ref()
        .ok_or_else(|| Error::Other("π0.5 FP8 fused GEMM+GELU requires a bias".into()))?;
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other("π0.5 FP8 GEMM+GELU shape mismatch".into()));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::F8E4M3)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out)
        .with_immutable_weight(ops::WeightVersion::new(1));
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.output_scale = output_scale;
    gemm.policy = policies.gemm.clone();
    ops::gemm_bias_gelu(ctx, ops::GemmBiasGeluArgs { gemm, bias })?;
    Ok(out)
}

fn fp8_gemm_bias_residual(
    ctx: &Context,
    policies: &Fp8L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    residual: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other(
            "π0.5 FP8 GEMM+bias+residual shape mismatch".into(),
        ));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::F16)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out)
        .with_immutable_weight(ops::WeightVersion::new(1));
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.policy = policies.gemm.clone();
    ops::gemm_bias_residual(
        ctx,
        ops::GemmBiasResidualArgs {
            gemm,
            bias: weights.bias.as_ref(),
            residual,
        },
    )?;
    Ok(out)
}

fn fp8_gemm_geglu(
    ctx: &Context,
    policies: &Fp8L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    output_scale: f32,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if weight_dims[1] % 2 != 0 {
        return Err(Error::Other("π0.5 FP8 GeGLU width must be even".into()));
    }
    let fused_shape = matches!(dims[0], 522 | 533)
        && dims[1] == 2048
        && weight_dims == [2048, 32768];
    if fused_shape {
        let mut out = output(ctx, vec![dims[0], weight_dims[1] / 2], DType::F8E4M3)?;
        let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out)
            .with_immutable_weight(ops::WeightVersion::new(1));
        gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
        gemm.alpha = input_scale * weights.weight_scale;
        gemm.output_scale = output_scale;
        gemm.policy = policies.gemm.clone();
        ops::gemm_geglu(ctx, ops::GemmGegluArgs { gemm })?;
        return Ok(out);
    }

    let gate_up = fp8_gemm(ctx, policies, input, input_scale, weights)?;
    let mut out = output(ctx, vec![dims[0], weight_dims[1] / 2], DType::F8E4M3)?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, &gate_up, &mut out);
    args.output_scale = output_scale;
    ops::pointwise(ctx, args)?;
    Ok(out)
}

fn bias_activation(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut out = output(ctx, input.shape().dims().to_vec(), input.dtype())?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut out);
    args.bias = bias;
    args.activation = activation;
    ops::pointwise(ctx, args)?;
    Ok(out)
}

fn rms_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::RmsNormArgs::new(input, weight, &mut normalized, eps);
    args.output_scale = scale;
    ops::rms_norm(ctx, args)?;
    Ok(normalized)
}

fn layer_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::LayerNormArgs::new(input, weight, bias, &mut normalized, eps);
    args.output_scale = scale;
    ops::layer_norm(ctx, args)?;
    Ok(normalized)
}

fn f16_vector_prefix(tensor: &Tensor, elements: usize) -> Result<Tensor> {
    if tensor.dtype() != DType::F16 {
        return Err(Error::Other(format!(
            "PI0.5 FP8 modulation must use F16 storage, got {}",
            tensor.dtype()
        )));
    }
    let bytes = elements
        .checked_mul(DType::F16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 F16 vector size overflow".into()))?;
    DeviceBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(0, bytes))
        .and_then(|buffer| buffer.as_tensor(Shape::new(vec![elements]), DType::F16))
        .map_err(Error::Cuda)
}

fn adaptive_rms_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 adaptive RMS input has no columns".into()))?;
    let norm_style = f16_vector_prefix(norm_style, 2 * cols)?;
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::AdaptiveRmsNormArgs::new(input, &norm_style, &mut normalized, eps);
    args.output_scale = scale;
    ops::adaptive_rms_norm(ctx, args)?;
    Ok(normalized)
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_rms_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::BiasResidualRmsNormArgs::new(
        input, bias, residual, weight, &mut hidden, &mut normalized, eps,
    );
    args.output_scale = scale;
    ops::bias_residual_rms_norm(ctx, args)?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::BiasResidualLayerNormArgs::new(
        input, bias, residual, weight, norm_bias, &mut hidden, &mut normalized, eps,
    );
    args.output_scale = scale;
    ops::bias_residual_layer_norm(ctx, args)?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn ada_gate_residual_rms_normalized_fp8(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    input: &Tensor,
    residual: &Tensor,
    norm_style: &Tensor,
    gate_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 AdaRMS input has no columns".into()))?;
    let norm_style = f16_vector_prefix(norm_style, 2 * cols)?;
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::AdaGateResidualRmsNormArgs::new(
        input, residual, &norm_style, gate_style, &mut hidden, &mut normalized, eps,
    );
    args.output_scale = scale;
    ops::ada_gate_residual_rms_norm(ctx, args)?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn split_qkv(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let tokens = qkv.shape().dims()[0];
    let mut q = output(ctx, vec![tokens, q_heads, head_dim], qkv.dtype())?;
    let mut k = output(ctx, vec![tokens, kv_heads, head_dim], qkv.dtype())?;
    let mut v = output(ctx, vec![tokens, kv_heads, head_dim], qkv.dtype())?;
    let args = ops::RopeArgs {
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
    };
    ops::rope(ctx, args)?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_to_cache(
    ctx: &Context,
    _policies: &Fp8L3Policies,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    key_cache: &Tensor,
    value_cache: &Tensor,
) -> Result<Tensor> {
    let tokens = qkv.shape().dims()[0];
    let mut q = output(ctx, vec![tokens, q_heads, head_dim], qkv.dtype())?;
    let mut key_cache = key_cache.clone();
    let mut value_cache = value_cache.clone();
    let args = ops::RopeArgs {
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
        kv_output_offset: position_offset,
    };
    ops::rope(ctx, args)?;
    Ok(q)
}

fn dense_attention(
    ctx: &Context,
    policies: &Fp8L3Policies,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output_scale: Option<f32>,
) -> Result<Tensor> {
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let q4 = q.reshape(vec![1, q_dims[0], q_dims[1], q_dims[2]])?;
    let k4 = k.reshape(vec![1, k_dims[0], k_dims[1], k_dims[2]])?;
    let v4 = v.reshape(vec![1, k_dims[0], k_dims[1], k_dims[2]])?;
    let dtype = output_scale.map_or(q.dtype(), |_| DType::F8E4M3);
    let mut out = output(ctx, q4.shape().dims().to_vec(), dtype)?;
    let mut args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut out);
    if let Some(scale) = output_scale {
        args.output_scale = scale;
    }
    args.policy = policies.attention.clone();
    ops::attention(ctx, args)?;
    Ok(out.reshape(vec![q_dims[0], q_dims[1] * q_dims[2]])?)
}

fn uses_direct_e4m3_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
    q.shape().dims() == [522, 8, 256]
        && k.shape().dims() == [522, 1, 256]
        && v.shape().dims() == [522, 1, 256]
}

pub fn language_layer_fp8_static(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceLanguageLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticLanguageLayerOutput> {
    language_layer_fp8_static_with_policies(
        ctx,
        &Fp8L3Policies::default(),
        config,
        weights,
        scales,
        input,
        compute_tail,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn language_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &Fp8L3Policies,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceLanguageLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticLanguageLayerOutput> {
    let normalized = rms_normalized_fp8(
        ctx,
        policies,
        input,
        &weights.input_norm_scale,
        rms_eps,
        scales.attention_norm,
    )?;
    let qkv = fp8_gemm(ctx, policies, &normalized, scales.attention_norm, &weights.qkv)?;
    let qkv = split_qkv(
        ctx,
        policies,
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
        return Ok(Fp8StaticLanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.k.reshape(vec![tokens, config.head_dim])?,
            value: qkv.v.reshape(vec![tokens, config.head_dim])?,
        });
    }
    // Preserve the validated legacy path: only the production T10 shape has
    // a fused F16-to-E4M3 FA2 epilogue. Other token lengths, including T21,
    // use ordinary F16 attention followed by the direct quantization kernel.
    let attention = if uses_direct_e4m3_attention(&qkv.q, &qkv.k, &qkv.v) {
        dense_attention(
            ctx,
            policies,
            &qkv.q,
            &qkv.k,
            &qkv.v,
            Some(scales.attention_output),
        )?
    } else {
        let attention = dense_attention(ctx, policies, &qkv.q, &qkv.k, &qkv.v, None)?;
        fixed_quantize(ctx, policies, &attention, scales.attention_output)?
    };
    let projected = fp8_gemm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
    )?;
    let (hidden, normalized) = bias_residual_rms_normalized_fp8(
        ctx,
        policies,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
        scales.mlp_norm,
    )?;
    let activated = fp8_gemm_geglu(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.gate_up,
        scales.mlp_activation,
    )?;
    let hidden = fp8_gemm_bias_residual(
        ctx,
        policies,
        &activated,
        scales.mlp_activation,
        &weights.down,
        &hidden,
    )?;
    Ok(Fp8StaticLanguageLayerOutput {
        hidden,
        key: qkv.k.reshape(vec![tokens, config.head_dim])?,
        value: qkv.v.reshape(vec![tokens, config.head_dim])?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_fp8_static(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceActionLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    next_norm_scale: f32,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticActionLayerOutput> {
    action_layer_fp8_static_with_policies(
        ctx,
        &Fp8L3Policies::default(),
        config,
        weights,
        scales,
        input,
        attention_normalized,
        attention_modulation,
        mlp_modulation,
        next_norm_modulation,
        next_norm_scale,
        prefix_k,
        prefix_v,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn action_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &Fp8L3Policies,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceActionLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    next_norm_scale: f32,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(normalized) => normalized.clone(),
        None => adaptive_rms_normalized_fp8(
            ctx,
            policies,
            input,
            attention_modulation,
            rms_eps,
            scales.attention_norm,
        )?,
    };
    let qkv = fp8_gemm(ctx, policies, &normalized, scales.attention_norm, &weights.qkv)?;
    let q = split_qkv_to_cache(
        ctx,
        policies,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
    )?;
    let key_tokens = position_offset + input.shape().dims()[0];
    let q_dims = q.shape().dims();
    let cache_rows = prefix_k.shape().dims()[0];
    let q4 = q.reshape(vec![1, q_dims[0], q_dims[1], q_dims[2]])?;
    let k4 = prefix_k.reshape(vec![1, cache_rows, config.num_kv_heads, config.head_dim])?;
    let v4 = prefix_v.reshape(vec![1, cache_rows, config.num_kv_heads, config.head_dim])?;
    let mut attention = output(ctx, q4.shape().dims().to_vec(), DType::F16)?;
    let attention_args = action_attention_args(
        &q4,
        &k4,
        &v4,
        &mut attention,
        key_tokens,
        &policies.attention,
    );
    ops::kv_cache_attention(ctx, attention_args)?;
    let attention = attention.reshape(vec![q_dims[0], config.num_heads * config.head_dim])?;
    let attention = fixed_quantize(ctx, policies, &attention, scales.attention_output)?;
    let projected = fp8_gemm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
    )?;
    let (hidden, normalized) = ada_gate_residual_rms_normalized_fp8(
        ctx,
        policies,
        &projected,
        input,
        mlp_modulation,
        attention_modulation,
        rms_eps,
        scales.mlp_norm,
    )?;
    let activated = fp8_gemm_geglu(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.gate_up,
        scales.mlp_activation,
    )?;
    let projected = fp8_gemm(
        ctx,
        policies,
        &activated,
        scales.mlp_activation,
        &weights.down,
    )?;
    let (hidden, normalized) = ada_gate_residual_rms_normalized_fp8(
        ctx,
        policies,
        &projected,
        &hidden,
        next_norm_modulation,
        mlp_modulation,
        rms_eps,
        next_norm_scale,
    )?;
    Ok(Fp8StaticActionLayerOutput {
        hidden,
        next_normalized: normalized,
    })
}

pub fn vision_patch_embed_fp8_static(
    ctx: &Context,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let policies = Fp8L3Policies::default();
    let patches = fixed_quantize(ctx, &policies, patches, input_scale)?;
    vision_patch_embed_fp8_static_native_with_policies(
        ctx,
        &policies,
        weights,
        position_embedding,
        &patches,
        patches_per_view,
        input_scale,
    )
}

/// Patch projection when preprocessing has already produced calibrated E4M3
/// patch tokens. This is the entry used by the fused raw-image graph.
pub fn vision_patch_embed_fp8_static_native(
    ctx: &Context,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    vision_patch_embed_fp8_static_native_with_policies(
        ctx,
        &Fp8L3Policies::default(),
        weights,
        position_embedding,
        patches,
        patches_per_view,
        input_scale,
    )
}

fn vision_patch_embed_fp8_static_native_with_policies(
    ctx: &Context,
    policies: &Fp8L3Policies,
    weights: &crate::pi05::Fp8StaticLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let projection = fp8_gemm(ctx, policies, patches, input_scale, weights)?;
    let mut out = output(ctx, projection.shape().dims().to_vec(), projection.dtype())?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::BiasPosition, &projection, &mut out);
    args.bias = weights.bias.as_ref();
    args.position = Some(position_embedding);
    args.tokens_per_view = patches_per_view;
    ops::gather(ctx, args)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_fp8_static(
    ctx: &Context,
    weights: &Fp8StaticDeviceVisionBlock,
    scales: Fp8StaticVisionLayerScales,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    vision_layer_fp8_static_with_policies(
        ctx,
        &Fp8L3Policies::default(),
        weights,
        scales,
        input,
        patches_per_view,
        heads,
        head_dim,
        layer_norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &Fp8L3Policies,
    weights: &Fp8StaticDeviceVisionBlock,
    scales: Fp8StaticVisionLayerScales,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = layer_normalized_fp8(
        ctx,
        policies,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
        scales.attention_norm,
    )?;
    let use_packed_qkv = ctx.caps().arch_family == apxinf_cuda_new::CudaArchFamily::Sm100
        && patches_per_view == 256
        && heads == 16
        && head_dim == 72
        && weights.qkv.bias.is_some();
    let qkv = if use_packed_qkv {
        fp8_gemm_bias(ctx, policies, &normalized, scales.attention_norm, &weights.qkv)?
    } else {
        fp8_gemm(ctx, policies, &normalized, scales.attention_norm, &weights.qkv)?
    };
    let tokens = input.shape().dims()[0];
    let views = tokens / patches_per_view;
    let attention = if use_packed_qkv {
        let qkv = qkv.reshape(vec![views, patches_per_view, 3, heads, head_dim])?;
        let mut attention = output(
            ctx,
            vec![views, patches_per_view, heads, head_dim],
            DType::F16,
        )?;
        let mut args = ops::PackedQkvAttentionArgs::new(&qkv, &mut attention);
        args.policy = policies.attention.clone();
        ops::packed_qkv_attention(ctx, args)?;
        attention
    } else {
        let mut q = output(ctx, vec![tokens, heads, head_dim], DType::F16)?;
        let mut k = output(ctx, vec![tokens, heads, head_dim], DType::F16)?;
        let mut v = output(ctx, vec![tokens, heads, head_dim], DType::F16)?;
        ops::rope(
            ctx,
            ops::RopeArgs {
                semantic: ops::RopeSemantic::SplitQkvBias,
                qkv: &qkv,
                bias: weights.qkv.bias.as_ref(),
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
        let q = q.reshape(vec![views, patches_per_view, heads, head_dim])?;
        let k = k.reshape(vec![views, patches_per_view, heads, head_dim])?;
        let v = v.reshape(vec![views, patches_per_view, heads, head_dim])?;
        let mut attention = output(ctx, q.shape().dims().to_vec(), DType::F16)?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut attention);
        args.policy = policies.attention.clone();
        ops::attention(ctx, args)?;
        attention
    };
    let attention = attention.reshape(vec![tokens, heads * head_dim])?;
    let attention = fixed_quantize(ctx, policies, &attention, scales.attention_output)?;
    let projection = fp8_gemm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
    )?;
    let (hidden, normalized) = bias_residual_layer_normalized_fp8(
        ctx,
        policies,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
        scales.mlp_norm,
    )?;
    let activation = fp8_gemm_bias_gelu_quant(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.fc1,
        scales.mlp_activation,
    )?;
    fp8_gemm_bias_residual(
        ctx,
        policies,
        &activation,
        scales.mlp_activation,
        &weights.fc2,
        &hidden,
    )
}

#[cfg(test)]
mod tests {
    use apxinf_core::Tensor;
    use half::f16;

    use super::*;
    use crate::pi05::backend::{self, Context};
    use crate::pi05::{
        Fp8StaticDeviceLayerNorm, Fp8StaticDeviceVisionBlock, Fp8StaticLinearWeights, LinearWeights,
    };

    fn zero_linear(input: usize, output: usize, backend: &Context) -> Fp8StaticLinearWeights {
        Fp8StaticLinearWeights::from_host(
            &LinearWeights {
                weight: Tensor::from_f32(vec![input, output], &vec![0.0; input * output]).unwrap(),
                bias: None,
            },
            backend,
        )
        .unwrap()
    }

    fn zero_linear_with_bias(
        input: usize,
        output: usize,
        backend: &Context,
    ) -> Fp8StaticLinearWeights {
        Fp8StaticLinearWeights::from_host(
            &LinearWeights {
                weight: Tensor::from_f32(vec![input, output], &vec![0.0; input * output]).unwrap(),
                bias: Some(Tensor::from_f32(vec![output], &vec![0.0; output]).unwrap()),
            },
            backend,
        )
        .unwrap()
    }

    #[test]
    fn zero_weight_language_layer_is_residual_identity() {
        let backend = Context::new(0).unwrap();
        let config = GemmaVariantConfig {
            width: 16,
            depth: 1,
            mlp_dim: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
        };
        let norm = Tensor::from_f16(vec![16], &vec![f16::ONE; 16]).unwrap();
        let weights = Fp8StaticDeviceLanguageLayer {
            input_norm_scale: backend::to_device(&backend, &norm).unwrap(),
            qkv: zero_linear(16, 32, &backend),
            output: zero_linear(16, 16, &backend),
            post_attention_norm_scale: backend::to_device(&backend, &norm).unwrap(),
            gate_up: zero_linear(16, 64, &backend),
            down: zero_linear(32, 16, &backend),
        };
        let source = (0..64)
            .map(|i| f16::from_f32((i as f32 - 31.0) / 32.0))
            .collect::<Vec<_>>();
        let input = backend::to_device(
            &backend,
            &Tensor::from_f16(vec![4, 16], &source).unwrap(),
        )
        .unwrap();
        let output = language_layer_fp8_static(
            &backend,
            config,
            &weights,
            Fp8StaticTransformerLayerScales {
                attention_norm: 0.01,
                attention_output: 0.01,
                mlp_norm: 0.01,
                mlp_activation: 0.01,
            },
            &input,
            true,
            0,
            1e-6,
            10_000.0,
        )
        .unwrap();
        let output = backend::to_cpu(&output.hidden).unwrap();
        assert_eq!(output.as_f16().unwrap(), source.as_slice());
    }

    #[test]
    fn zero_weight_vision_layer_is_residual_identity_across_views() {
        let backend = Context::new(0).unwrap();
        let width = 16;
        let inner = 32;
        let heads = 2;
        let head_dim = 8;
        let affine = Fp8StaticDeviceLayerNorm {
            weight: backend::to_device(
                &backend,
                &Tensor::from_f16(vec![width], &vec![f16::ONE; width]).unwrap(),
            )
            .unwrap(),
            bias: backend::to_device(
                &backend,
                &Tensor::from_f16(vec![width], &vec![f16::ZERO; width]).unwrap(),
            )
            .unwrap(),
        };
        let weights = Fp8StaticDeviceVisionBlock {
            norm1: Fp8StaticDeviceLayerNorm {
                weight: affine.weight.clone(),
                bias: affine.bias.clone(),
            },
            qkv: zero_linear_with_bias(width, 3 * width, &backend),
            output: zero_linear_with_bias(width, width, &backend),
            norm2: affine,
            fc1: zero_linear_with_bias(width, inner, &backend),
            fc2: zero_linear_with_bias(inner, width, &backend),
        };
        let source = (0..8 * width)
            .map(|i| f16::from_f32((i as f32 - 63.0) / 64.0))
            .collect::<Vec<_>>();
        let input = backend::to_device(
            &backend,
            &Tensor::from_f16(vec![8, width], &source).unwrap(),
        )
        .unwrap();
        let output = vision_layer_fp8_static(
            &backend,
            &weights,
            Fp8StaticVisionLayerScales {
                attention_norm: 0.01,
                attention_output: 0.01,
                mlp_norm: 0.01,
                mlp_activation: 0.01,
            },
            &input,
            4,
            heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        let output = backend::to_cpu(&output).unwrap();
        assert_eq!(output.as_f16().unwrap(), source.as_slice());
    }

    #[test]
    fn action_attention_contract_is_non_causal_and_executes() {
        let backend = Context::new(0).unwrap();
        let query_shape = vec![1, 10, 8, 256];
        let cache_shape = vec![1, 533, 1, 256];
        let query = output(&backend, query_shape.clone(), DType::F16).unwrap();
        let key = output(&backend, cache_shape.clone(), DType::F16).unwrap();
        let value = output(&backend, cache_shape, DType::F16).unwrap();
        let mut attention = output(&backend, query_shape, DType::F16).unwrap();
        let mut policy = ops::AttentionPolicy::default();
        policy.workspace_limit = 64 * 1024 * 1024;
        policy.online_tune = false;
        policy.allow_fallback = true;
        policy.graph_safe = false;
        policy.deterministic = true;
        policy.cache_dir = Some("action-attention-test".into());
        let args = action_attention_args(&query, &key, &value, &mut attention, 532, &policy);
        assert_eq!(args.mask, ops::AttentionMask::None);
        assert_eq!(args.query_start, 0);
        assert_eq!(args.valid_key_tokens, 532);
        assert_eq!(args.policy.workspace_limit, policy.workspace_limit);
        assert_eq!(args.policy.online_tune, policy.online_tune);
        assert_eq!(args.policy.allow_fallback, policy.allow_fallback);
        assert_eq!(args.policy.graph_safe, policy.graph_safe);
        assert_eq!(args.policy.deterministic, policy.deterministic);
        assert_eq!(args.policy.cache_dir, policy.cache_dir);
        ops::kv_cache_attention(&backend, args).unwrap();
        backend::synchronize(&backend).unwrap();
    }
}

// Precision-specific backbone operations share this file with their layers.
pub(in crate::pi05::model) mod backbone {
    use super::*;
    use crate::pi05::backend::{Context, DeviceBuffer as CudaBuffer};
    use crate::pi05::weights::*;
    use crate::pi05::Pi05Config;
    use apxinf_core::{Error, Result, Tensor};
    use std::sync::Arc;
    pub struct Fp8StaticPrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Fp8StaticStepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub struct Fp8StaticBlocks {
        pub(in crate::pi05::model) backend: Arc<Context>,
        pub(in crate::pi05::model) config: Arc<Pi05Config>,
        pub(in crate::pi05::model) weights: Arc<Fp8StaticWeights>,
        pub(in crate::pi05::model) scales: Arc<Fp8StaticActivationScales>,
        pub(in crate::pi05::model) policies: Fp8L3Policies,
    }
    impl Fp8StaticBlocks {
        pub(in crate::pi05) fn new(
            backend: Arc<Context>,
            config: Arc<Pi05Config>,
            weights: Arc<Fp8StaticWeights>,
            scales: Arc<Fp8StaticActivationScales>,
            policies: Fp8L3Policies,
        ) -> Result<Self> {
            config.validate()?;
            scales.validate(&config)?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other("π0.5 device weight depth mismatch".into()));
            }
            Ok(Self {
                backend,
                config,
                weights,
                scales,
                policies,
            })
        }

        fn ctx(&self) -> &Context {
            &self.backend
        }

        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            let patches = fixed_quantize(
                self.ctx(),
                &self.policies,
                patches,
                self.scales.vision_patch_input,
            )?;
            self.encode_vision_fp8_patches(&patches)
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
            let mut language = output(self.ctx(), vec![token_count, width], DType::F16)?;
            let mut args = ops::GatherArgs::new(
                ops::GatherSemantic::EmbeddingLookup,
                &self.weights.token_embedding,
                &mut language,
            );
            args.ids = Some(token_ids);
            args.vocab_size = self.weights.token_embedding.shape().dims()[0];
            ops::gather(self.ctx(), args)?;
            ops::concat_rows(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Fp8StaticPrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            for (index, (layer, scale)) in self
                .weights
                .language_layers
                .iter()
                .zip(&self.scales.language_layers)
                .enumerate()
            {
                let output = language_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.language,
                    layer,
                    *scale,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(ops::reserve_prefix(
                    self.ctx(),
                    &output.key,
                    cache_rows,
                )?);
                values.push(ops::reserve_prefix(
                    self.ctx(),
                    &output.value,
                    cache_rows,
                )?);
            }
            Ok(Fp8StaticPrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let input = fixed_quantize(
                self.ctx(),
                &self.policies,
                time_embedding,
                self.scales.time_input,
            )?;
            let hidden = fp8_gemm(
                self.ctx(),
                &self.policies,
                &input,
                self.scales.time_input,
                &self.weights.time_mlp_in,
            )?;
            let hidden = bias_activation(
                self.ctx(),
                &self.policies,
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )?;
            let hidden = fixed_quantize(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.time_hidden,
            )?;
            let output = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.time_hidden,
                &self.weights.time_mlp_out,
            )?;
            bias_activation(
                self.ctx(),
                &self.policies,
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )
        }

        fn modulation(
            &self,
            conditioning: &Tensor,
            weights: &crate::pi05::Fp8StaticLinearWeights,
        ) -> Result<Tensor> {
            let projected = fp8_gemm(
                self.ctx(),
                &self.policies,
                conditioning,
                self.scales.conditioning,
                weights,
            )?;
            let modulation = match weights.bias.as_ref() {
                Some(bias) => bias_activation(
                    self.ctx(),
                    &self.policies,
                    &projected,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => projected,
            };
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(
            &self,
            time_embedding: &Tensor,
        ) -> Result<Fp8StaticStepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let conditioning = fixed_quantize(
                self.ctx(),
                &self.policies,
                &conditioning,
                self.scales.conditioning,
            )?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Fp8StaticStepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Fp8StaticStepModulation>> {
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
            modulation: &Fp8StaticStepModulation,
            prefix: &Fp8StaticPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 prefix KV/modulation depth mismatch".into(),
                ));
            }
            let state_fp8 = fixed_quantize(
                self.ctx(),
                &self.policies,
                state,
                self.scales.action_input,
            )?;
            let hidden = fp8_gemm(
                self.ctx(),
                &self.policies,
                &state_fp8,
                self.scales.action_input,
                &self.weights.action_in,
            )?;
            let mut hidden = match self.weights.action_in.bias.as_ref() {
                Some(bias) => bias_activation(
                    self.ctx(),
                    &self.policies,
                    &hidden,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => hidden,
            };

            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let (next_norm_modulation, next_norm_scale) =
                    if index + 1 < self.config.action_expert.depth {
                        (
                            &modulation.attention[index + 1],
                            self.scales.action_layers[index + 1].attention_norm,
                        )
                    } else {
                        (&modulation.final_norm, self.scales.action_final_norm)
                    };
                let output = action_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.action_expert,
                    layer,
                    self.scales.action_layers[index],
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention[index],
                    &modulation.mlp[index],
                    next_norm_modulation,
                    next_norm_scale,
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
            let velocity = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.action_final_norm,
                &self.weights.action_out,
            )?;
            let velocity = match self.weights.action_out.bias.as_ref() {
                Some(bias) => bias_activation(
                    self.ctx(),
                    &self.policies,
                    &velocity,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => velocity,
            };
            let mut updated = output(self.ctx(), state.shape().dims().to_vec(), state.dtype())?;
            let mut args =
                ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut updated);
            args.secondary = Some(&velocity);
            args.dt = dt;
            ops::pointwise(self.ctx(), args)?;
            Ok(updated)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Fp8StaticPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }

        fn encode_vision_fp8_patches(&self, patches: &Tensor) -> Result<Tensor> {
            let mut hidden = vision_patch_embed_fp8_static_native_with_policies(
                self.ctx(),
                &self.policies,
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
                self.scales.vision_patch_input,
            )?;
            for (layer, scale) in self
                .weights
                .vision_layers
                .iter()
                .zip(&self.scales.vision_layers)
            {
                hidden = vision_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    layer,
                    *scale,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = layer_normalized_fp8(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
                self.scales.vision_post_norm,
            )?;
            let projected = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.vision_post_norm,
                &self.weights.multimodal_projector,
            )?;
            match self.weights.multimodal_projector.bias.as_ref() {
                Some(bias) => bias_activation(
                    self.ctx(),
                    &self.policies,
                    &projected,
                    Some(bias),
                    ops::PointwiseActivation::None,
                ),
                None => Ok(projected),
            }
        }
    }

    impl super::super::Blocks for Fp8StaticBlocks {
        type Prefix = Fp8StaticPrefixKvCache;
        type StepModulation = Fp8StaticStepModulation;
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor> {
            if native {
                self.encode_vision_fp8_patches(patches)
            } else {
                self.encode_vision(patches)
            }
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
            _embeddings: &[Tensor],
        ) -> Result<Option<Vec<Self::StepModulation>>> {
            Ok(None)
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

impl crate::pi05::model::PrepareBlocks for backbone::Fp8StaticBlocks {
    fn backend(&self) -> &std::sync::Arc<crate::pi05::backend::Context> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::pi05::model::WorkspaceRequirements> {
        Ok(crate::pi05::model::WorkspaceRequirements {
            bytes: self.config.cuda_graph_workspace_bytes_fp8_static(tokens)?,
        })
    }
    fn raw_patch_dtype(&self) -> apxinf_core::DType {
        apxinf_core::DType::F8E4M3
    }
    fn preprocess(
        &self,
        images: &crate::pi05::backend::DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        let ctx = &self.backend;
        let mut f16_patches = output(ctx, patches.shape().dims().to_vec(), DType::F16)?;
        let geometry = ops::GatherPatchGeometry {
            views: self.config.num_views,
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
            nhwc: matches!(layout, crate::pi05::Pi05ImageLayout::Nhwc),
        };
        let gather = ops::GatherArgs::rgb_to_patches(images, &mut f16_patches, geometry);
        ops::gather(ctx, gather)?;
        let mut patches = patches.clone();
        let mut quantization = ops::QuantizationArgs::new(
            ops::QuantizationSemantic::FixedScaleE4m3,
            &f16_patches,
            &mut patches,
        );
        quantization.scale = self.scales.vision_patch_input;
        ops::quantization(ctx, quantization)
    }
}
