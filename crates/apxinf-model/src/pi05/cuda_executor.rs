//! π0.5 CUDA transformer-layer execution.

use super::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{activation, attention, embedding, fused, gemm, norm, quantization, rope};

use super::{DeviceActionLayer, DeviceLanguageLayer, DeviceVisionBlock, GemmaVariantConfig};

#[derive(Clone, Copy, Debug)]
pub struct TransformerLayerScales {
    /// Output scale for the first (Ada)RMSNorm.
    pub attention_norm: f32,
    /// Input scale for the attention output projection.
    pub attention_output: f32,
    /// Output scale for the second (Ada)RMSNorm.
    pub mlp_norm: f32,
    /// Output scale for GELU-tanh(gate) * up.
    pub mlp_activation: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionLayerScales {
    pub attention_norm: f32,
    pub attention_output: f32,
    pub mlp_norm: f32,
    pub mlp_activation: f32,
}

pub struct LanguageLayerOutput {
    pub hidden: Tensor,
    /// Prefix K/V are retained per layer for the paired action expert.
    pub key: Tensor,
    pub value: Tensor,
}

pub struct ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

pub fn language_layer(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &DeviceLanguageLayer,
    scales: TransformerLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<LanguageLayerOutput> {
    let normalized = norm::rms_quant_f16_e4m3(
        ctx,
        input,
        &weights.input_norm_scale,
        rms_eps,
        scales.attention_norm,
    )?;
    let qkv = gemm::fp8(
        ctx,
        &normalized,
        scales.attention_norm,
        weights.qkv.as_kernel_view(),
    )?;
    let qkv = rope::split_qkv_apply_f16(
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
        return Ok(LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.k.reshape(vec![tokens, config.head_dim])?,
            value: qkv.v.reshape(vec![tokens, config.head_dim])?,
        });
    }
    let attention = attention::mqa_f16(ctx, &qkv.q, &qkv.k, &qkv.v)?.reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let attention = quantization::quantize_f16_e4m3(ctx, &attention, scales.attention_output)?;
    let projected = gemm::fp8(
        ctx,
        &attention,
        scales.attention_output,
        weights.output.as_kernel_view(),
    )?;
    let fused = fused::bias_residual_rms_quant_f16_e4m3(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
        scales.mlp_norm,
    )?;
    let hidden = fused.hidden;
    let normalized = fused.normalized;
    let gate_up = gemm::fp8(
        ctx,
        &normalized,
        scales.mlp_norm,
        weights.gate_up.as_kernel_view(),
    )?;
    let activated = activation::geglu_quant_f16_e4m3(ctx, &gate_up, scales.mlp_activation)?;
    let hidden = fused::gemm_bias_residual_fp8(
        ctx,
        &activated,
        &weights.down.weight,
        weights.down.bias.as_ref(),
        &hidden,
        scales.mlp_activation,
        weights.down.weight_scale,
    )?;
    Ok(LanguageLayerOutput {
        hidden,
        key: qkv.k.reshape(vec![tokens, config.head_dim])?,
        value: qkv.v.reshape(vec![tokens, config.head_dim])?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &DeviceActionLayer,
    scales: TransformerLayerScales,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_style: &Tensor,
    mlp_style: &Tensor,
    next_norm_style: &Tensor,
    next_norm_scale: f32,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(normalized) => normalized.clone(),
        None => norm::adaptive_rms_quant_f16_e4m3(
            ctx,
            input,
            attention_style,
            rms_eps,
            scales.attention_norm,
        )?,
    };
    let qkv = gemm::fp8(
        ctx,
        &normalized,
        scales.attention_norm,
        weights.qkv.as_kernel_view(),
    )?;
    let q = rope::apply_q_write_kv_f16(
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
    let attention = attention::mqa_cached_f16(
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
    let attention = quantization::quantize_f16_e4m3(ctx, &attention, scales.attention_output)?;
    let projected = gemm::fp8(
        ctx,
        &attention,
        scales.attention_output,
        weights.output.as_kernel_view(),
    )?;
    let fused = fused::adaptive_gate_residual_rms_quant_f16_e4m3(
        ctx,
        &projected,
        input,
        attention_style,
        mlp_style,
        rms_eps,
        scales.mlp_norm,
    )?;
    let hidden = fused.hidden;
    let normalized = fused.normalized;
    let gate_up = gemm::fp8(
        ctx,
        &normalized,
        scales.mlp_norm,
        weights.gate_up.as_kernel_view(),
    )?;
    let activated = activation::geglu_quant_f16_e4m3(ctx, &gate_up, scales.mlp_activation)?;
    let projected = gemm::fp8(
        ctx,
        &activated,
        scales.mlp_activation,
        weights.down.as_kernel_view(),
    )?;
    let fused = fused::adaptive_gate_residual_rms_quant_f16_e4m3(
        ctx,
        &projected,
        &hidden,
        mlp_style,
        next_norm_style,
        rms_eps,
        next_norm_scale,
    )?;
    Ok(ActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

pub fn vision_patch_embed(
    ctx: &Context,
    weights: &super::Fp8LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let patches = quantization::quantize_f16_e4m3(ctx, patches, input_scale)?;
    vision_patch_embed_fp8(
        ctx,
        weights,
        position_embedding,
        &patches,
        patches_per_view,
        input_scale,
    )
}

/// Patch projection when preprocessing has already produced calibrated E4M3
/// patch tokens. This is the entry used by the fused raw-image graph.
pub fn vision_patch_embed_fp8(
    ctx: &Context,
    weights: &super::Fp8LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let projection = gemm::fp8(ctx, patches, input_scale, weights.as_kernel_view())?;
    embedding::add_position_f16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

pub fn vision_qkv_packed_from_env() -> Result<bool> {
    let Some(value) = std::env::var_os("APXINF_CUDA_VISION_QKV_LAYOUT") else {
        return Ok(true);
    };
    match value.to_str() {
        Some("packed") => Ok(true),
        Some("split") => Ok(false),
        Some(value) => Err(apxinf_core::Error::Other(format!(
            "APXINF_CUDA_VISION_QKV_LAYOUT must be packed or split, got {value}"
        ))),
        None => Err(apxinf_core::Error::Other(
            "APXINF_CUDA_VISION_QKV_LAYOUT must be valid UTF-8".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer(
    ctx: &Context,
    weights: &DeviceVisionBlock,
    scales: VisionLayerScales,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    packed_qkv: bool,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = norm::layer_quant_f16_e4m3(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
        scales.attention_norm,
    )?;
    let qkv = if packed_qkv {
        let bias =
            weights.qkv.bias.as_ref().ok_or_else(|| {
                apxinf_core::Error::Other("π0.5 SigLIP QKV bias is required".into())
            })?;
        fused::gemm_bias_fp8(
            ctx,
            &normalized,
            &weights.qkv.weight,
            bias,
            scales.attention_norm,
            weights.qkv.weight_scale,
        )?
    } else {
        gemm::fp8(
            ctx,
            &normalized,
            scales.attention_norm,
            weights.qkv.as_kernel_view(),
        )?
    };
    let attention = if packed_qkv {
        attention::mha_packed_qkv_bias_f16(ctx, &qkv, None, patches_per_view, heads, head_dim)?
    } else {
        let qkv =
            attention::split_qkv_bias_f16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
        attention::mha_f16(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
    }
    .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let attention = quantization::quantize_f16_e4m3(ctx, &attention, scales.attention_output)?;
    let projection = gemm::fp8(
        ctx,
        &attention,
        scales.attention_output,
        weights.output.as_kernel_view(),
    )?;
    let fused = fused::bias_residual_layer_quant_f16_e4m3(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
        scales.mlp_norm,
    )?;
    let hidden = fused.hidden;
    let normalized = fused.normalized;
    let bias = weights
        .fc1
        .bias
        .as_ref()
        .ok_or_else(|| apxinf_core::Error::Other("π0.5 SigLIP fc1 bias is required".into()))?;
    let activation = fused::gemm_bias_gelu_fp8(
        ctx,
        &normalized,
        &weights.fc1.weight,
        bias,
        scales.mlp_norm,
        weights.fc1.weight_scale,
        scales.mlp_activation,
    )?;
    fused::gemm_bias_residual_fp8(
        ctx,
        &activation,
        &weights.fc2.weight,
        weights.fc2.bias.as_ref(),
        &hidden,
        scales.mlp_activation,
        weights.fc2.weight_scale,
    )
}

#[cfg(test)]
mod tests {
    use apxinf_core::{Backend, Tensor};
    use half::f16;

    use super::*;
    use crate::pi05::backend::RuntimeBackend as CudaBackend;
    use crate::pi05::{DeviceLayerNorm, DeviceVisionBlock, Fp8LinearWeights, LinearWeights};

    fn zero_linear(input: usize, output: usize, backend: &dyn Backend) -> Fp8LinearWeights {
        Fp8LinearWeights::from_host(
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
        backend: &dyn Backend,
    ) -> Fp8LinearWeights {
        Fp8LinearWeights::from_host(
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
        let backend = CudaBackend::new(0).unwrap();
        let config = GemmaVariantConfig {
            width: 16,
            depth: 1,
            mlp_dim: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
        };
        let norm = Tensor::from_f16(vec![16], &vec![f16::ONE; 16]).unwrap();
        let weights = DeviceLanguageLayer {
            input_norm_scale: backend.to_device(&norm).unwrap(),
            qkv: zero_linear(16, 32, &backend),
            output: zero_linear(16, 16, &backend),
            post_attention_norm_scale: backend.to_device(&norm).unwrap(),
            gate_up: zero_linear(16, 64, &backend),
            down: zero_linear(32, 16, &backend),
        };
        let source = (0..64)
            .map(|i| f16::from_f32((i as f32 - 31.0) / 32.0))
            .collect::<Vec<_>>();
        let input = backend
            .to_device(&Tensor::from_f16(vec![4, 16], &source).unwrap())
            .unwrap();
        let output = language_layer(
            backend.context(),
            config,
            &weights,
            TransformerLayerScales {
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
        let output = backend.to_cpu(&output.hidden).unwrap();
        assert_eq!(output.as_f16().unwrap(), source.as_slice());
    }

    #[test]
    fn zero_weight_vision_layer_is_residual_identity_across_views() {
        let backend = CudaBackend::new(0).unwrap();
        let width = 16;
        let inner = 32;
        let heads = 2;
        let head_dim = 8;
        let affine = DeviceLayerNorm {
            weight: backend
                .to_device(&Tensor::from_f16(vec![width], &vec![f16::ONE; width]).unwrap())
                .unwrap(),
            bias: backend
                .to_device(&Tensor::from_f16(vec![width], &vec![f16::ZERO; width]).unwrap())
                .unwrap(),
        };
        let weights = DeviceVisionBlock {
            norm1: DeviceLayerNorm {
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
        let input = backend
            .to_device(&Tensor::from_f16(vec![8, width], &source).unwrap())
            .unwrap();
        let output = vision_layer(
            backend.context(),
            &weights,
            VisionLayerScales {
                attention_norm: 0.01,
                attention_output: 0.01,
                mlp_norm: 0.01,
                mlp_activation: 0.01,
            },
            &input,
            4,
            heads,
            head_dim,
            true,
            1e-6,
        )
        .unwrap();
        let output = backend.to_cpu(&output).unwrap();
        assert_eq!(output.as_f16().unwrap(), source.as_slice());
    }
}
