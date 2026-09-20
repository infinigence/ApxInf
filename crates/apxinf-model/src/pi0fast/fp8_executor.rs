//! π0-FAST FP8 E4M3 transformer-layer execution.
//!
//! Same layer topology, norms, RoPE, attention, residual paths and
//! `gelu(gate) * up` epilogue as `bf16_executor`. The single difference is that
//! every projection quantizes its BF16 activation to E4M3 under a calibrated
//! per-tensor scale and then dispatches to `gemm::fp8_bf16`, whose BF16 output
//! feeds the unchanged BF16 tail.
//!
//! The fused dual-GeGLU epilogue is not used here: π0-FAST packs gate/up plainly
//! (`dual_geglu_interleaved = false`), and at one token the extra BF16 pass over
//! the `[1, 2*mlp]` gate/up result is a few tens of kilobytes against a hundred
//! megabytes of weights.

use super::backend::{kernels, Context};
use apxinf_core::{Result, Tensor};
use kernels::{activation, attention, fused, gemm, norm, quantization, rope};

use super::{
    Fp8DeviceLanguageLayer, Fp8DeviceVisionBlock, Pi0FastLanguageConfig,
};

/// Activation scales for one vision block, one per GEMM input.
#[derive(Clone, Copy, Debug)]
pub struct VisionLayerScales {
    /// Input to `qkv`, i.e. the output of `norm1`.
    pub qkv_input: f32,
    /// Input to the attention output projection.
    pub attention_output: f32,
    /// Input to `fc1`, i.e. the output of `norm2`.
    pub fc1_input: f32,
    /// Input to `fc2`, i.e. the GELU activation.
    pub fc2_input: f32,
}

/// Activation scales for one language layer, one per GEMM input.
#[derive(Clone, Copy, Debug)]
pub struct LanguageLayerScales {
    /// Input to `qkv`, i.e. the output of the input RMSNorm.
    pub qkv_input: f32,
    /// Input to the attention output projection.
    pub attention_output: f32,
    /// Input to `gate_up`, i.e. the post-attention RMSNorm output.
    pub gate_up_input: f32,
    /// Input to `down`, i.e. the `gelu(gate) * up` activation.
    pub down_input: f32,
}

/// Every activation site the FP8 runtime needs a scale for.
#[derive(Clone, Debug)]
pub struct Pi0FastFp8Scales {
    pub vision_layers: Vec<VisionLayerScales>,
    pub multimodal_projector: f32,
    pub language_layers: Vec<LanguageLayerScales>,
    /// Input to the LM head, i.e. the final RMSNorm output.
    pub lm_head: f32,
}

impl Pi0FastFp8Scales {
    /// One scale for every site. Used to bring the path up before calibration
    /// exists, and by synthetic runs; a real deployment wants measured scales.
    pub fn uniform(config: &super::Pi0FastConfig, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(apxinf_core::Error::Other(format!(
                "π0-FAST FP8 scale must be finite and positive, got {scale}"
            )));
        }
        let vision = VisionLayerScales {
            qkv_input: scale,
            attention_output: scale,
            fc1_input: scale,
            fc2_input: scale,
        };
        let language = LanguageLayerScales {
            qkv_input: scale,
            attention_output: scale,
            gate_up_input: scale,
            down_input: scale,
        };
        Ok(Self {
            vision_layers: vec![vision; config.vision_depth],
            multimodal_projector: scale,
            language_layers: vec![language; config.language.depth],
            lm_head: scale,
        })
    }

    /// Resolve every graph activation scale from a measured calibration profile.
    ///
    /// One scale per GEMM input, in the order the runtime consumes them. The
    /// profile has already been validated against this config's site plan, so a
    /// missing name here is a programming error rather than a data error.
    pub fn from_calibration(
        config: &super::Pi0FastConfig,
        calibration: &super::fp8_calibration::Pi0FastFp8Calibration,
    ) -> Result<Self> {
        use super::fp8_calibration::{LM_HEAD_SITE, MULTIMODAL_PROJECTOR_SITE};
        let plan = super::fp8_calibration::Pi0FastCalibrationPlan::for_config(config);
        let vision_layers = plan
            .vision_layers()
            .iter()
            .map(|sites| {
                Ok(VisionLayerScales {
                    qkv_input: calibration.scale(&sites.qkv_input)?,
                    attention_output: calibration.scale(&sites.attention_output)?,
                    fc1_input: calibration.scale(&sites.fc1_input)?,
                    fc2_input: calibration.scale(&sites.fc2_input)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let language_layers = plan
            .language_layers()
            .iter()
            .map(|sites| {
                Ok(LanguageLayerScales {
                    qkv_input: calibration.scale(&sites.qkv_input)?,
                    attention_output: calibration.scale(&sites.attention_output)?,
                    gate_up_input: calibration.scale(&sites.gate_up_input)?,
                    down_input: calibration.scale(&sites.down_input)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vision_layers,
            multimodal_projector: calibration.scale(MULTIMODAL_PROJECTOR_SITE)?,
            language_layers,
            lm_head: calibration.scale(LM_HEAD_SITE)?,
        })
    }
}

pub struct Fp8LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_fp8(
    ctx: &Context,
    config: Pi0FastLanguageConfig,
    weights: &Fp8DeviceLanguageLayer,
    scales: LanguageLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8LanguageLayerOutput> {
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
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
        return Ok(Fp8LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.k.reshape(vec![tokens, config.head_dim])?,
            value: qkv.v.reshape(vec![tokens, config.head_dim])?,
        });
    }
    let attention = attention::mqa_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = fp8_projection(ctx, &fused.normalized, &weights.gate_up, scales.gate_up_input)?;
    let activated = activation::geglu_bf16(ctx, &gate_up)?;
    let projected = fp8_projection(ctx, &activated, &weights.down, scales.down_input)?;
    let hidden =
        fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Fp8LanguageLayerOutput {
        hidden,
        key: qkv.k.reshape(vec![tokens, config.head_dim])?,
        value: qkv.v.reshape(vec![tokens, config.head_dim])?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_fp8(
    ctx: &Context,
    weights: &Fp8DeviceVisionBlock,
    scales: VisionLayerScales,
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
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
    let qkv =
        attention::split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = attention::mha_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
    let fused = fused::bias_residual_layer_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = fp8_projection(ctx, &fused.normalized, &weights.fc1, scales.fc1_input)?;
    let activation = activation::bias_gelu_bf16(ctx, &activation, weights.fc1.bias.as_ref())?;
    let projection = fp8_projection(ctx, &activation, &weights.fc2, scales.fc2_input)?;
    fused::bias_residual_bf16(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

/// One cached (single-token) language layer for the autoregressive loop.
#[allow(clippy::too_many_arguments)]
pub fn language_layer_cached_decode_fp8(
    ctx: &Context,
    config: Pi0FastLanguageConfig,
    weights: &Fp8DeviceLanguageLayer,
    scales: LanguageLayerScales,
    input: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
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
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let gate_up = fp8_projection(ctx, &fused.normalized, &weights.gate_up, scales.gate_up_input)?;
    let activated = activation::geglu_bf16(ctx, &gate_up)?;
    let projected = fp8_projection(ctx, &activated, &weights.down, scales.down_input)?;
    fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)
}

/// Quantize `input` to E4M3, then run the FP8 GEMM with a BF16 output.
///
/// The LM head uses this too; at one token it is the widest GEMM in the model.
pub(super) fn fp8_projection(
    ctx: &Context,
    input: &Tensor,
    weight: &super::Fp8LinearWeights,
    activation_scale: f32,
) -> Result<Tensor> {
    let quantized = quantization::quantize_bf16_e4m3(ctx, input, activation_scale)?;
    gemm::fp8_bf16(ctx, &quantized, activation_scale, weight.as_kernel_view())
}
