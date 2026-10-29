//! π0-FAST FP8 E4M3 transformer-layer execution.
//!
//! Same layer topology, norms, RoPE, attention, residual paths and
//! `gelu(gate) * up` epilogue as `bf16_executor`. The single difference is that
//! every projection quantizes its BF16 activation to E4M3 under a calibrated
//! per-tensor scale and then dispatches through the formal GEMM contract with
//! `Fp8UnitScale`; its BF16 output feeds the unchanged BF16 tail.
//!
//! The fused dual-GeGLU epilogue is not used here: π0-FAST packs gate/up plainly,
//! and at one token the extra BF16 pass over the `[1, 2*mlp]` gate/up result is a
//! few tens of kilobytes against a hundred megabytes of weights.

use super::backend::{ops, Context};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use ops::{gemm, pointwise, quantization};

use super::{Fp8DeviceLanguageLayer, Fp8DeviceVisionBlock, Pi0FastLanguageConfig};

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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::rms(input, &weights.input_norm_scale, &normalized, rms_eps),
    )?;
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
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
        return Ok(Fp8LanguageLayerOutput {
            hidden: input.clone(),
            key: key.reshape(vec![tokens, config.head_dim])?,
            value: value.reshape(vec![tokens, config.head_dim])?,
        });
    }
    let attention = super::attention_mqa(ctx, &query, &key, &value, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
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
    let gate_up = fp8_projection(ctx, &normalized, &weights.gate_up, scales.gate_up_input)?;
    let activated = geglu(ctx, &gate_up)?;
    let projected = fp8_projection(ctx, &activated, &weights.down, scales.down_input)?;
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
    Ok(Fp8LanguageLayerOutput {
        hidden,
        key: key.reshape(vec![tokens, config.head_dim])?,
        value: value.reshape(vec![tokens, config.head_dim])?,
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
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
    let qkv = super::split_qkv_bias(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = super::attention_mha(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
    let hidden = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(projected.shape().clone(), DType::BF16)?;
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input: &projected,
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
    let activation = fp8_projection(ctx, &normalized, &weights.fc1, scales.fc1_input)?;
    let activation = bias_activation(
        ctx,
        &activation,
        weights.fc1.bias.as_ref(),
        pointwise::PointwiseActivation::Gelu,
    )?;
    let projection = fp8_projection(ctx, &activation, &weights.fc2, scales.fc2_input)?;
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
    let normalized = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        ctx,
        ops::NormArgs::rms(input, &weights.input_norm_scale, &normalized, rms_eps),
    )?;
    let qkv = fp8_projection(ctx, &normalized, &weights.qkv, scales.qkv_input)?;
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
    let projected = fp8_projection(ctx, &attention, &weights.output, scales.attention_output)?;
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
    let gate_up = fp8_projection(ctx, &normalized, &weights.gate_up, scales.gate_up_input)?;
    let activated = geglu(ctx, &gate_up)?;
    let projected = fp8_projection(ctx, &activated, &weights.down, scales.down_input)?;
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

/// Quantize `input` to E4M3, then run the FP8 GEMM with a BF16 output.
///
/// The LM head uses this too; at one token it is the widest GEMM in the model.
pub(super) fn fp8_projection(
    ctx: &Context,
    input: &Tensor,
    weight: &super::Fp8LinearWeights,
    activation_scale: f32,
) -> Result<Tensor> {
    let quantized = quantization::quantize_fixed_scale(ctx, input, activation_scale)?;
    let a = quantized.shape().dims();
    let b = weight.weight.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "PI0-FAST FP8 GEMM shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    let mut args = gemm::GemmArgs::new(&quantized, &weight.weight, &mut output);
    args.quantization = gemm::GemmQuantization::Fp8UnitScale;
    args.alpha = activation_scale * weight.weight_scale;
    gemm::gemm(ctx, args)?;
    Ok(output)
}
