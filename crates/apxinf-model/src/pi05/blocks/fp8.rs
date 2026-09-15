//! π0.5 FP8 CUDA transformer-layer execution.

use crate::pi05::backend::{kernels, Context};
use apxinf_core::{Error, Result, Tensor};
use kernels::{activation, attention, embedding, fused, gemm, norm, quantization, rope};

use crate::pi05::{DeviceActionLayer, DeviceLanguageLayer, DeviceVisionBlock, GemmaVariantConfig};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fa2DirectE4m3Mode {
    Auto,
    Off,
    On,
}

fn parse_fa2_direct_e4m3_mode(value: Option<&str>) -> Result<Fa2DirectE4m3Mode> {
    match value {
        None | Some("auto") => Ok(Fa2DirectE4m3Mode::Auto),
        Some("0" | "off") => Ok(Fa2DirectE4m3Mode::Off),
        Some("1" | "on") => Ok(Fa2DirectE4m3Mode::On),
        Some(value) => Err(Error::Other(format!(
            "APXINF_PI05_FA2_DIRECT_E4M3 must be auto, 0/off, or 1/on; got {value}"
        ))),
    }
}

fn fa2_direct_e4m3_mode() -> Result<Fa2DirectE4m3Mode> {
    match std::env::var("APXINF_PI05_FA2_DIRECT_E4M3") {
        Ok(value) => parse_fa2_direct_e4m3_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_fa2_direct_e4m3_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::Other(
            "APXINF_PI05_FA2_DIRECT_E4M3 must be valid Unicode".into(),
        )),
    }
}

fn fa2_direct_e4m3_exact_shape(q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
    q.shape().dims() == [522, 8, 256]
        && k.shape().dims() == [522, 1, 256]
        && v.shape().dims() == [522, 1, 256]
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
    let fa2_direct = match fa2_direct_e4m3_mode()? {
        Fa2DirectE4m3Mode::Auto => fa2_direct_e4m3_exact_shape(&qkv.q, &qkv.k, &qkv.v),
        Fa2DirectE4m3Mode::Off => false,
        Fa2DirectE4m3Mode::On => true,
    };
    let attention = if fa2_direct {
        attention::mqa_f16_e4m3_522(ctx, &qkv.q, &qkv.k, &qkv.v, scales.attention_output)?.reshape(
            vec![input.shape().dims()[0], config.num_heads * config.head_dim],
        )?
    } else {
        let attention = attention::mqa_f16(ctx, &qkv.q, &qkv.k, &qkv.v)?.reshape(vec![
            input.shape().dims()[0],
            config.num_heads * config.head_dim,
        ])?;
        quantization::quantize_f16_e4m3(ctx, &attention, scales.attention_output)?
    };
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
    let activated = if let Some(activated) = gemm::fp8_geglu_fused(
        ctx,
        &normalized,
        scales.mlp_norm,
        weights.gate_up.as_kernel_view(),
        scales.mlp_activation,
    )? {
        activated
    } else {
        let gate_up = gemm::fp8(
            ctx,
            &normalized,
            scales.mlp_norm,
            weights.gate_up.as_kernel_view(),
        )?;
        activation::geglu_quant_f16_e4m3(ctx, &gate_up, scales.mlp_activation)?
    };
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

#[cfg(test)]
mod fa2_direct_e4m3_tests {
    use super::{parse_fa2_direct_e4m3_mode, Fa2DirectE4m3Mode};

    #[test]
    fn fa2_direct_e4m3_parser_is_fail_closed() {
        assert_eq!(
            parse_fa2_direct_e4m3_mode(None).unwrap(),
            Fa2DirectE4m3Mode::Auto
        );
        assert_eq!(
            parse_fa2_direct_e4m3_mode(Some("auto")).unwrap(),
            Fa2DirectE4m3Mode::Auto
        );
        for value in ["0", "off"] {
            assert_eq!(
                parse_fa2_direct_e4m3_mode(Some(value)).unwrap(),
                Fa2DirectE4m3Mode::Off
            );
        }
        for value in ["1", "on"] {
            assert_eq!(
                parse_fa2_direct_e4m3_mode(Some(value)).unwrap(),
                Fa2DirectE4m3Mode::On
            );
        }
        for value in ["", "true", "AUTO", "2"] {
            assert!(parse_fa2_direct_e4m3_mode(Some(value)).is_err());
        }
    }
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
    let key_tokens = position_offset + input.shape().dims()[0];
    let attention = attention::mqa_cached_f16(ctx, &q, prefix_k, prefix_v, key_tokens)?;
    let attention =
        quantization::quantize_f16_e4m3(ctx, &attention, scales.attention_output)?.reshape(
            vec![input.shape().dims()[0], config.num_heads * config.head_dim],
        )?;
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
    weights: &crate::pi05::Fp8LinearWeights,
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
    weights: &crate::pi05::Fp8LinearWeights,
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

// Precision-specific backbone operations share this file with their layers.
pub(in crate::pi05) mod backbone {
    use crate::pi05::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
    use crate::pi05::*;
    use apxinf_core::{Error, Result, Tensor};
    use kernels::{activation, cache, elementwise, embedding, gemm, norm, quantization};
    use std::sync::Arc;
    pub struct PrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub(in crate::pi05) struct Pi05StepStyles {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub(in crate::pi05) struct Fp8Blocks {
        pub(in crate::pi05) backend: Arc<RuntimeBackend>,
        pub(in crate::pi05) config: Arc<Pi05Config>,
        pub(in crate::pi05) weights: Arc<StaticFp8Pi05Weights>,
        scales: Arc<Pi05ActivationScales>,
        packed_vision_qkv: bool,
    }
    impl Fp8Blocks {
        pub fn new(
            backend: Arc<RuntimeBackend>,
            config: Arc<Pi05Config>,
            weights: Arc<StaticFp8Pi05Weights>,
            scales: Arc<Pi05ActivationScales>,
        ) -> Result<Self> {
            config.validate()?;
            scales.validate(&config)?;
            let packed_vision_qkv = vision_qkv_packed_from_env()?;
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
                packed_vision_qkv,
            })
        }

        fn ctx(&self) -> &Context {
            self.backend.context()
        }

        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            let patches = quantization::quantize_f16_e4m3(
                self.ctx(),
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
            let language = embedding::lookup_f16(
                self.ctx(),
                &self.weights.token_embedding,
                token_ids,
                token_count,
            )?;
            elementwise::concat_rows_f16(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<PrefixKvCache> {
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
                let output = language_layer(
                    self.ctx(),
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
                keys.push(cache::reserve_prefix_f16(
                    self.ctx(),
                    &output.key,
                    cache_rows,
                )?);
                values.push(cache::reserve_prefix_f16(
                    self.ctx(),
                    &output.value,
                    cache_rows,
                )?);
            }
            Ok(PrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let input = quantization::quantize_f16_e4m3(
                self.ctx(),
                time_embedding,
                self.scales.time_input,
            )?;
            let hidden = gemm::fp8(
                self.ctx(),
                &input,
                self.scales.time_input,
                self.weights.time_mlp_in.as_kernel_view(),
            )?;
            let hidden = activation::bias_silu_quant_f16_e4m3(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                self.scales.time_hidden,
            )?;
            let output = gemm::fp8(
                self.ctx(),
                &hidden,
                self.scales.time_hidden,
                self.weights.time_mlp_out.as_kernel_view(),
            )?;
            activation::bias_silu_f16(self.ctx(), &output, self.weights.time_mlp_out.bias.as_ref())
        }

        fn style(
            &self,
            conditioning: &Tensor,
            weights: &crate::pi05::Fp8LinearWeights,
        ) -> Result<Tensor> {
            let projected = gemm::fp8(
                self.ctx(),
                conditioning,
                self.scales.conditioning,
                weights.as_kernel_view(),
            )?;
            let style = elementwise::bias_f16(self.ctx(), &projected, weights.bias.as_ref())?;
            style.reshape(vec![style.numel()])
        }

        fn prepare_step_styles(&self, time_embedding: &Tensor) -> Result<Pi05StepStyles> {
            let conditioning = self.conditioning(time_embedding)?;
            let conditioning = quantization::quantize_f16_e4m3(
                self.ctx(),
                &conditioning,
                self.scales.conditioning,
            )?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.style(&conditioning, &layer.input_style)?);
                mlp.push(self.style(&conditioning, &layer.post_attention_style)?);
            }
            let final_norm = self.style(&conditioning, &self.weights.action_final_style)?;
            Ok(Pi05StepStyles {
                attention,
                mlp,
                final_norm,
            })
        }

        fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<Pi05StepStyles>> {
            if time_embeddings.len() != self.config.num_flow_steps {
                return Err(Error::Other(format!(
                    "π0.5 expected {} timestep embeddings, got {}",
                    self.config.num_flow_steps,
                    time_embeddings.len()
                )));
            }
            time_embeddings
                .iter()
                .map(|embedding| self.prepare_step_styles(embedding))
                .collect()
        }

        fn denoise_step_with_styles(
            &self,
            state: &Tensor,
            styles: &Pi05StepStyles,
            prefix: &PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || styles.attention.len() != self.config.action_expert.depth
                || styles.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other("π0.5 prefix KV/style depth mismatch".into()));
            }
            let state_fp8 =
                quantization::quantize_f16_e4m3(self.ctx(), state, self.scales.action_input)?;
            let hidden = gemm::fp8(
                self.ctx(),
                &state_fp8,
                self.scales.action_input,
                self.weights.action_in.as_kernel_view(),
            )?;
            let mut hidden =
                elementwise::bias_f16(self.ctx(), &hidden, self.weights.action_in.bias.as_ref())?;

            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let (next_norm_style, next_norm_scale) =
                    if index + 1 < self.config.action_expert.depth {
                        (
                            &styles.attention[index + 1],
                            self.scales.action_layers[index + 1].attention_norm,
                        )
                    } else {
                        (&styles.final_norm, self.scales.action_final_norm)
                    };
                let output = action_layer(
                    self.ctx(),
                    self.config.action_expert,
                    layer,
                    self.scales.action_layers[index],
                    &hidden,
                    attention_normalized.as_ref(),
                    &styles.attention[index],
                    &styles.mlp[index],
                    next_norm_style,
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
            let velocity = gemm::fp8(
                self.ctx(),
                &hidden,
                self.scales.action_final_norm,
                self.weights.action_out.as_kernel_view(),
            )?;
            let velocity = elementwise::bias_f16(
                self.ctx(),
                &velocity,
                self.weights.action_out.bias.as_ref(),
            )?;
            elementwise::euler_update_f16(self.ctx(), state, &velocity, dt)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let styles = self.prepare_step_styles(time_embedding)?;
            self.denoise_step_with_styles(state, &styles, prefix, dt)
        }

        fn encode_vision_fp8_patches(&self, patches: &Tensor) -> Result<Tensor> {
            let mut hidden = vision_patch_embed_fp8(
                self.ctx(),
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
                hidden = vision_layer(
                    self.ctx(),
                    layer,
                    *scale,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.packed_vision_qkv,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = norm::layer_quant_f16_e4m3(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
                self.scales.vision_post_norm,
            )?;
            let projected = gemm::fp8(
                self.ctx(),
                &hidden,
                self.scales.vision_post_norm,
                self.weights.multimodal_projector.as_kernel_view(),
            )?;
            elementwise::bias_f16(
                self.ctx(),
                &projected,
                self.weights.multimodal_projector.bias.as_ref(),
            )
        }
    }

    impl super::super::Blocks for Fp8Blocks {
        type Prefix = PrefixKvCache;
        type Styles = Pi05StepStyles;
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
        fn prepare_styles(&self, embeddings: &[Tensor]) -> Result<Vec<Self::Styles>> {
            self.prepare_all_styles(embeddings)
        }
        fn eager_styles(&self, _embeddings: &[Tensor]) -> Result<Option<Vec<Self::Styles>>> {
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
        fn step_with_styles(
            &self,
            state: &Tensor,
            styles: &Self::Styles,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step_with_styles(state, styles, prefix, dt)
        }
    }
}
