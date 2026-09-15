//! Fully materialized static-FP8 π0.5 weights.

use apxinf_core::{Backend, Result, Tensor};

use super::{
    fp16_to_device, ActionLayerWeights, AdaRmsNormWeights, Fp8LinearWeights, LanguageLayerWeights,
    LayerNormWeights, Pi05Weights, VisionBlockWeights,
};

#[derive(Debug)]
pub struct DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct DeviceVisionBlock {
    pub norm1: DeviceLayerNorm,
    pub qkv: Fp8LinearWeights,
    pub output: Fp8LinearWeights,
    pub norm2: DeviceLayerNorm,
    pub fc1: Fp8LinearWeights,
    pub fc2: Fp8LinearWeights,
}

#[derive(Debug)]
pub struct DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Fp8LinearWeights,
    pub output: Fp8LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Fp8LinearWeights,
    pub down: Fp8LinearWeights,
}

#[derive(Debug)]
pub struct DeviceActionLayer {
    pub input_style: Fp8LinearWeights,
    pub qkv: Fp8LinearWeights,
    pub output: Fp8LinearWeights,
    pub post_attention_style: Fp8LinearWeights,
    pub gate_up: Fp8LinearWeights,
    pub down: Fp8LinearWeights,
}

#[derive(Debug)]
pub struct StaticFp8Pi05Weights {
    pub patch_embedding: Fp8LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<DeviceVisionBlock>,
    pub vision_post_norm: DeviceLayerNorm,
    pub multimodal_projector: Fp8LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<DeviceActionLayer>,
    pub action_final_style: Fp8LinearWeights,
    pub action_in: Fp8LinearWeights,
    pub action_out: Fp8LinearWeights,
    pub time_mlp_in: Fp8LinearWeights,
    pub time_mlp_out: Fp8LinearWeights,
}

impl StaticFp8Pi05Weights {
    pub fn from_host(
        weights: &Pi05Weights,
        backend: &dyn Backend,
        language_dual_layout: bool,
    ) -> Result<Self> {
        let vision_layers = weights
            .vision
            .blocks
            .iter()
            .map(|layer| DeviceVisionBlock::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let language_layers = weights
            .language_layers
            .iter()
            .map(|layer| DeviceLanguageLayer::from_host(layer, backend, language_dual_layout))
            .collect::<Result<Vec<_>>>()?;
        let action_layers = weights
            .action_layers
            .iter()
            .map(|layer| DeviceActionLayer::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding: Fp8LinearWeights::from_host(&weights.vision.patch_embedding, backend)?,
            position_embedding: fp16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers,
            vision_post_norm: DeviceLayerNorm::from_host(&weights.vision.post_layer_norm, backend)?,
            multimodal_projector: Fp8LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: fp16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers,
            language_final_norm_scale: fp16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers,
            action_final_style: style_to_device(&weights.action_final_norm, backend)?,
            action_in: Fp8LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Fp8LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Fp8LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Fp8LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: fp16_to_device(&weights.weight, backend)?,
            bias: fp16_to_device(&weights.bias, backend)?,
        })
    }
}

impl DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Fp8LinearWeights::from_host_parts(&[&weights.q, &weights.k, &weights.v], backend)?,
            output: Fp8LinearWeights::from_host(&weights.output, backend)?,
            norm2: DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Fp8LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Fp8LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl DeviceLanguageLayer {
    fn from_host(
        weights: &LanguageLayerWeights,
        backend: &dyn Backend,
        allow_dual_layout: bool,
    ) -> Result<Self> {
        Ok(Self {
            input_norm_scale: fp16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Fp8LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: fp16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Fp8LinearWeights::from_host_parts_with_dual_layout(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
                allow_dual_layout,
            )?,
            down: Fp8LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_style: style_to_device(&weights.input_norm, backend)?,
            qkv: Fp8LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_style: style_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Fp8LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp8LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn style_to_device(weights: &AdaRmsNormWeights, backend: &dyn Backend) -> Result<Fp8LinearWeights> {
    Fp8LinearWeights::from_host(&weights.style, backend)
}

#[cfg(feature = "cuda")]
pub use activation_scales::Pi05ActivationScales;
#[cfg(feature = "cuda")]
mod activation_scales {
    use crate::pi05::{
        LayerCalibrationSites, Pi05CalibrationPlan, Pi05Config, StaticFp8Calibration,
        TransformerLayerScales, VisionLayerScales,
    };
    use apxinf_core::{Error, Result};
    #[derive(Clone, Debug)]
    pub struct Pi05ActivationScales {
        pub vision_patch_input: f32,
        pub vision_layers: Vec<VisionLayerScales>,
        pub vision_post_norm: f32,
        pub language_layers: Vec<TransformerLayerScales>,
        pub action_input: f32,
        pub time_input: f32,
        pub time_hidden: f32,
        pub conditioning: f32,
        pub action_layers: Vec<TransformerLayerScales>,
        pub action_final_norm: f32,
    }

    impl Pi05ActivationScales {
        /// Resolve every graph activation scale from a named calibration file.
        pub fn from_calibration(
            config: &Pi05Config,
            calibration: &StaticFp8Calibration,
        ) -> Result<Self> {
            let plan = Pi05CalibrationPlan::for_config(config);
            let optional_scale = |site: &Option<String>| -> Result<f32> {
                site.as_deref()
                    .map(|name| calibration.scale(name))
                    .transpose()
                    .map(|scale| scale.unwrap_or(1.0))
            };
            let transformer_layer =
                |sites: &LayerCalibrationSites| -> Result<TransformerLayerScales> {
                    Ok(TransformerLayerScales {
                        attention_norm: calibration.scale(&sites.attention_norm)?,
                        attention_output: optional_scale(&sites.attention_output)?,
                        mlp_norm: optional_scale(&sites.mlp_norm)?,
                        mlp_activation: optional_scale(&sites.mlp_activation)?,
                    })
                };
            let vision_layers = plan
                .vision_layers()
                .iter()
                .map(|sites| {
                    Ok(VisionLayerScales {
                        attention_norm: calibration.scale(&sites.attention_norm)?,
                        attention_output: calibration
                            .scale(sites.attention_output.as_deref().expect("vision tail site"))?,
                        mlp_norm: calibration
                            .scale(sites.mlp_norm.as_deref().expect("vision tail site"))?,
                        mlp_activation: calibration
                            .scale(sites.mlp_activation.as_deref().expect("vision tail site"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let language_layers = plan
                .language_layers()
                .iter()
                .map(transformer_layer)
                .collect::<Result<Vec<_>>>()?;
            let action_layers = plan
                .action_layers()
                .iter()
                .map(transformer_layer)
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                vision_patch_input: calibration.scale("vision.patch_input")?,
                vision_layers,
                vision_post_norm: calibration.scale("vision.post_norm")?,
                language_layers,
                action_input: calibration.scale("action.input")?,
                time_input: calibration.scale("time.input")?,
                time_hidden: calibration.scale("time.hidden")?,
                conditioning: calibration.scale("action.conditioning")?,
                action_layers,
                action_final_norm: calibration.scale("action.final_norm")?,
            })
        }

        /// Useful for kernel smoke tests. Production inference should load named,
        /// measured scales from `StaticFp8Calibration`.
        pub fn uniform(config: &Pi05Config, scale: f32) -> Result<Self> {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::Other(format!("invalid uniform FP8 scale {scale}")));
            }
            let transformer = TransformerLayerScales {
                attention_norm: scale,
                attention_output: scale,
                mlp_norm: scale,
                mlp_activation: scale,
            };
            let vision = VisionLayerScales {
                attention_norm: scale,
                attention_output: scale,
                mlp_norm: scale,
                mlp_activation: scale,
            };
            Ok(Self {
                vision_patch_input: scale,
                vision_layers: vec![vision; config.vision_depth],
                vision_post_norm: scale,
                language_layers: vec![transformer; config.language.depth],
                action_input: scale,
                time_input: scale,
                time_hidden: scale,
                conditioning: scale,
                action_layers: vec![transformer; config.action_expert.depth],
                action_final_norm: scale,
            })
        }

        pub(in crate::pi05) fn validate(&self, config: &Pi05Config) -> Result<()> {
            if self.vision_layers.len() != config.vision_depth
                || self.language_layers.len() != config.language.depth
                || self.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 activation calibration depth mismatch".into(),
                ));
            }
            Ok(())
        }
    }
}
