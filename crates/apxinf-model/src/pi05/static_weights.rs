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
    pub fn from_host(weights: &Pi05Weights, backend: &dyn Backend) -> Result<Self> {
        let vision_layers = weights
            .vision
            .blocks
            .iter()
            .map(|layer| DeviceVisionBlock::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let language_layers = weights
            .language_layers
            .iter()
            .map(|layer| DeviceLanguageLayer::from_host(layer, backend))
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
    fn from_host(weights: &LanguageLayerWeights, backend: &dyn Backend) -> Result<Self> {
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
            gate_up: Fp8LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
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
