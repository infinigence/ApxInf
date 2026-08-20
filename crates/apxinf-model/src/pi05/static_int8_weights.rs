//! Fully materialized W8A8 π0.5 weights.

use apxinf_core::{Result, Tensor};

use super::backend::RuntimeBackend;
use super::{
    bf16_to_device, ActionLayerWeights, AdaRmsNormWeights, Int8LinearWeights, LanguageLayerWeights,
    LayerNormWeights, Pi05Weights, VisionBlockWeights,
};

pub struct Int8DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct Int8DeviceVisionBlock {
    pub norm1: Int8DeviceLayerNorm,
    pub qkv: Int8LinearWeights,
    pub output: Int8LinearWeights,
    pub norm2: Int8DeviceLayerNorm,
    pub fc1: Int8LinearWeights,
    pub fc2: Int8LinearWeights,
}

pub struct Int8DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Int8LinearWeights,
    pub output: Int8LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Int8LinearWeights,
    pub down: Int8LinearWeights,
}

pub struct Int8DeviceActionLayer {
    pub input_style: Int8LinearWeights,
    pub qkv: Int8LinearWeights,
    pub output: Int8LinearWeights,
    pub post_attention_style: Int8LinearWeights,
    pub gate_up: Int8LinearWeights,
    pub down: Int8LinearWeights,
}

pub struct StaticInt8Pi05Weights {
    pub patch_embedding: Int8LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Int8DeviceVisionBlock>,
    pub vision_post_norm: Int8DeviceLayerNorm,
    pub multimodal_projector: Int8LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Int8DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Int8DeviceActionLayer>,
    pub action_final_style: Int8LinearWeights,
    pub action_in: Int8LinearWeights,
    pub action_out: Int8LinearWeights,
    pub time_mlp_in: Int8LinearWeights,
    pub time_mlp_out: Int8LinearWeights,
}

impl StaticInt8Pi05Weights {
    pub fn from_host(weights: &Pi05Weights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            patch_embedding: Int8LinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Int8DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Int8DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Int8LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Int8DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Int8DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_style: style_to_device(&weights.action_final_norm, backend)?,
            action_in: Int8LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Int8LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Int8LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Int8LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Int8DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Int8DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            norm1: Int8DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Int8LinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Int8LinearWeights::from_host(&weights.output, backend)?,
            norm2: Int8DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Int8LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Int8LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Int8DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Int8LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Int8LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Int8LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Int8LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Int8DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            input_style: style_to_device(&weights.input_norm, backend)?,
            qkv: Int8LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Int8LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_style: style_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Int8LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Int8LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn style_to_device(
    weights: &AdaRmsNormWeights,
    backend: &RuntimeBackend,
) -> Result<Int8LinearWeights> {
    Int8LinearWeights::from_host(&weights.style, backend)
}
