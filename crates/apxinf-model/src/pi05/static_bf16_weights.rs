//! Fully materialized native-BF16 π0.5 weights.

use apxinf_core::{Backend, Result, Tensor};

use super::{
    bf16_to_device, ActionLayerWeights, AdaRmsNormWeights, Bf16LinearWeights, LanguageLayerWeights,
    LayerNormWeights, Pi05Weights, VisionBlockWeights,
};

#[derive(Debug)]
pub struct Bf16DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Bf16DeviceVisionBlock {
    pub norm1: Bf16DeviceLayerNorm,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub norm2: Bf16DeviceLayerNorm,
    pub fc1: Bf16LinearWeights,
    pub fc2: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceActionLayer {
    pub input_style: Bf16LinearWeights,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_style: Bf16LinearWeights,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct StaticBf16Pi05Weights {
    pub patch_embedding: Bf16LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Bf16DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Bf16LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Bf16DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Bf16DeviceActionLayer>,
    pub action_final_style: Bf16LinearWeights,
    pub action_in: Bf16LinearWeights,
    pub action_out: Bf16LinearWeights,
    pub time_mlp_in: Bf16LinearWeights,
    pub time_mlp_out: Bf16LinearWeights,
}

impl StaticBf16Pi05Weights {
    pub fn from_host(weights: &Pi05Weights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_embedding: Bf16LinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Bf16DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Bf16DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Bf16LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Bf16DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Bf16DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_style: style_to_device(&weights.action_final_norm, backend)?,
            action_in: Bf16LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Bf16LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Bf16LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Bf16LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Bf16DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Bf16DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: Bf16DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.output, backend)?,
            norm2: Bf16DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Bf16LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Bf16LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Bf16DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Bf16DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_style: style_to_device(&weights.input_norm, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_style: style_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn style_to_device(
    weights: &AdaRmsNormWeights,
    backend: &dyn Backend,
) -> Result<Bf16LinearWeights> {
    Bf16LinearWeights::from_host(&weights.style, backend)
}
