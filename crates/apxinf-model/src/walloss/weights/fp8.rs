//! Device-resident dynamic FP8 matrices for WallOSS inference.

use apxinf_core::{Backend, Result, Tensor};

use super::bf16_to_device;
use super::device::DynamicFp8LinearWeights;
use super::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights,
    WallossWeights,
};

pub struct WallossDynamicFp8Weights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossDynamicFp8LayerWeights>,
    pub action_layers: Vec<WallossDynamicFp8LayerWeights>,
    pub action_norm: Tensor,
    pub vision: WallossDynamicFp8VisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossDynamicFp8LayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub qkv: DynamicFp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: DynamicFp8LinearWeights,
    pub gate_up: DynamicFp8LinearWeights,
    pub down: DynamicFp8LinearWeights,
}

pub struct WallossDynamicFp8VisionWeights {
    pub patch_projection: DynamicFp8LinearWeights,
    pub blocks: Vec<WallossDynamicFp8VisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: DynamicFp8LinearWeights,
    pub merger_hidden_bias: Tensor,
    pub merger_output: DynamicFp8LinearWeights,
    pub merger_output_bias: Tensor,
}

pub struct WallossDynamicFp8VisionBlockWeights {
    pub input_norm: Tensor,
    pub qkv: DynamicFp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: DynamicFp8LinearWeights,
    pub output_bias: Tensor,
    pub post_attention_norm: Tensor,
    pub gate_up: DynamicFp8LinearWeights,
    pub gate_up_bias: Tensor,
    pub down: DynamicFp8LinearWeights,
    pub down_bias: Tensor,
}

impl WallossDynamicFp8Weights {
    pub fn from_host(weights: &WallossWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            token_embedding: bf16_to_device(&weights.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| WallossDynamicFp8LayerWeights::from_host(layer, backend))
                .collect::<Result<_>>()?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| WallossDynamicFp8LayerWeights::from_host(layer, backend))
                .collect::<Result<_>>()?,
            action_norm: bf16_to_device(&weights.action_norm, backend)?,
            vision: WallossDynamicFp8VisionWeights::from_host(&weights.vision, backend)?,
            action: weights.action.to_bf16_device(backend)?,
        })
    }
}

impl WallossDynamicFp8LayerWeights {
    fn from_host(weights: &WallossLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            qkv: dynamic_fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: dynamic_fp8_matrix(&weights.output, backend)?,
            gate_up: dynamic_fp8_matrix(&weights.gate_up, backend)?,
            down: dynamic_fp8_matrix(&weights.down, backend)?,
        })
    }
}

impl WallossDynamicFp8VisionWeights {
    fn from_host(weights: &WallossVisionWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_projection: dynamic_fp8_matrix(&weights.patch_projection, backend)?,
            blocks: weights
                .blocks
                .iter()
                .map(|block| WallossDynamicFp8VisionBlockWeights::from_host(block, backend))
                .collect::<Result<_>>()?,
            merger_norm: bf16_to_device(&weights.merger_norm, backend)?,
            merger_hidden: dynamic_fp8_matrix(&weights.merger_hidden, backend)?,
            merger_hidden_bias: bf16_to_device(&weights.merger_hidden_bias, backend)?,
            merger_output: dynamic_fp8_matrix(&weights.merger_output, backend)?,
            merger_output_bias: bf16_to_device(&weights.merger_output_bias, backend)?,
        })
    }
}

impl WallossDynamicFp8VisionBlockWeights {
    fn from_host(weights: &WallossVisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            qkv: dynamic_fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: dynamic_fp8_matrix(&weights.output, backend)?,
            output_bias: bf16_to_device(&weights.output_bias, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            gate_up: dynamic_fp8_matrix(&weights.gate_up, backend)?,
            gate_up_bias: bf16_to_device(&weights.gate_up_bias, backend)?,
            down: dynamic_fp8_matrix(&weights.down, backend)?,
            down_bias: bf16_to_device(&weights.down_bias, backend)?,
        })
    }
}

fn dynamic_fp8_matrix(weight: &Tensor, backend: &dyn Backend) -> Result<DynamicFp8LinearWeights> {
    DynamicFp8LinearWeights::from_host(weight, backend)
}
