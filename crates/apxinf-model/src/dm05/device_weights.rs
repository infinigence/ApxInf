//! Device-resident BF16 DM05 weights without speculative packing.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::weights::{
    ActionLayerWeights, Dm05Weights, GemmaAttentionWeights, GemmaMlpWeights, GemmaRmsWeights,
    LanguageLayerWeights, LayerNormWeights, LinearWeights, VisionBlockWeights,
};
use super::Dm05LayerType;

#[derive(Debug)]
pub struct DeviceLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

#[derive(Debug)]
pub struct DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct DeviceGemmaRms {
    pub raw_weight: Tensor,
}

#[derive(Debug)]
pub struct DeviceAttention {
    pub q: DeviceLinear,
    pub k: DeviceLinear,
    pub v: DeviceLinear,
    pub output: DeviceLinear,
    pub q_norm: DeviceGemmaRms,
    pub k_norm: DeviceGemmaRms,
}

#[derive(Debug)]
pub struct DeviceMlp {
    pub gate: DeviceLinear,
    pub up: DeviceLinear,
    pub down: DeviceLinear,
}

#[derive(Debug)]
pub struct DeviceVisionBlock {
    pub norm1: DeviceLayerNorm,
    pub q: DeviceLinear,
    pub k: DeviceLinear,
    pub v: DeviceLinear,
    pub output: DeviceLinear,
    pub norm2: DeviceLayerNorm,
    pub fc1: DeviceLinear,
    pub fc2: DeviceLinear,
}

#[derive(Debug)]
pub struct DeviceLanguageLayer {
    pub layer_type: Dm05LayerType,
    pub input_norm: DeviceGemmaRms,
    pub attention: DeviceAttention,
    pub post_attention_norm: DeviceGemmaRms,
    pub pre_feedforward_norm: DeviceGemmaRms,
    pub mlp: DeviceMlp,
    pub post_feedforward_norm: DeviceGemmaRms,
}

#[derive(Debug)]
pub struct DeviceActionLayer {
    pub layer_type: Dm05LayerType,
    pub input_modulator: DeviceLinear,
    pub attention: DeviceAttention,
    pub post_attention_norm: DeviceGemmaRms,
    pub mlp_modulator: DeviceLinear,
    pub mlp: DeviceMlp,
    pub post_feedforward_norm: DeviceGemmaRms,
}

#[derive(Debug)]
pub struct DeviceDm05Weights {
    pub patch_embedding: DeviceLinear,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<DeviceVisionBlock>,
    pub vision_post_norm: DeviceLayerNorm,
    pub projector_norm: DeviceGemmaRms,
    pub projector: Tensor,
    pub token_embedding: Tensor,
    pub language_layers: Vec<DeviceLanguageLayer>,
    pub language_final_norm: DeviceGemmaRms,
    pub action_layers: Vec<DeviceActionLayer>,
    pub action_final_modulator: DeviceLinear,
    pub action_in: DeviceLinear,
    pub action_out: DeviceLinear,
    pub time_mlp_in: DeviceLinear,
    pub time_mlp_out: DeviceLinear,
}

impl DeviceDm05Weights {
    pub fn from_host(weights: &Dm05Weights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_embedding: DeviceLinear::from_host(&weights.vision.patch_embedding, backend)?,
            position_embedding: upload(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: DeviceLayerNorm::from_host(&weights.vision.post_layer_norm, backend)?,
            projector_norm: DeviceGemmaRms::from_host(&weights.vision.projector_norm, backend)?,
            projector: upload(&weights.vision.projector, backend)?,
            token_embedding: upload(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm: DeviceGemmaRms::from_host(&weights.language_final_norm, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_modulator: DeviceLinear::from_host(
                &weights.action_final_modulator,
                backend,
            )?,
            action_in: DeviceLinear::from_host(&weights.action_in, backend)?,
            action_out: DeviceLinear::from_host(&weights.action_out, backend)?,
            time_mlp_in: DeviceLinear::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: DeviceLinear::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl DeviceLinear {
    fn from_host(weights: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: upload(&weights.weight, backend)?,
            bias: weights
                .bias
                .as_ref()
                .map(|bias| upload(bias, backend))
                .transpose()?,
        })
    }
}

impl DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            weight: upload(&weights.weight, backend)?,
            bias: upload(&weights.bias, backend)?,
        })
    }
}

impl DeviceGemmaRms {
    fn from_host(weights: &GemmaRmsWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            raw_weight: upload(&weights.raw_weight, backend)?,
        })
    }
}

impl DeviceAttention {
    fn from_host(weights: &GemmaAttentionWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            q: DeviceLinear::from_host(&weights.q, backend)?,
            k: DeviceLinear::from_host(&weights.k, backend)?,
            v: DeviceLinear::from_host(&weights.v, backend)?,
            output: DeviceLinear::from_host(&weights.output, backend)?,
            q_norm: DeviceGemmaRms::from_host(&weights.q_norm, backend)?,
            k_norm: DeviceGemmaRms::from_host(&weights.k_norm, backend)?,
        })
    }
}

impl DeviceMlp {
    fn from_host(weights: &GemmaMlpWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            gate: DeviceLinear::from_host(&weights.gate, backend)?,
            up: DeviceLinear::from_host(&weights.up, backend)?,
            down: DeviceLinear::from_host(&weights.down, backend)?,
        })
    }
}

impl DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            q: DeviceLinear::from_host(&weights.q, backend)?,
            k: DeviceLinear::from_host(&weights.k, backend)?,
            v: DeviceLinear::from_host(&weights.v, backend)?,
            output: DeviceLinear::from_host(&weights.output, backend)?,
            norm2: DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: DeviceLinear::from_host(&weights.fc1, backend)?,
            fc2: DeviceLinear::from_host(&weights.fc2, backend)?,
        })
    }
}

impl DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            layer_type: weights.layer_type,
            input_norm: DeviceGemmaRms::from_host(&weights.input_norm, backend)?,
            attention: DeviceAttention::from_host(&weights.attention, backend)?,
            post_attention_norm: DeviceGemmaRms::from_host(&weights.post_attention_norm, backend)?,
            pre_feedforward_norm: DeviceGemmaRms::from_host(
                &weights.pre_feedforward_norm,
                backend,
            )?,
            mlp: DeviceMlp::from_host(&weights.mlp, backend)?,
            post_feedforward_norm: DeviceGemmaRms::from_host(
                &weights.post_feedforward_norm,
                backend,
            )?,
        })
    }
}

impl DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            layer_type: weights.layer_type,
            input_modulator: DeviceLinear::from_host(&weights.input_modulator, backend)?,
            attention: DeviceAttention::from_host(&weights.attention, backend)?,
            post_attention_norm: DeviceGemmaRms::from_host(&weights.post_attention_norm, backend)?,
            mlp_modulator: DeviceLinear::from_host(&weights.mlp_modulator, backend)?,
            mlp: DeviceMlp::from_host(&weights.mlp, backend)?,
            post_feedforward_norm: DeviceGemmaRms::from_host(
                &weights.post_feedforward_norm,
                backend,
            )?,
        })
    }
}

fn upload(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "DM05 device upload requires BF16, got {}",
            tensor.dtype()
        )));
    }
    backend.to_device(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;
    use half::bf16;

    #[test]
    fn device_linear_preserves_bf16_shape_and_optional_bias() {
        let host = LinearWeights {
            weight: Tensor::from_bf16(vec![2, 3], &[bf16::ONE; 6]).unwrap(),
            bias: Some(Tensor::from_bf16(vec![3], &[bf16::ZERO; 3]).unwrap()),
        };
        let device = DeviceLinear::from_host(&host, &CpuBackend).unwrap();
        assert_eq!(device.weight.shape().dims(), [2, 3]);
        assert_eq!(device.weight.dtype(), DType::BF16);
        assert_eq!(device.bias.unwrap().shape().dims(), [3]);
    }
}
