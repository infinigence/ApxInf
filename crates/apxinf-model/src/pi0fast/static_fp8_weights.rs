//! Fully materialized FP8 π0-FAST weights: E4M3 matrices, BF16 everything else.
//!
//! Only the linear projections are quantized. Norms, the FP32 SigLIP patch
//! embedding and the tied token-embedding table keep the BF16 module's dtypes,
//! because `gemm::fp8_bf16` returns BF16 and the surrounding pipeline is
//! therefore reused unchanged.

use apxinf_core::{Backend, Result, Tensor};

use super::{
    bf16_to_device, Fp8LinearWeights, LanguageLayerWeights, Pi0FastWeights, VisionBlockWeights,
};
use super::static_bf16_weights::{
    Bf16DeviceLayerNorm, LinearView, VisionPatchEmbeddingF32,
};

#[derive(Debug)]
pub struct Fp8DeviceVisionBlock {
    pub norm1: Bf16DeviceLayerNorm,
    pub qkv: Fp8LinearWeights,
    pub output: Fp8LinearWeights,
    pub norm2: Bf16DeviceLayerNorm,
    pub fc1: Fp8LinearWeights,
    pub fc2: Fp8LinearWeights,
}

#[derive(Debug)]
pub struct Fp8DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Fp8LinearWeights,
    pub output: Fp8LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Fp8LinearWeights,
    pub down: Fp8LinearWeights,
}

#[derive(Debug)]
pub struct StaticFp8Pi0FastWeights {
    pub patch_embedding: VisionPatchEmbeddingF32,
    pub vision_layers: Vec<Fp8DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Fp8LinearWeights,
    /// Tied token embedding `[vocab, width]`, kept BF16 because a lookup reads
    /// one 2048-wide row per decode step — quantizing it would buy no bandwidth
    /// and would need a new E4M3 gather kernel.
    pub token_embedding: Tensor,
    /// Tied LM head `[width, columns]`, E4M3, transposed for the GEMM and pruned
    /// to `Pi0FastConfig::action_head_columns`.
    pub lm_head: Fp8LinearWeights,
    pub language_layers: Vec<Fp8DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
}

impl StaticFp8Pi0FastWeights {
    pub fn from_host(
        weights: &Pi0FastWeights,
        config: &super::Pi0FastConfig,
        backend: &dyn Backend,
    ) -> Result<Self> {
        Ok(Self {
            patch_embedding: VisionPatchEmbeddingF32::from_host(
                &weights.vision.patch_embedding,
                &weights.vision.position_embedding,
                backend,
            )?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Fp8DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Bf16DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Fp8LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.lm_head, backend)?,
            lm_head: Fp8LinearWeights::from_host(
                &LinearView::columns(&weights.lm_head, &config.action_head_columns()?)?,
                backend,
            )?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Fp8DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(
                &weights.language_final_norm_scale,
                backend,
            )?,
        })
    }
}

impl Fp8DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            norm1: Bf16DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Fp8LinearWeights::from_host_parts(&[&weights.q, &weights.k, &weights.v], backend)?,
            output: Fp8LinearWeights::from_host(&weights.output, backend)?,
            norm2: Bf16DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Fp8LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Fp8LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Fp8DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Fp8LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(
                &weights.post_attention_norm_scale,
                backend,
            )?,
            gate_up: Fp8LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp8LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}
