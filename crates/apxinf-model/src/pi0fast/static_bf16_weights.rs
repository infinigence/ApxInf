//! Fully materialized native-BF16 π0-FAST weights.

use apxinf_core::{Backend, Result, Tensor};

use super::{
    bf16_to_device, f32_to_device, Bf16LinearWeights, LanguageLayerWeights, LayerNormWeights,
    Pi0FastWeights, VisionBlockWeights,
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

/// SigLIP patch embedding kept in FP32.
///
/// LeRobot's `to_bfloat16_for_selected_params` returns exactly these three
/// tensors to FP32 after the blanket BF16 cast, and HF's `SiglipVisionEmbeddings`
/// casts the pixels up to the projection dtype before convolving. Running the
/// same convolution in BF16 instead perturbs the vision features enough to flip
/// autoregressive action tokens, so π0-FAST keeps the reference's precision here.
#[derive(Debug)]
pub struct VisionPatchEmbeddingF32 {
    /// Physical row-major `[3*patch*patch, vision_width]` projection matrix.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    /// Learned `[patches_per_view, vision_width]` position embedding.
    pub position: Tensor,
}

#[derive(Debug)]
pub struct StaticBf16Pi0FastWeights {
    pub patch_embedding: VisionPatchEmbeddingF32,
    pub vision_layers: Vec<Bf16DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Bf16LinearWeights,
    /// Tied token embedding `[vocab, width]`, used for image/language/action looks.
    pub token_embedding: Tensor,
    /// Tied LM head `[width, vocab]`, physically transposed for the output GEMM.
    pub lm_head: Bf16LinearWeights,
    pub language_layers: Vec<Bf16DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
}

impl StaticBf16Pi0FastWeights {
    pub fn from_host(weights: &Pi0FastWeights, backend: &dyn Backend) -> Result<Self> {
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
            token_embedding: bf16_to_device(&weights.lm_head, backend)?,
            lm_head: Bf16LinearWeights::from_host(
                &LinearView::transposed(&weights.lm_head)?,
                backend,
            )?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Bf16DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(
                &weights.language_final_norm_scale,
                backend,
            )?,
        })
    }
}

impl VisionPatchEmbeddingF32 {
    fn from_host(
        projection: &super::LinearWeights,
        position: &Tensor,
        backend: &dyn Backend,
    ) -> Result<Self> {
        Ok(Self {
            weight: f32_to_device(&projection.weight, backend)?,
            bias: projection
                .bias
                .as_ref()
                .map(|value| f32_to_device(value, backend))
                .transpose()?,
            position: f32_to_device(position, backend)?,
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
            qkv: Bf16LinearWeights::from_host_parts(&[&weights.q, &weights.k, &weights.v], backend)?,
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
            post_attention_norm_scale: bf16_to_device(
                &weights.post_attention_norm_scale,
                backend,
            )?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

/// Adapter that lets the LM head reuse the generic linear upload path after a
/// host transpose, keeping the tied embedding readable as `[vocab, width]`.
struct LinearView;

impl LinearView {
    fn transposed(tensor: &Tensor) -> Result<super::LinearWeights> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(apxinf_core::Error::Other(format!(
                "π0-FAST lm_head must be 2D, got {dims:?}"
            )));
        }
        let (rows, cols) = (dims[0], dims[1]);
        let src = tensor.to_f32_vec()?;
        let mut dst = vec![0.0f32; src.len()];
        for row in 0..rows {
            for col in 0..cols {
                dst[col * rows + row] = src[row * cols + col];
            }
        }
        Ok(super::LinearWeights {
            weight: Tensor::from_f32(vec![cols, rows], &dst)?,
            bias: None,
        })
    }
}
