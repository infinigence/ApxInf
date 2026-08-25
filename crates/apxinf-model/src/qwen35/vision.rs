use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use apxinf_loader::safetensors;

use crate::accelerator::create_backend;
use crate::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
use crate::qwen3vl::vision;
use crate::qwen3vl::vision_weights::{transfer_vision_weights, Qwen3VLVisionWeights};

use super::Qwen35Config;

/// Native Qwen3.5/Qwen3.8 visual encoder and primary merger.
///
/// The hybrid language runtime remains the sole owner of image-token
/// embedding injection, KV positions, and multimodal RoPE.
pub struct Qwen35VisionEncoder {
    config: Qwen3VLConfig,
    weights: Qwen3VLVisionWeights,
    backend: Arc<dyn Backend>,
}

impl Qwen35VisionEncoder {
    pub fn load(model_dir: &Path, config: &Qwen35Config) -> Result<Self> {
        if !config.vision.deepstack_visual_indexes.is_empty() {
            return Err(Error::Other(
                "Qwen3.5 vision encoder does not support deepstack outputs".into(),
            ));
        }
        let vision_config = qwen3vl_config(config);
        let (tensors, _) = safetensors::load_native_path(model_dir)
            .map_err(|error| Error::Other(format!("load Qwen3.5 vision tensors: {error}")))?;
        let weights = Qwen3VLVisionWeights::from_map(&vision_config, tensors)?;
        let backend = create_backend(Device::Cuda(0))?;
        let weights = transfer_vision_weights(&weights, &*backend)?;
        Ok(Self {
            config: vision_config,
            weights,
            backend,
        })
    }

    pub fn encode_cpu(&self, pixel_values: &Tensor, grid_thw: [u32; 3]) -> Result<Tensor> {
        self.encode_impl(pixel_values, grid_thw, None)
    }

    pub fn encode_cpu_debug(
        &self,
        pixel_values: &Tensor,
        grid_thw: [u32; 3],
        dump_prefix: &str,
    ) -> Result<Tensor> {
        self.encode_impl(pixel_values, grid_thw, Some(dump_prefix))
    }

    fn encode_impl(
        &self,
        pixel_values: &Tensor,
        grid_thw: [u32; 3],
        dump_prefix: Option<&str>,
    ) -> Result<Tensor> {
        let patch_width = self
            .config
            .vision
            .in_channels
            .checked_mul(self.config.vision.temporal_patch_size)
            .and_then(|value| value.checked_mul(self.config.vision.patch_size))
            .and_then(|value| value.checked_mul(self.config.vision.patch_size))
            .ok_or_else(|| Error::Other("Qwen3.5 vision patch width overflow".into()))?;
        let patch_count = grid_thw
            .iter()
            .try_fold(1usize, |product, value| {
                product.checked_mul(*value as usize)
            })
            .ok_or_else(|| Error::Other("Qwen3.5 vision patch count overflow".into()))?;
        if pixel_values.device() != Device::Cpu
            || pixel_values.dtype() != DType::BF16
            || pixel_values.shape().dims() != [patch_count, patch_width]
        {
            return Err(Error::Other(format!(
                "Qwen3.5 pixel_values must be CPU BF16 [{patch_count},{patch_width}], got {} {:?} on {}",
                pixel_values.dtype(),
                pixel_values.shape().dims(),
                pixel_values.device(),
            )));
        }
        let device_pixels = self.backend.to_device(pixel_values)?;
        let output = match dump_prefix {
            Some(prefix) => vision::forward_debug(
                &self.config,
                &self.weights,
                &*self.backend,
                &device_pixels,
                &[grid_thw],
                prefix,
            )?,
            None => vision::forward(
                &self.config,
                &self.weights,
                &*self.backend,
                &device_pixels,
                &[grid_thw],
            )?,
        };
        if !output.deepstack.is_empty() {
            return Err(Error::Other(
                "Qwen3.5 vision encoder produced unexpected deepstack outputs".into(),
            ));
        }
        let primary = self.backend.to_cpu(&output.primary)?;
        let expected_rows = patch_count / self.config.vision.spatial_merge_size.pow(2);
        if primary.dtype() != DType::BF16
            || primary.shape().dims() != [expected_rows, self.config.vision.out_hidden_size]
        {
            return Err(Error::Other(format!(
                "Qwen3.5 vision primary must be BF16 [{expected_rows},{}], got {} {:?}",
                self.config.vision.out_hidden_size,
                primary.dtype(),
                primary.shape().dims(),
            )));
        }
        Ok(primary)
    }
}

fn qwen3vl_config(config: &Qwen35Config) -> Qwen3VLConfig {
    let text = &config.text;
    let vision = &config.vision;
    Qwen3VLConfig {
        text: Qwen3VLTextConfig {
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            n_layers: text.n_layers,
            n_heads: text.n_heads,
            n_kv_heads: text.n_kv_heads,
            head_dim: text.head_dim,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            mrope_section: text.mrope_section,
            mrope_interleaved: text.mrope_interleaved,
            tie_word_embeddings: false,
        },
        vision: Qwen3VLVisionConfig {
            depth: vision.depth,
            hidden_size: vision.hidden_size,
            intermediate_size: vision.intermediate_size,
            num_heads: vision.num_heads,
            head_dim: vision.hidden_size / vision.num_heads,
            patch_size: vision.patch_size,
            temporal_patch_size: vision.temporal_patch_size,
            in_channels: vision.in_channels,
            spatial_merge_size: vision.spatial_merge_size,
            num_position_embeddings: vision.num_position_embeddings,
            out_hidden_size: vision.out_hidden_size,
            deepstack_visual_indexes: vision.deepstack_visual_indexes.clone(),
        },
        image_token_id: config.image_token_id,
        video_token_id: config.video_token_id,
        vision_start_token_id: config.vision_start_token_id,
        vision_end_token_id: config.vision_end_token_id,
    }
}
