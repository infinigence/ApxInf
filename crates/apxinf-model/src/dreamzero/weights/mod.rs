mod text;
pub(super) use text::{TextBlockWeights, TextWeights};

pub(super) mod clip;

pub(super) mod dit;
mod unused_vae;
pub(super) mod vae;

use super::backend::Context;
use apxinf_core::{DType, Error, Result};
use std::path::Path;

pub(super) struct DreamZeroWeights {
    pub text: TextWeights,
    pub clip: clip::ClipWeights,
    pub vae: vae::VaeWeights,
    pub dit: dit::DitWeights,
}
impl DreamZeroWeights {
    pub fn from_safetensors(ctx: &Context, path: &Path) -> Result<Self> {
        let (mut map, _) =
            apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
        // Validate unused image-decoder tensors as part of the checkpoint
        // contract, then release their host storage before device materialization.
        for (name, shape) in unused_vae::DECODER {
            let key = format!("action_head.vae.model.{name}");
            let tensor = map
                .remove(&key)
                .ok_or_else(|| Error::Other(format!("missing DreamZero {key}")))?;
            if tensor.dtype() != DType::BF16 || tensor.shape().dims() != *shape {
                return Err(Error::Other(format!(
                    "DreamZero unused decoder {key} must be BF16 {shape:?}"
                )));
            }
        }
        let weights = Self {
            text: TextWeights::from_map(ctx, &mut map)?,
            clip: clip::ClipWeights::from_map(ctx, &mut map)?,
            vae: vae::VaeWeights::from_map(ctx, &mut map)?,
            dit: dit::DitWeights::from_map(ctx, &mut map)?,
        };
        if !map.is_empty() {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            return Err(Error::Other(format!(
                "unexpected DreamZero checkpoint tensors: {keys:?}"
            )));
        }
        Ok(weights)
    }
}
