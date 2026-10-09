//! Dense Qwen3 native MLX implementation. No Python model execution.

pub mod config;
mod model;
mod weights;
pub use model::Qwen3Model;

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, DType, Device, Error, Result};
use config::{Qwen3Config, Variant};
use std::{path::Path, sync::Arc};

/// Registry loader; construction retains the exact backend stream selected by AutoModel.
pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) || backend.device() != device {
        return Err(Error::UnsupportedDevice(device));
    }
    if options.synthetic.is_some()
        || options.config.is_some()
        || !options.assets.is_empty()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.autotune
        || !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        )
        || options
            .text_weight_dtype
            .is_some_and(|dtype| dtype != DType::BF16)
    {
        return Err(Error::Other("Qwen3 MLX requires the original BF16 checkpoint and its family-local variant; CUDA calibration/tuning and synthetic options are unsupported".into()));
    }
    let variant = Variant::parse(options.model_variant.as_deref())?;
    let json = std::fs::read_to_string(path.join("config.json"))
        .map_err(|e| Error::Other(format!("cannot read Qwen3 config: {e}")))?;
    let config = Qwen3Config::from_json(&json)?;
    config.validate_checkpoint_scope()?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
    Ok(LoadedModel::text(Box::new(Qwen3Model::new(
        config, tensors, backend, variant,
    )?)))
}
