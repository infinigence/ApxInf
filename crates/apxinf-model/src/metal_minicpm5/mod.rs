//! Native MiniCPM5-2B on Apple Metal, with independent family mathematics.
mod config;
mod dspark;
mod math;
mod model;
#[cfg(test)]
mod tests;
mod weights;

pub use config::{Config, Variant, CONTEXT_CAPACITY};
pub use dspark::DSpark;
pub use model::MiniCpm5;

use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, DType, Device, Error, Result};
use apxinf_mlx::MlxBackend;
use std::{path::Path, sync::Arc};

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some_and(|d| d != DType::BF16)
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Contract(
            "MiniCPM5 MLX requires native checkpoint BF16 and no CUDA tuning/calibration options",
        ));
    }
    let variant = Variant::parse(options.model_variant.as_deref())?;
    if variant != Variant::DSpark && !options.assets.is_empty() {
        return Err(Error::Contract(
            "MiniCPM5 BF16 variants do not accept auxiliary assets",
        ));
    }
    let draft = if variant == Variant::DSpark {
        if options.assets.len() != 1 {
            return Err(Error::Contract(
                "MiniCPM5 DSpark requires exactly one draft asset",
            ));
        }
        Some(
            options
                .assets
                .get("draft")
                .ok_or(Error::Contract("missing MiniCPM5 DSpark draft asset"))?,
        )
    } else {
        None
    };
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract("MiniCPM5 requires the native MLX backend"))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    let directory = if path.is_dir() {
        path
    } else {
        path.parent()
            .ok_or(Error::Contract("checkpoint file has no parent"))?
    };
    let config = Config::from_json(&std::fs::read_to_string(directory.join("config.json"))?)?;
    let (map, _) = apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
    let weights = weights::Weights::load(&config, native.stream(), map)?;
    let target = MiniCpm5::new(config, weights, native, variant)?;
    if let Some(draft) = draft {
        Ok(LoadedModel::text(Box::new(DSpark::load(target, draft)?)))
    } else {
        Ok(LoadedModel::text(Box::new(target)))
    }
}
