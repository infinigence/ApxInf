//! Registry entry for the qwen3-asr family.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

use super::model::Qwen3AsrModel;

/// Registry entry for the qwen3-asr family.
pub fn load(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "qwen3-asr requires the BF16 converted checkpoint; use precision=auto".into(),
        ));
    }
    let device_id = match device {
        Device::Cuda(device_id) => device_id,
        other => return Err(Error::UnsupportedDevice(other)),
    };
    let recipe = crate::auto::cuda_recipe_options(options)?;
    let model = Qwen3AsrModel::load_with_recipe_options(
        path,
        device_id,
        recipe.online_tune,
        recipe.cache_dir,
    )?;
    Ok(LoadedModel::Asr(Box::new(model)))
}
