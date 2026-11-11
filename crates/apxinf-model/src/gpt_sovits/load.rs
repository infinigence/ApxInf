use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

use super::model::GptSovitsModel;

/// Registry entry for the GPT-SoVITS V5 family.
pub fn load(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "gpt_sovits v5 requires an FP16 converted checkpoint; use precision=auto".into(),
        ));
    }
    let device_id = match device {
        Device::Cuda(device_id) => device_id,
        other => return Err(Error::UnsupportedDevice(other)),
    };
    let model = GptSovitsModel::load_with_autotune(path, device_id, options.autotune)?;
    Ok(LoadedModel::Speech(Box::new(model)))
}
