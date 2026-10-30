//! Load WallOSS assets and assemble the computation and model_runner modules.
//!
//! Loading parses the checkpoint config, materializes host weights, selects
//! the precision implementation, uploads device weights and the fixed
//! time-step schedule, and constructs the runner through its constructor.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Device, Error, Result};

use super::backend::RuntimeBackend;
use super::model::{ModelVariant, WallossSchedule};
use super::model_runner::WallossModelRunner;
use super::weights::{WallossDynamicFp8Weights, WallossWeights};
use super::{WallossConfig, WallossImageProcessorConfig};
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

pub(crate) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn apxinf_core::Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.config.is_some() {
        return Err(Error::Other(
            "walloss does not support action_horizon/num_views config overrides".into(),
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16 | ModelPrecision::Fp8
    ) {
        return Err(Error::Other("walloss supports BF16 and FP8 on CUDA".into()));
    }
    if options.calibration_path.is_some() || options.uniform_fp8_scale.is_some() {
        return Err(Error::Other(
            "walloss FP8 uses dynamic rowwise activation scales and does not support calibration"
                .into(),
        ));
    }
    let backend = crate::accelerator::downcast_cuda_new_arc(backend)
        .ok_or_else(|| Error::Other("walloss is only registered for CUDA".into()))?;
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other("walloss is only registered for CUDA".into()));
    }
    Ok(LoadedModel::Vla(Box::new(load_model_runner(
        path, backend, options,
    )?)))
}

pub(in crate::walloss) fn load_model_runner(
    path: &Path,
    backend: Arc<RuntimeBackend>,
    options: &LoadOptions,
) -> Result<WallossModelRunner> {
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let mut config = WallossConfig::from_json_file(&root.join("config.json"))?;
    let image_processor_path = root.join("preprocessor_config.json");
    let image_processor = image_processor_path
        .is_file()
        .then(|| WallossImageProcessorConfig::from_json_file(&image_processor_path, &config.vision))
        .transpose()?
        .map(Arc::new);
    let host_weights = WallossWeights::from_safetensors(&mut config, path)?;
    let config = Arc::new(config);
    let schedule = Arc::new(WallossSchedule::new(&config, &backend)?);
    let model = match options.precision {
        ModelPrecision::Fp8 => ModelVariant::new_dynamic_fp8(
            Arc::clone(&backend),
            Arc::clone(&config),
            image_processor.clone(),
            WallossDynamicFp8Weights::from_host(&host_weights, &*backend)?,
        ),
        ModelPrecision::Auto | ModelPrecision::Bf16 => ModelVariant::new_bf16(
            Arc::clone(&backend),
            Arc::clone(&config),
            image_processor.clone(),
            host_weights.to_bf16_device(&*backend)?,
        ),
        ModelPrecision::W8A8 => unreachable!(),
    };
    eprintln!("[apxinf] WallOSS model_variant={}", model.name());
    Ok(WallossModelRunner::new(backend, config, schedule, model))
}
