//! Checkpoint loading and construction of the planning-only VLA.
use super::{
    backend::downcast_arc,
    config::{ProjectionLayout, QwenDriveConfig},
    model::QwenDriveModel,
    model_runner::QwenDriveModelRunner,
    weights::{
        bf16::{BackboneDeviceWeights, ExpertDeviceWeights},
        QwenDriveExpertWeights, QwenDriveVlmWeights,
    },
};
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, sync::Arc};

pub(crate) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "bf16")
    ) {
        return Err(Error::Other(
            "qwen_drive supports only model_variant=bf16 (or auto)".into(),
        ));
    }
    if options.assets.keys().any(|key| key != "planner") {
        return Err(Error::Other(
            "qwen_drive accepts only the named asset 'planner'".into(),
        ));
    }
    if options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other("qwen_drive requires its native checkpoint/config and does not support synthetic or calibrated weights".into()));
    }
    let cuda = downcast_arc(backend.clone())
        .ok_or_else(|| Error::Other("qwen_drive planning requires CUDA".into()))?;
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let planner = options
        .assets
        .get("planner")
        .cloned()
        .unwrap_or_else(|| root.join("planner-sft"));
    if !planner.exists() {
        return Err(Error::Other(format!("qwen_drive planning requires a planner checkpoint: pass assets['planner'] or provide {}",planner.display())));
    }
    let config = QwenDriveConfig::from_json_file(&root.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(root)
        .map_err(|e| Error::Other(format!("load qwen_drive backbone: {e}")))?;
    let vlm = QwenDriveVlmWeights::from_map(tensors)?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(&planner)
        .map_err(|e| Error::Other(format!("load qwen_drive planner: {e}")))?;
    let expert = QwenDriveExpertWeights::from_map(&config, &tensors)?;
    let backbone = BackboneDeviceWeights::from_maps(
        &config,
        vlm,
        ProjectionLayout::from_environment(),
        &*backend,
    )?;
    let planner = ExpertDeviceWeights::from_weights(&config, expert, &*backend)?;
    let model = QwenDriveModel::new(config, cuda, backbone, planner);
    Ok(LoadedModel::Vla(Box::new(QwenDriveModelRunner::new(model))))
}
