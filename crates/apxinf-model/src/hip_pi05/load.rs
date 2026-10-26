//! Load a π0.5 checkpoint onto a HIP device.

use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Device, Error, Result};
use apxinf_hip::HipContext;

use super::checkpoint::Checkpoint;
use super::config::Config;
use super::model::HipPi05Model;
use super::runner::HipPi05Runner;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

pub(crate) fn load(
    path: &Path,
    device: Device,
    options: &LoadOptions,
    config: Option<Config>,
) -> Result<LoadedModel> {
    let Device::Hip(device_id) = device else {
        return Err(Error::Other("π0.5 HIP loader requires a HIP device".into()));
    };
    let ctx = Arc::new(HipContext::new(device_id)?);
    Ok(LoadedModel::Vla(Box::new(load_runner(
        path, ctx, options, config,
    )?)))
}

/// `config` is the caller's when given — AutoModel converts a frontend's
/// `LoadOptions::config` into it — else the checkpoint's `config.json`, else
/// the default. `LoadOptions::config` itself is not read here: it is another
/// family's type. Weights are synthetic when requested, else the SafeTensors
/// at `path`.
pub fn load_runner(
    path: &Path,
    ctx: Arc<HipContext>,
    options: &LoadOptions,
    config: Option<Config>,
) -> Result<HipPi05Runner> {
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "π0.5 uses model_variant instead of precision".into(),
        ));
    }
    match options.model_variant.as_deref().unwrap_or("auto") {
        "auto" | "bf16" => {}
        other => {
            return Err(Error::Other(format!(
                "π0.5 on HIP supports model_variant auto or bf16, got {other}"
            )))
        }
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config_path = root.join("config.json");
    let config = Arc::new(match config {
        Some(config) => {
            config.validate()?;
            config
        }
        None if config_path.is_file() => Config::from_json_file(&config_path)?,
        None => Config::default(),
    });
    let checkpoint = match options.synthetic {
        Some(synthetic) => Checkpoint::synthetic(&config, synthetic.seed)?,
        None => Checkpoint::from_safetensors(&config, path)?,
    };
    eprintln!("[apxinf] PI0.5 model_variant=bf16 (hip)");
    let model = HipPi05Model::new(ctx, config, &checkpoint)?;
    Ok(HipPi05Runner::new(Rc::new(model)))
}
