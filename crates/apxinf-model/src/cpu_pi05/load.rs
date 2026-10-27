//! CPU asset selection and assembly; no CUDA context is created.
use super::config::Config;
use super::model::CpuPi05Model;
use super::runner::CpuPi05Runner;
use super::weights::Weights;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Error, Result};
use std::path::Path;
use std::rc::Rc;

pub(crate) fn load(
    path: &Path,
    options: &LoadOptions,
    config: Option<Config>,
) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "PI0.5 CPU uses model_variant=fp32_reference instead of precision".into(),
        ));
    }
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto") | Some("fp32_reference")
    ) {
        return Err(Error::Other(
            "PI0.5 CPU supports only auto or fp32_reference".into(),
        ));
    }
    if options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.tuning_path.is_some()
        || options.autotune
    {
        return Err(Error::Other(
            "PI0.5 CPU reference does not support CUDA tuning or quantization calibration".into(),
        ));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let mut config = if let Some(c) = config {
        c
    } else if root.join("config.json").is_file() {
        Config::from_json_file(&root.join("config.json"))?
    } else {
        Config::default()
    };
    if let Some(views) = options.num_views {
        if views == 0 || views > config.num_views {
            return Err(Error::Other(
                "PI0.5 CPU num_views must be within the configured view count".into(),
            ));
        }
        config.num_views = views;
    }
    config.validate()?;
    let weights = match options.synthetic {
        Some(s) => Weights::synthetic(&config, s.seed)?,
        None => Weights::from_safetensors(&config, path)?,
    };
    let model = Rc::new(CpuPi05Model::new(config.clone(), weights)?);
    Ok(LoadedModel::Vla(Box::new(CpuPi05Runner::new(
        config, model,
    )?)))
}
