use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::{Backend, Device, Result};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

use super::{SmolVlaConfig, SmolVlaModel, SmolVlaModelRunner, SmolVlaWeights};

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto {
        return Err(apxinf_core::Error::Other(
            "SmolVLA uses model_variant instead of precision".into(),
        ));
    }
    match options.model_variant.as_deref() {
        None | Some("bf16") | Some("fp16") => {}
        Some(variant) => {
            return Err(apxinf_core::Error::Other(format!(
                "SmolVLA supports model_variant bf16 or fp16, got {variant}"
            )))
        }
    }
    let backend = super::model::downcast_backend(backend)?;
    let model_path = resolve_model_path(path)?;
    let config_path = resolve_config_path(path)?;
    let config = Arc::new(SmolVlaConfig::from_json_file(&config_path)?);
    if let Some(views) = options.num_views {
        if views == 0 || views > config.num_views {
            return Err(apxinf_core::Error::Other(format!(
                "SmolVLA num_views={views} must be in 1..={}",
                config.num_views
            )));
        }
        let mut config = Arc::try_unwrap(config)
            .map_err(|_| apxinf_core::Error::Other("SmolVLA config is shared".into()))?;
        config.num_views = views;
        config.validate()?;
        let config = Arc::new(config);
        let host_weights = SmolVlaWeights::from_safetensors(&config, &model_path)?;
        let weights = Arc::new(host_weights.upload(
            &*backend,
            &config,
            matches!(options.model_variant.as_deref(), Some("fp16")),
        )?);
        let model = Arc::new(SmolVlaModel::new(
            Arc::clone(&backend),
            Arc::clone(&config),
            weights,
        )?);
        return Ok(LoadedModel::Vla(Box::new(SmolVlaModelRunner::new(
            model, config,
        ))));
    }
    let host_weights = SmolVlaWeights::from_safetensors(&config, &model_path)?;
    let weights = Arc::new(host_weights.upload(
        &*backend,
        &config,
        matches!(options.model_variant.as_deref(), Some("fp16")),
    )?);
    let model = Arc::new(SmolVlaModel::new(
        Arc::clone(&backend),
        Arc::clone(&config),
        weights,
    )?);
    Ok(LoadedModel::Vla(Box::new(SmolVlaModelRunner::new(
        model, config,
    ))))
}

fn resolve_model_path(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let candidates = [
        path.join("model.safetensors"),
        path.join("smolvla_libero_model.safetensors"),
        path.join("smolvla_libero").join("model.safetensors"),
    ];
    let candidate = candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            apxinf_core::Error::Other(format!(
                "SmolVLA checkpoint not found under {}",
                path.display()
            ))
        })?;
    Ok(candidate)
}

fn resolve_config_path(path: &Path) -> Result<PathBuf> {
    let mut roots = if path.is_file() {
        let parent = path.parent().unwrap_or(path);
        let mut roots = vec![parent.to_path_buf()];
        if let Some(grandparent) = parent.parent() {
            roots.push(grandparent.to_path_buf());
        }
        roots
    } else {
        vec![path.to_path_buf(), path.join("smolvla_libero")]
    };
    if path.is_file() {
        if let Some(parent) = path.parent() {
            roots.push(parent.join("smolvla_libero"));
        }
    }
    let candidate = roots
        .into_iter()
        .find_map(|root| {
            let candidate = root.join("config.json");
            candidate.is_file().then_some(candidate)
        })
        .ok_or_else(|| {
            apxinf_core::Error::Other(format!(
                "SmolVLA config.json not found near {}",
                path.display()
            ))
        })?;
    Ok(candidate)
}
