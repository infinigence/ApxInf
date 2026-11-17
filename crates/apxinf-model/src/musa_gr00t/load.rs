//! Load the MUSA GR00T configuration and establish the backend-owned runner.

use std::path::{Path, PathBuf};

use apxinf_core::{Device, Error, Result};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

use super::assets;
use super::model::precision;
use super::weights::Gr00tWeights;
use super::{backend, Gr00tConfig, Gr00tModelRunner};

const BACKBONE_ASSET: &str = "backbone";

pub(crate) fn load(path: &Path, device: Device, options: &LoadOptions) -> Result<LoadedModel> {
    let Device::Musa(_) = device else {
        return Err(Error::Other(format!("GR00T requires MUSA, not {device}")));
    };
    validate_options(options)?;
    let root = artifact_root(path);
    if let Some(name) = options
        .assets
        .keys()
        .find(|name| name.as_str() != BACKBONE_ASSET)
    {
        return Err(Error::Other(format!(
            "GR00T does not recognize asset {name:?}; expected only {BACKBONE_ASSET:?}"
        )));
    }
    let backbone_path = assets::resolve_assets(
        root,
        options
            .assets
            .get(BACKBONE_ASSET)
            .map(|path| path.as_path()),
    )?;
    let config = Gr00tConfig::from_json_file(&root.join("config.json"))?;
    let (backbone_config, weights) = Gr00tWeights::from_safetensors(&config, &backbone_path, root)?;
    let backend = backend::create(device)?;
    install_tactics(&backend, root, options)?;
    if options.precision == ModelPrecision::Fp8 {
        let caps = backend.context().caps();
        if apxinf_musa::MUSA_ARCH != "mp_31"
            || (caps.compute_major, caps.compute_minor) != (3, 1)
        {
            return Err(Error::Other("MUSA GR00T FP8 requires MP31".into()));
        }
        let calibration = options.calibration_path.as_deref().ok_or_else(|| {
            Error::Other("GR00T FP8 requires a calibration file".into())
        })?;
        let execution = super::model::fp8::Gr00tFp8Execution::from_file(
            calibration, root, &backbone_path, &config, &backbone_config,
        )?;
        let executor = super::model::executor::Gr00tExecutor::from_backend(
            config, backbone_config, weights, execution, backend,
        )?;
        let runner = Gr00tModelRunner::new(executor);
        let runner = match options.execution_policy {
            Some(policy) => runner.with_execution_policy(policy),
            None => runner,
        };
        return Ok(LoadedModel::Vla(Box::new(runner)));
    }
    let executor = precision::build(config, backbone_config, weights, backend)?;
    let runner = Gr00tModelRunner::new(executor);
    let runner = match options.execution_policy {
        Some(policy) => runner.with_execution_policy(policy),
        None => runner,
    };
    Ok(LoadedModel::Vla(Box::new(runner)))
}

fn select_tactics(default: &Path, legacy: &Path, options: &LoadOptions) -> Option<PathBuf> {
    match options.tuning_path.as_ref() {
        Some(path) if path.is_file() => Some(path.clone()),
        Some(path) if options.autotune && !path.exists() => None,
        Some(path) => Some(path.clone()),
        None => default.is_file().then(|| default.to_path_buf())
            .or_else(|| legacy.is_file().then(|| legacy.to_path_buf())),
    }
}

fn install_tactics(runtime: &backend::RuntimeBackend, root: &Path, options: &LoadOptions) -> Result<()> {
    use backend::tuning::{configure_tuning, TuningDb, TuningMode, TuningPaths};

    let context = runtime.context();
    let defaults = TuningPaths::resolve_for_device(
        "configs/tuning", context.caps(), context.library_versions(),
    );
    let selected = select_tactics(&defaults.tactics, &root.join("tactics.json"), options);
    let database = selected.as_deref().map(TuningDb::from_json_file).transpose()?;
    let paths = options.tuning_path.clone().map(TuningPaths::from_tactics).unwrap_or(defaults);
    let mode = if options.autotune { TuningMode::AutoTune } else { TuningMode::Inference };
    configure_tuning(context, mode, database.as_slice(), Some(paths))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn validate_options(options: &LoadOptions) -> Result<()> {
    if options.synthetic.is_some()
        || options.config.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other(
            "GR00T does not support PI0.5 synthetic/config load options".into(),
        ));
    }
    match options.precision {
        ModelPrecision::Auto | ModelPrecision::Bf16 => Ok(()),
        ModelPrecision::Fp8 if options.calibration_path.is_some() => Ok(()),
        ModelPrecision::Fp8 => Err(Error::Other("GR00T FP8 requires a calibration file".into())),
        precision => Err(Error::Other(format!(
            "MUSA GR00T supports BF16 and calibrated MP31 FP8, not {precision:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_missing_tactics_are_only_created_by_autotune() {
        let directory = std::env::temp_dir().join(format!("gr00t-tactics-{}", std::process::id()));
        let path = directory.join("missing.json");
        let mut options = LoadOptions { tuning_path: Some(path.clone()), ..LoadOptions::default() };
        assert_eq!(select_tactics(&directory.join("default"), &directory.join("legacy"), &options), Some(path));
        options.autotune = true;
        assert_eq!(select_tactics(&directory.join("default"), &directory.join("legacy"), &options), None);
    }

    #[test]
    fn accepts_auto_and_bf16_precision() {
        for precision in [ModelPrecision::Auto, ModelPrecision::Bf16] {
            let options = LoadOptions {
                precision,
                ..LoadOptions::default()
            };
            validate_options(&options).unwrap();
        }
    }

    #[test]
    fn fp8_requires_explicit_calibration() {
        let mut options = LoadOptions { precision: ModelPrecision::Fp8, ..LoadOptions::default() };
        assert!(validate_options(&options).is_err());
        options.calibration_path = Some("calibration.json".into());
        validate_options(&options).unwrap();
    }

    #[test]
    fn rejects_unimplemented_precisions() {
        for precision in [ModelPrecision::Fp8, ModelPrecision::W8A8] {
            let options = LoadOptions {
                precision,
                ..LoadOptions::default()
            };
            assert!(validate_options(&options).is_err());
        }
    }
}
