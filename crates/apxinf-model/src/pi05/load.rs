//! Load PI0.5 assets and assemble the computation and execution modules.
use super::network::LoadedCompute;
use super::*;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Result};
use apxinf_core::{Device, Error};
use std::path::{Path, PathBuf};
use std::sync::Arc;
pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::Vla(Box::new(load_session(
        path, backend, options,
    )?)))
}

pub(super) fn load_session(
    path: &Path,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<Pi05Session> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("PI0.5 is only registered for CUDA".into()))?;
    let cuda = &*backend;
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    let config = Arc::new(if let Some(cfg) = options.config.clone() {
        cfg
    } else if config_path.is_file() {
        Pi05Config::from_json_file(&config_path)?
    } else {
        Pi05Config::default()
    });
    let synthetic = options.synthetic;
    let host_weights = match synthetic {
        Some(synthetic) => Pi05Weights::synthetic(&config, synthetic.seed)?,
        None => Pi05Weights::from_safetensors(&config, path)?,
    };
    // Synthetic (checkpoint-free) loads must not pick up stray calibration/tuning
    // files from the working directory; only honor explicitly passed paths.
    let calibration_path = options.calibration_path.clone().or_else(|| {
        (synthetic.is_none())
            .then(|| existing(root.join("calibration.json")))
            .flatten()
    });
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "PI0.5 uses compute_variant instead of precision".into(),
        ));
    }
    let compute_variant = options
        .compute_variant
        .as_deref()
        .unwrap_or("auto")
        .parse::<ComputeVariant>()?
        .resolve(
            cuda.context().caps().sm,
            calibration_path.is_some() || options.uniform_fp8_scale.is_some(),
        );
    eprintln!(
        "[apxinf] PI0.5 compute_variant={}",
        compute_variant.as_str()
    );

    let compute = match compute_variant {
        ComputeVariant::Fp8Static => {
            let scales = if let Some(scale) = options.uniform_fp8_scale {
                Arc::new(Fp8StaticActivationScales::uniform(&config, scale)?)
            } else {
                let calibration_path = calibration_path.ok_or_else(|| {
                    Error::Other(
                        "FP8 PI0.5 requires LoadOptions.calibration_path or calibration.json"
                            .into(),
                    )
                })?;
                let checkpoint = checkpoint_identity(path)?;
                let calibration =
                    Fp8StaticCalibration::from_json_file(&calibration_path, &config, &checkpoint)?;
                Arc::new(Fp8StaticActivationScales::from_calibration(
                    &config,
                    &calibration,
                )?)
            };
            let weights = Arc::new(Fp8StaticWeights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_fp8_static(&config, &*backend)?);
            LoadedCompute::Fp8Static {
                network: build_fp8_static_network(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    scales,
                )?,
                time_embeddings,
            }
        }
        ComputeVariant::Bf16 => {
            let weights = Arc::new(Bf16Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_bf16(&config, &*backend)?);
            LoadedCompute::Bf16 {
                network: build_bf16_network(Arc::clone(&backend), Arc::clone(&config), weights)?,
                time_embeddings,
            }
        }
        ComputeVariant::Int8Dynamic => {
            let weights = Arc::new(Int8DynamicWeights::from_host(&host_weights, cuda)?);
            let time_embeddings =
                Arc::new(upload_time_embeddings_int8_dynamic(&config, &*backend)?);
            LoadedCompute::Int8Dynamic {
                network: build_int8_dynamic_network(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                )?,
                time_embeddings,
            }
        }
        ComputeVariant::Auto => unreachable!("automatic precision was resolved"),
    };

    Ok(Pi05Session::new(backend, config, compute))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}
