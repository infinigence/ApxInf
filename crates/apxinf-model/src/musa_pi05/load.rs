//! Load PI0.5 assets onto a MUSA device and assemble this port's model.
//!
//! The device arm is the only dispatch here: `LoadOptions::config` belongs to
//! the CUDA family, so `AutoModel` converts it and hands this module the port's
//! own [`Pi05Config`]. Weights are synthetic when requested, else the
//! SafeTensors at `path`.

use super::model::{
    build_bf16_model_with_policies, build_int8_dynamic_model_with_policies,
    upload_time_embeddings_bf16, upload_time_embeddings_int8_dynamic, L3Policies, ModelVariant,
};
use super::weights::{Bf16Weights, Int8DynamicWeights, Int8Scope, Pi05Weights};
use super::backend::Context;
use super::{backend, ModelVariantChoice, Pi05Config, Pi05ModelRunner};
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Device, Error, Result};
use std::path::Path;
use std::sync::Arc;

pub(crate) fn load(
    path: &Path,
    device: Device,
    options: &LoadOptions,
    config: Option<Pi05Config>,
) -> Result<LoadedModel> {
    let Device::Musa(device_id) = device else {
        return Err(Error::Other(format!("PI0.5 requires MUSA, not {device}")));
    };
    let context = Arc::new(backend::Context::new(device_id)?);
    Ok(LoadedModel::Vla(Box::new(load_model_runner(
        path, context, options, config,
    )?)))
}

pub(crate) fn load_model_runner(
    path: &Path,
    backend: Arc<backend::Context>,
    options: &LoadOptions,
    config: Option<Pi05Config>,
) -> Result<Pi05ModelRunner> {
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    let config = Arc::new(match config {
        Some(config) => config,
        None if config_path.is_file() => Pi05Config::from_json_file(&config_path)?,
        None => Pi05Config::default(),
    });
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "PI0.5 uses model_variant instead of precision".into(),
        ));
    }
    // FP8 calibration is an input to the CUDA quantization path MUSA does not
    // have; accepting one here would leave the caller believing it took effect.
    if options.calibration_path.is_some() || options.uniform_fp8_scale.is_some() {
        return Err(Error::Other(
            "PI0.5 calibration is an FP8 input; MUSA has no FP8 path".into(),
        ));
    }
    let requested = options
        .model_variant
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelVariantChoice>()?;
    // Zero is the device key: MUSA reports `mp_22`, not an `sm_` compute
    // capability, so `auto` resolves to BF16. INT8 is a device key like any
    // other: which projections a tile can serve is decided per shape at pack
    // time.
    //
    // The FP8 refusal sits before `ensure_supported`, not after: that check
    // reports a vendor `sm_N` compute capability, which this device does not
    // have. Answering a MUSA caller with "unsupported on sm_0" would name a
    // capability nothing here reports and would not say what to ask for
    // instead.
    let model_variant = requested.resolve(0, false);
    if matches!(model_variant, ModelVariantChoice::Fp8Static) {
        return Err(Error::Other(
            "PI0.5 model_variant=fp8_static needs an FP8 path MUSA does not have; \
             ask for bf16, int8_dynamic or int8_dynamic_no_encode"
                .into(),
        ));
    }
    let model_variant = model_variant.ensure_supported(0)?;
    eprintln!("[apxinf] PI0.5 model_variant={}", model_variant.as_str());

    let host_weights = match options.synthetic {
        Some(synthetic) => Pi05Weights::synthetic(&config, synthetic.seed)?,
        None => Pi05Weights::from_safetensors(&config, path)?,
    };
    let policies = L3Policies::default();
    install_tactics(&backend, options)?;
    let model = match model_variant {
        ModelVariantChoice::Int8Dynamic | ModelVariantChoice::Int8DynamicNoEncode => {
            let scope = match model_variant {
                ModelVariantChoice::Int8DynamicNoEncode => Int8Scope::WithoutEncode,
                _ => Int8Scope::All,
            };
            let weights = Arc::new(Int8DynamicWeights::from_host(
                &host_weights, &*backend, scope,
            )?);
            // Packing is the only reader of the host copy; the device weights
            // and their FP16 scale renderings are independent of it.
            drop(host_weights);
            let time_embeddings = upload_time_embeddings_int8_dynamic(&config, &*backend)?;
            ModelVariant::Int8Dynamic {
                model: build_int8_dynamic_model_with_policies(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    policies,
                    &time_embeddings,
                )?,
                name: model_variant.as_str(),
            }
        }
        _ => {
            let weights = Arc::new(Bf16Weights::from_host(&host_weights, &*backend)?);
            // Bound peak memory: the decode layouts are materialised from here, and
            // there is no reason for them to meet a live FP32 copy of the checkpoint.
            drop(host_weights);
            weights.prepare_assembly_weights(&backend)?;
            let time_embeddings = upload_time_embeddings_bf16(&config, &*backend)?;
            ModelVariant::Bf16 {
                model: build_bf16_model_with_policies(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    policies,
                    &time_embeddings,
                )?,
            }
        }
    };

    let runner = Pi05ModelRunner::new(backend, config, model);
    Ok(match options.execution_policy {
        Some(policy) => runner.with_execution_policy(policy),
        None => runner,
    })
}

/// Where this load's tactic store lives.
///
/// `None` uses the directory derived from the running device; `Some("")` is the
/// policy's own spelling for "no disk caching"; `Some(dir)` names the directory
/// (the loader appends `tactics.json` itself).
///
/// This is the declaration; [`install_tactics`] is what enacts it. The operator
/// policies no longer carry `cache_dir` / `online_tune`: tuning is resolved
/// once here, and a per-call copy could only drift from it.
fn tuning_paths(
    cache_dir: Option<&Path>,
    caps: &backend::MusaDeviceCaps,
    versions: &backend::MusaLibraryVersions,
) -> Option<backend::tuning::TuningPaths> {
    match cache_dir {
        Some(directory) if directory.as_os_str().is_empty() => None,
        Some(directory) => Some(backend::tuning::TuningPaths::for_directory(directory)),
        None => Some(backend::tuning::TuningPaths::resolve_for_device(
            "configs/tuning",
            caps,
            versions,
        )),
    }
}

/// Whether autotuning may measure missing shapes this session.
fn tuning_mode(autotune: bool) -> backend::tuning::TuningMode {
    if autotune {
        backend::tuning::TuningMode::AutoTune
    } else {
        backend::tuning::TuningMode::Inference
    }
}

/// Install the tactic database [`tuning_paths`] declares.
///
/// Once, at load, before the model prepares or captures anything: a capture
/// must not be where a provider first gets chosen, and a weight prepared under
/// one choice has to be used under that same one.
///
/// A missing database is not an error: every shape then resolves to its safe
/// inference default, and failing here would report "this device has not been
/// measured" as though the checkpoint were broken.
fn install_tactics(backend: &Context, options: &LoadOptions) -> Result<()> {
    let paths = tuning_paths(
        options.tuning_path.as_deref(),
        backend.caps(),
        backend.library_versions(),
    );
    let database = paths
        .as_ref()
        .filter(|paths| paths.tactics.is_file())
        .map(|paths| backend::tuning::TuningDb::from_json_file(&paths.tactics))
        .transpose()?;
    backend::tuning::configure_tuning(
        backend,
        tuning_mode(options.autotune),
        database.as_slice(),
        paths,
    )
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "apxinf-musa-pi05-load-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn caps() -> backend::MusaDeviceCaps {
        backend::MusaDeviceCaps {
            device_name: "M1000".into(),
            compute_major: 2,
            compute_minor: 2,
            mp: 22,
            multiprocessor_count: 8,
            arch_family: apxinf_musa::MusaArchFamily::Mp22,
            hardware_warp_size: 128,
        }
    }

    fn versions() -> backend::MusaLibraryVersions {
        backend::MusaLibraryVersions {
            musa: 1,
            mublas_lt: 0,
            binaries: [("libmublas.so".to_string(), "test".to_string())].into(),
        }
    }

    #[test]
    fn an_empty_cache_dir_disables_disk_caching() {
        assert!(tuning_paths(Some(Path::new("")), &caps(), &versions()).is_none());
    }

    #[test]
    fn a_named_directory_is_taken_as_given() {
        let directory = temporary_directory("tactics");
        let requested = directory.join("explicit.recipes");
        let paths = tuning_paths(Some(requested.as_path()), &caps(), &versions())
            .expect("a named directory");
        assert_eq!(paths.directory, requested);
        assert_eq!(paths.tactics, requested.join("tactics.json"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn the_default_follows_the_running_device() {
        let paths = tuning_paths(None, &caps(), &versions()).expect("the default");
        assert_eq!(paths.directory, Path::new("configs/tuning/musa/m1000-mp22"));
    }

    #[test]
    fn autotune_selects_the_mode() {
        assert_eq!(tuning_mode(true), backend::tuning::TuningMode::AutoTune);
        assert_eq!(tuning_mode(false), backend::tuning::TuningMode::Inference);
    }

    /// The declaration has to reach the context, or `--tactics` is a lie: a
    /// policy a caller can read back but no kernel consults is the gap this
    /// closes. [`tuning_paths`] builds the declaration; this is its
    /// installation.
    #[test]
    #[ignore = "requires a physical MUSA device"]
    fn the_declared_tactics_directory_is_installed_on_the_context() {
        let backend = Context::new(0).expect("MUSA device 0");
        let directory = temporary_directory("installed");
        let options = LoadOptions {
            tuning_path: Some(directory.clone()),
            autotune: true,
            ..LoadOptions::default()
        };

        install_tactics(&backend, &options).expect("install");

        let session = backend.tuning();
        assert_eq!(session.mode(), backend::tuning::TuningMode::AutoTune);
        assert_eq!(
            session.paths().map(|paths| paths.directory.clone()),
            Some(directory.clone())
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
