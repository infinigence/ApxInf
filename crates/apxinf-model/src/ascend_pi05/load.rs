//! Load PI0.5 assets and assemble the Ascend model runner (BF16 only).

use super::model::{build_bf16_model_with_policies, L3Policies, ModelVariant};
use super::*;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Device, Error};
use apxinf_core::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) fn load_with_ascend(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Npu(device_id) = device else {
        return Err(Error::Other("PI0.5 Ascend requires Device::Npu".into()));
    };
    let context = Arc::new(backend::Context::new(device_id)?);
    Ok(LoadedModel::Vla(Box::new(load_model_runner(
        path, context, options,
    )?)))
}

pub(super) fn load_model_runner(
    path: &Path,
    backend: Arc<backend::Context>,
    options: &LoadOptions,
) -> Result<Pi05ModelRunner> {
    let policies = L3Policies::default();
    load_model_runner_with_policies(path, backend, options, policies)
}

fn load_model_runner_with_policies(
    path: &Path,
    backend: Arc<backend::Context>,
    options: &LoadOptions,
    policies: L3Policies,
) -> Result<Pi05ModelRunner> {
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    // LoadOptions.config is typed as pi05::Pi05Config (the canonical shared
    // type in auto.rs). Since ascend_pi05 has its own Pi05Config copy (to
    // satisfy model family boundary checks), we cannot use options.config
    // directly. The config override path is not yet wired for Ascend; users
    // must provide config.json on disk.
    if options.config.is_some() {
        return Err(Error::Other(
            "PI0.5 Ascend does not yet support LoadOptions.config override; use config.json".into(),
        ));
    }
    let config = Arc::new(if config_path.is_file() {
        Pi05Config::from_json_file(&config_path)?
    } else {
        Pi05Config::default()
    });
    let synthetic = options.synthetic;
    let gen_start = std::time::Instant::now();
    let host_weights = match synthetic {
        Some(synthetic) => Pi05Weights::synthetic(&config, synthetic.seed)?,
        None => Pi05Weights::from_safetensors(&config, path)?,
    };
    eprintln!(
        "[apxinf] host_weights generation: {:.3}s",
        gen_start.elapsed().as_secs_f64()
    );
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "PI0.5 uses model_variant instead of precision".into(),
        ));
    }

    // BF16 only for Ascend
    let model_variant = options
        .model_variant
        .as_deref()
        .unwrap_or("bf16")
        .parse::<ModelVariantChoice>()?;
    if model_variant != ModelVariantChoice::Bf16 {
        return Err(Error::Other(format!(
            "PI0.5 Ascend currently supports bf16 only, got {model_variant:?}"
        )));
    }
    eprintln!("[apxinf] PI0.5 Ascend model_variant=bf16");

    apxinf_ascend::l3::reset_h2d_diagnostics();
    let weights_start = std::time::Instant::now();
    let weights = Arc::new(Bf16Weights::from_host(&host_weights, &*backend)?);
    let weights_enqueue = weights_start.elapsed();
    // Synchronize stream once — all weight H2D copies were async.
    backend::synchronize(&backend)?;
    let weights_sync = weights_start.elapsed();
    eprintln!(
        "[apxinf] Bf16Weights::from_host: enqueue={:.3}s sync={:.3}s total={:.3}s",
        weights_enqueue.as_secs_f64(),
        (weights_sync - weights_enqueue).as_secs_f64(),
        weights_sync.as_secs_f64()
    );
    apxinf_ascend::l3::print_h2d_diagnostics();

    apxinf_ascend::l3::reset_h2d_diagnostics();
    let time_start = std::time::Instant::now();
    let time_embeddings = Arc::new(upload_time_embeddings_bf16(&config, &*backend)?);
    let time_enqueue = time_start.elapsed();
    // Synchronize stream once — time embedding H2D copies were async.
    backend::synchronize(&backend)?;
    let time_sync = time_start.elapsed();
    eprintln!(
        "[apxinf] upload_time_embeddings: enqueue={:.3}s sync={:.3}s total={:.3}s",
        time_enqueue.as_secs_f64(),
        (time_sync - time_enqueue).as_secs_f64(),
        time_sync.as_secs_f64()
    );
    apxinf_ascend::l3::print_h2d_diagnostics();
    let model_start = std::time::Instant::now();
    let model = ModelVariant::Bf16 {
        model: build_bf16_model_with_policies(
            Arc::clone(&backend),
            Arc::clone(&config),
            weights,
            policies,
        )?,
        time_embeddings,
    };
    eprintln!(
        "[apxinf] build_bf16_model: {:.3}s",
        model_start.elapsed().as_secs_f64()
    );

    Ok(Pi05ModelRunner::new(backend, config, model))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}
