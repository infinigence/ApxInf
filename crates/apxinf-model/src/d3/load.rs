//! Official package loader for the cuda-new d3 runtime.

use std::path::Path;

use apxinf_core::{Device, Error, Result};
use apxinf_cuda_new::CudaContext;

use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use crate::d3::model::D3Policies;
use crate::d3::runner::D3Runner;
use crate::d3::weights::bf16::D3DeviceWeights;
use crate::d3::{D3CheckpointWeights, D3PackageConfig, D3ReadoutWeights};

pub(crate) fn load_with_cuda_new(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Cuda(ordinal) = device else {
        return Err(Error::Other("d3-mini requires CUDA".into()));
    };
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "bf16")
    ) {
        return Err(Error::Other(
            "d3-mini supports only the official BF16 variant".into(),
        ));
    }
    if !options.assets.is_empty()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.text_weight_dtype.is_some()
    {
        return Err(Error::Other(
            "d3-mini requires its native package and does not accept replacement assets, synthetic weights, or calibration".into(),
        ));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = D3PackageConfig::from_dir(root)?;
    let (checkpoint, _) = apxinf_loader::safetensors::load_native_path(root)
        .map_err(|error| Error::Other(format!("load d3 backbone: {error}")))?;
    let (readout, _) = apxinf_loader::safetensors::load_native(&root.join("readout.safetensors"))
        .map_err(|error| Error::Other(format!("load d3 readout: {error}")))?;
    let checkpoint = D3CheckpointWeights::from_map(checkpoint)?;
    let readout = D3ReadoutWeights::from_map(&config, readout)?;
    let recipe = cuda_recipe_options(options)?;
    let ctx = CudaContext::new_with_autotune(ordinal, options.autotune).map_err(Error::Cuda)?;
    let policies = D3Policies::for_recipe(recipe.cache_dir, recipe.online_tune);
    let weights = D3DeviceWeights::from_host(&config, checkpoint, readout, &ctx)?;
    ctx.synchronize().map_err(Error::Cuda)?;
    Ok(LoadedModel::Decision(Box::new(D3Runner::new(
        ctx,
        config,
        weights,
        policies,
        options.d3_text_profile_limit,
        options.d3_text_profile_token_budget,
    )?)))
}
