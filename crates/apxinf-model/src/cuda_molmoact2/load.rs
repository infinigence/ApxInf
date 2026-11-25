//! Loading of the MolmoAct2 LIBERO BF16 profile and its registry entries.
use super::{model_runner::ModelRunner, weights::Bf16Weights, MolmoAct2Config};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use crate::cuda_molmoact2::backend::ops::GemmPolicy;
use crate::cuda_molmoact2::backend::{AttentionPolicy, CudaContext};
use crate::vla::ExecutionPolicy;
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, rc::Rc, sync::Arc};

/// Reject every option this fixed profile cannot honour before touching the device or the weights.
fn check_options(options: &LoadOptions) -> Result<()> {
    if !matches!(options.model_variant.as_deref(), None | Some("bf16"))
        || !matches!(options.precision, ModelPrecision::Auto | ModelPrecision::Bf16)
    {
        return Err(Error::Other("MolmoAct2 supports only the BF16 profile".into()));
    }
    if options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.text_weight_dtype.is_some()
        || !options.assets.is_empty()
        || options.num_views.is_some_and(|v| v != 2)
    {
        return Err(Error::Other(
            "MolmoAct2 does not accept architecture, weight, calibration, asset or camera overrides".into(),
        ));
    }
    Ok(())
}

pub(crate) fn load_with_cuda_new(path: &Path, device: Device, options: &LoadOptions) -> Result<LoadedModel> {
    let Device::Cuda(device) = device else {
        return Err(Error::UnsupportedDevice(device));
    };
    check_options(options)?;
    let dir = if path.is_dir() { path } else { path.parent().unwrap_or_else(|| Path::new(".")) };
    // The config decides whether this checkpoint is the plain continuous-action model; a pure VLM or
    // pretrain checkpoint with the same `model_type` is rejected here, before any weight is read.
    let config = MolmoAct2Config::from_file(&dir.join("config.json"))?;
    config.check_libero_stats_file(&dir.join("norm_stats.json"))?;
    let recipe = cuda_recipe_options(options)?;
    let ctx = Rc::new(CudaContext::new_with_autotune(device, options.autotune).map_err(Error::Cuda)?);
    let weights = Rc::new(Bf16Weights::load(dir, &config, device)?);
    let gemm = GemmPolicy { online_tune: recipe.online_tune, cache_dir: recipe.cache_dir.clone(), ..Default::default() };
    let attention = AttentionPolicy { online_tune: recipe.online_tune, cache_dir: recipe.cache_dir, ..Default::default() };
    // `infer` builds and caches a prepared plan: with a captured graph by default (`PreferGraph`, like PI0.5), or a
    // prepared-eager plan without graph when the caller asks for `Eager` (`execution_policy`); `execution_mode()`
    // reports what ran.
    let policy = options.execution_policy.unwrap_or(ExecutionPolicy::PreferGraph);
    Ok(LoadedModel::Vla(Box::new(ModelRunner::with_policy(ctx, weights, config, gemm, attention, policy)?)))
}

pub(super) fn load_registered(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    load_with_cuda_new(path, device, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_options_fail_before_device_or_asset_access() {
        let missing = Path::new("/nonexistent-molmoact2-checkpoint");
        for precision in [ModelPrecision::Fp8, ModelPrecision::W8A8] {
            let options = LoadOptions { precision, ..Default::default() };
            let error = load_with_cuda_new(missing, Device::Cuda(0), &options).err().unwrap();
            assert!(error.to_string().contains("only the BF16"), "{error}");
        }
        let cases = [
            LoadOptions { model_variant: Some("fp16".into()), ..Default::default() },
            LoadOptions { num_views: Some(1), ..Default::default() },
        ];
        for options in cases {
            let error = load_with_cuda_new(missing, Device::Cuda(0), &options).err().unwrap();
            let text = error.to_string();
            assert!(text.contains("MolmoAct2"), "{text}");
        }
    }

    #[test]
    fn non_cuda_devices_are_unsupported() {
        let error = load_with_cuda_new(Path::new("/x"), Device::Cpu, &LoadOptions::default()).err().unwrap();
        assert!(matches!(error, Error::UnsupportedDevice(_)));
    }
}
