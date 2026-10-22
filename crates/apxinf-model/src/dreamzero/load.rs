use super::{
    backend::{ops, Context, Policies},
    model::DreamZeroModel,
    model_runner::DreamZeroModelRunner,
    weights::DreamZeroWeights,
    DreamZeroConfig,
};
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, rc::Rc, sync::Arc};

pub(super) fn load_registered(
    path: &Path,
    device: Device,
    _backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    load_with_cuda_new(path, device, options)
}

pub(crate) fn load_with_cuda_new(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Cuda(index) = device else {
        return Err(Error::UnsupportedDevice(device));
    };
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto") | Some("bf16")
    ) || !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) {
        return Err(Error::Other("DreamZero supports the BF16 joint standard model; Flash requires separately released and verified trained weights/schedules".into()));
    }
    if options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
        || options.num_views.is_some_and(|v| v != 3)
    {
        return Err(Error::Other("DreamZero DROID requires its complete checkpoint and all three views; synthetic/quantized/custom asset profiles are unsupported".into()));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent()
            .ok_or_else(|| Error::Other("DreamZero checkpoint has no parent directory".into()))?
    };
    DreamZeroConfig::from_json_file(&root.join("config.json"))?;
    let recipes = crate::auto::cuda_recipe_options(options)?;
    if ops::convolution_cudnn_version().is_none() {
        return Err(Error::Other("DreamZero BF16 requires the native cuDNN 9 convolution provider; install its runtime and rebuild ApxInf with cuDNN development headers available".into()));
    }
    let ctx = Rc::new(Context::new_with_autotune(index, recipes.online_tune).map_err(Error::Cuda)?);
    let mut policies = Policies::default();
    policies.gemm.cache_dir = recipes.cache_dir.clone();
    policies.attention.cache_dir = recipes.cache_dir;
    let weights = DreamZeroWeights::from_safetensors(&ctx, path)?;
    let model = DreamZeroModel::new(ctx, weights, policies);
    Ok(LoadedModel::Vla(Box::new(DreamZeroModelRunner::new(model))))
}
