use super::{
    backend::{ops, Context},
    config::Config,
    model::build_model,
    model_runner::FastWamModelRunner,
    weights::Bf16Weights,
};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, rc::Rc, sync::Arc};
pub(in crate::cuda_fastwam) fn load_registered(
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
    let Device::Cuda(device_id) = device else {
        return Err(Error::Other("Fast-WAM requires CUDA".into()));
    };
    if options.precision != ModelPrecision::Auto
        || !matches!(
            options.model_variant.as_deref(),
            None | Some("auto") | Some("bf16")
        )
    {
        return Err(Error::Other(
            "Fast-WAM supports model_variant=bf16 only".into(),
        ));
    }
    if options.num_views.is_some_and(|views| views != 2)
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other("Fast-WAM does not support view/config overrides, synthetic weights or calibration options".into()));
    }
    let config = Config::load(path)?;
    debug_assert_eq!(config.steps, 10);
    let asset = |name: &str| {
        options
            .assets
            .get(name)
            .cloned()
            .or_else(|| {
                let p = path.join(name);
                p.is_dir().then_some(p)
            })
            .ok_or_else(|| {
                Error::Other(format!(
                    "Fast-WAM requires LoadOptions.assets[{name:?}] or checkpoint/{name}"
                ))
            })
    };
    let text = asset("text_encoder")?;
    let vae = asset("vae")?;
    let ctx = Rc::new(Context::new(device_id).map_err(Error::Cuda)?);
    let recipe = cuda_recipe_options(options)?;
    let mut gemm = ops::GemmPolicy::default();
    gemm.cache_dir = recipe.cache_dir.clone();
    gemm.online_tune = recipe.online_tune;
    let mut attention = ops::AttentionPolicy::default();
    attention.cache_dir = recipe.cache_dir;
    attention.online_tune = recipe.online_tune;
    let weights = Bf16Weights::load(&ctx, path, &text, &vae)?;
    let model = build_model(&ctx, weights, gemm, attention)?;
    Ok(LoadedModel::Vla(Box::new(FastWamModelRunner::new(
        ctx, model,
    ))))
}
