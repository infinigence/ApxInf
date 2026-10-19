use super::{
    backend::{ops, Context},
    config::Config,
    model::Model,
    model_runner::ModelRunner,
};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
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
    let model = build_model(&ctx, path, &text, &vae, gemm, attention)?;
    Ok(LoadedModel::Vla(Box::new(ModelRunner::new(ctx, model))))
}

use super::backend::{self, host_tensor};
use super::weights::{dit::Expert, text::TextEncoder, vae::Vae, Bias, Store};
use apxinf_core::DType;
pub(super) fn build_model(
    ctx: &Context,
    path: &Path,
    text_path: &Path,
    vae_path: &Path,
    gemm: ops::GemmPolicy,
    attention: ops::AttentionPolicy,
) -> Result<Model> {
    let mut store = Store::load(path)?;
    let video = Expert::load(&mut store, ctx, true, &gemm, &[0.])?;
    let action = Expert::load(
        &mut store,
        ctx,
        false,
        &gemm,
        &(0..10)
            .map(|step| (super::dit::sigma(step) * 1000.) as f32)
            .collect::<Vec<_>>(),
    )?;
    let proprio = store.linear(ctx, "model.proprio_encoder", Bias::Required)?;
    proprio.validate("proprio encoder", 8, 4096)?;
    drop(store);
    let rgb = (0..256)
        .map(|v| {
            let unit = half::bf16::from_f32(v as f32 / 255.).to_f32();
            half::bf16::from_f32(half::bf16::from_f32(unit * 2.).to_f32() - 1.).to_f32()
        })
        .collect::<Vec<_>>();
    Ok(Model {
        video,
        action,
        proprio,
        text: TextEncoder::load(ctx, text_path)?,
        vae: Vae::load(ctx, vae_path, DType::BF16)?,
        rgb_table: backend::upload(ctx, &host_tensor(vec![256, 1], &rgb, DType::BF16)?)?,
        gemm,
        attention,
    })
}
