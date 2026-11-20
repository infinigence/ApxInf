use super::{model_runner::ModelRunner, weights::Bf16Weights, Cosmos3EdgeConfig};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use crate::cuda_cosmos3_edge::backend::ops::GemmPolicy;
use crate::cuda_cosmos3_edge::backend::{AttentionPolicy, CudaContext};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, rc::Rc, sync::Arc};

pub(crate) fn load_with_cuda_new(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Cuda(device) = device else {
        return Err(Error::UnsupportedDevice(device));
    };
    if !matches!(options.model_variant.as_deref(), None | Some("bf16"))
        || !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        )
    {
        return Err(Error::Other(
            "Cosmos3-Edge supports only the BF16 DROID direct profile".into(),
        ));
    }
    if options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.text_weight_dtype.is_some()
        || !options.assets.is_empty()
        || options.weight_layout_budget_bytes.is_some()
        || options.num_views.is_some_and(|v| v != 1)
    {
        return Err(Error::Other(
            "Cosmos3-Edge does not accept architecture, weight, calibration, or camera overrides"
                .into(),
        ));
    }
    let config = Cosmos3EdgeConfig::from_file(&path.join("config.json"))?;
    let recipe = cuda_recipe_options(options)?;
    let ctx =
        Rc::new(CudaContext::new_with_autotune(device, options.autotune).map_err(Error::Cuda)?);
    let weights = Rc::new(Bf16Weights::load(
        path,
        device,
        config.num_tasks,
        config.max_text_tokens,
    )?);
    let gemm = GemmPolicy {
        online_tune: recipe.online_tune,
        cache_dir: recipe.cache_dir.clone(),
        ..Default::default()
    };
    let attention = AttentionPolicy {
        online_tune: recipe.online_tune,
        cache_dir: recipe.cache_dir,
        ..Default::default()
    };
    Ok(LoadedModel::Vla(Box::new(ModelRunner::new(
        ctx,
        weights,
        gemm,
        attention,
        config.num_inference_steps,
        options
            .execution_policy
            .unwrap_or(crate::vla::ExecutionPolicy::PreferGraph),
    ))))
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
    fn unsupported_precision_and_overrides_fail_before_device_or_asset_access() {
        let missing = Path::new("/nonexistent-cosmos3-edge-checkpoint");
        for precision in [ModelPrecision::Fp8, ModelPrecision::W8A8] {
            let options = LoadOptions {
                precision,
                ..Default::default()
            };
            let error = load_with_cuda_new(missing, Device::Cuda(0), &options)
                .err()
                .unwrap();
            assert!(error.to_string().contains("only the BF16"));
        }
        let options = LoadOptions {
            model_variant: Some("fp16".into()),
            ..Default::default()
        };
        assert!(load_with_cuda_new(missing, Device::Cuda(0), &options)
            .err()
            .unwrap()
            .to_string()
            .contains("only the BF16"));
        let options = LoadOptions {
            num_views: Some(2),
            ..Default::default()
        };
        assert!(load_with_cuda_new(missing, Device::Cuda(0), &options)
            .err()
            .unwrap()
            .to_string()
            .contains("overrides"));
    }
}
