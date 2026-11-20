//! `AutoModel` entry: validate the request for this family, then load the three checkpoints into a runner.
use super::backend;
use super::model::blocks::VaeBlocks;
use super::model_runner::{ModelRunner, PreparedVae};
use super::package::Package;
use super::weights::{DitWeights, Pinning, TextWeights, VaeWeights};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_tokenizer::Tokenizer;
use backend::ops::GemmPolicy;
use backend::{AttentionPolicy, CudaContext};
use std::{path::Path, rc::Rc, sync::Arc};

/// Tiles of one VAE batch. Four 256-pixel tiles keep the decoder's activations near 100 MiB.
const VAE_BATCH_TILES: usize = 4;

/// Options this family does not implement are rejected, not ignored.
fn check_options(options: &LoadOptions) -> Result<()> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "int8_convrot")
    ) || !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::W8A8
    ) {
        return Err(Error::Other(
            "MiniMax-H3-Image ships only the INT8 ConvRot (W8A8) package; use precision auto or w8a8".into(),
        ));
    }
    if !options.assets.is_empty()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.text_weight_dtype.is_some()
        || options.num_views.is_some()
    {
        return Err(Error::Other(
            "MiniMax-H3-Image takes its weights from the package and accepts no replacement assets, synthetic weights, calibration or view overrides".into(),
        ));
    }
    Ok(())
}

pub(crate) fn load_with_cuda_new(
    path: &Path,
    device: Device,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let Device::Cuda(ordinal) = device else {
        return Err(Error::UnsupportedDevice(device));
    };
    check_options(options)?;
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let package = Package::open(root)?;
    let recipe = cuda_recipe_options(options)?;
    let ctx =
        Rc::new(CudaContext::new_with_autotune(ordinal, options.autotune).map_err(Error::Cuda)?);
    let caps = ctx.caps();
    if caps.sm < 80 {
        return Err(Error::Other(format!(
            "MiniMax-H3-Image needs an sm_80 or newer GPU, found sm_{}",
            caps.sm
        )));
    }
    let tokenizer = Tokenizer::from_qwen2_vocab(&package.tokenizer)?;
    let dit = Rc::new(DitWeights::load(&package.transformer, &ctx)?);
    let vae_weights = Rc::new(VaeWeights::load(&package.vae, &ctx)?);
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
    let capture = ModelRunner::capture_enabled();
    let text = TextWeights::open(&package.text_encoder, ctx.clone(), Pinning::from_env())?;
    let vae_blocks = Rc::new(VaeBlocks::new(ctx.clone(), vae_weights, VAE_BATCH_TILES)?);
    let vae = PreparedVae::new(
        ctx.clone(),
        vae_blocks,
        capture,
        gemm.clone(),
        attention.clone(),
    )?;
    Ok(LoadedModel::Image(Box::new(ModelRunner::new(
        ctx, dit, vae, tokenizer, text, gemm, attention,
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
    fn unsupported_options_fail_before_any_file_or_device_access() {
        let missing = Path::new("/nonexistent-h3-image-package");
        for options in [
            LoadOptions {
                precision: ModelPrecision::Fp8,
                ..Default::default()
            },
            LoadOptions {
                precision: ModelPrecision::Bf16,
                ..Default::default()
            },
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..Default::default()
            },
            LoadOptions {
                num_views: Some(2),
                ..Default::default()
            },
            LoadOptions {
                text_weight_dtype: Some(apxinf_core::DType::BF16),
                ..Default::default()
            },
            LoadOptions {
                uniform_fp8_scale: Some(1.0),
                ..Default::default()
            },
        ] {
            let error = load_with_cuda_new(missing, Device::Cuda(0), &options)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("MiniMax-H3-Image"), "{error}");
            assert!(
                !error.contains("nonexistent"),
                "rejected before reading the package: {error}"
            );
        }
        let cpu = load_with_cuda_new(missing, Device::Cpu, &LoadOptions::default())
            .err()
            .unwrap();
        assert!(matches!(cpu, Error::UnsupportedDevice(_)));
        assert!(check_options(&LoadOptions {
            model_variant: Some("int8_convrot".into()),
            ..Default::default()
        })
        .is_ok());
    }

    #[test]
    fn a_directory_without_the_manifest_is_reported_by_the_package_reader() {
        let error = load_with_cuda_new(
            Path::new("/nonexistent-h3-image-package"),
            Device::Cuda(0),
            &LoadOptions::default(),
        );
        assert!(error
            .err()
            .unwrap()
            .to_string()
            .contains("apxinf_h3_image.json"));
    }
}
