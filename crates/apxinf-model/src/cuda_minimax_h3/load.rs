//! `AutoModel` entry: validate the request for this family, then load the checkpoints into a runner.
use super::backend;
use super::model::blocks::{AudioVaeBlocks, VideoVaeBlocks};
use super::model_runner::{AudioDecoder, ModelRunner, VideoDecoder, TILE_BATCH};
use super::package::Package;
use super::weights::{AudioVaeWeights, DitWeights, Pinning, TextWeights, VideoVaeWeights};
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_tokenizer::Tokenizer;
use backend::ops::GemmPolicy;
use backend::{AttentionPolicy, CudaContext};
use std::{path::Path, rc::Rc, sync::Arc};

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
            "MiniMax-H3 ships only the INT8 ConvRot (W8A8) package; use precision auto or w8a8"
                .into(),
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
            "MiniMax-H3 takes its weights from the package and accepts no replacement assets, synthetic weights, calibration or view overrides".into(),
        ));
    }
    Ok(())
}

/// Tiles the video decoder runs per pass over its weights: `APXINF_H3_VAE_BATCH` or the default. A value that is not a
/// positive integer is an error naming the variable, so a typo cannot pass for a setting that took effect.
fn vae_batch(value: Option<&str>) -> Result<usize> {
    let Some(text) = value else {
        return Ok(TILE_BATCH);
    };
    match text.trim().parse::<usize>() {
        Ok(batch) if batch > 0 => Ok(batch),
        _ => Err(Error::Other(format!(
            "APXINF_H3_VAE_BATCH={text:?} is not a positive number of tiles"
        ))),
    }
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
            "MiniMax-H3 needs an sm_80 or newer GPU, found sm_{}",
            caps.sm
        )));
    }
    let tokenizer = Tokenizer::from_qwen2_vocab(&package.tokenizer)?;
    let dit = Rc::new(DitWeights::load(&package.transformer, &ctx)?);
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
    let text = TextWeights::open(&package.text_encoder, ctx.clone(), Pinning::from_env())?;
    let batch = vae_batch(std::env::var("APXINF_H3_VAE_BATCH").ok().as_deref())?;
    let video = VideoDecoder::new(
        ctx.clone(),
        VideoVaeBlocks::new(
            ctx.clone(),
            Rc::new(VideoVaeWeights::load(&package.video_vae, ctx.clone())?),
        ),
        batch,
        gemm.clone(),
        attention.clone(),
    );
    let audio = AudioDecoder::new(
        ctx.clone(),
        AudioVaeBlocks::new(
            ctx.clone(),
            Rc::new(AudioVaeWeights::load(&package.audio_vae, &ctx)?),
        ),
    );
    Ok(LoadedModel::Video(Box::new(ModelRunner::new(
        ctx, dit, tokenizer, text, video, audio, gemm, attention,
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
    fn vae_batch_is_the_default_when_unset_and_a_positive_count_otherwise() {
        assert_eq!(vae_batch(None).unwrap(), TILE_BATCH);
        assert_eq!(vae_batch(Some("4")).unwrap(), 4);
        assert_eq!(vae_batch(Some(" 3 ")).unwrap(), 3);
        for bad in ["", "0", "-1", "two", "1.5"] {
            let message = vae_batch(Some(bad)).unwrap_err().to_string();
            assert!(
                message.contains("APXINF_H3_VAE_BATCH"),
                "{bad:?}: {message}"
            );
        }
    }

    #[test]
    fn unsupported_options_fail_before_any_file_or_device_access() {
        let missing = Path::new("/nonexistent-h3-package");
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
            assert!(error.contains("MiniMax-H3"), "{error}");
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
            Path::new("/nonexistent-h3-package"),
            Device::Cuda(0),
            &LoadOptions::default(),
        );
        assert!(error.err().unwrap().to_string().contains("apxinf_h3.json"));
    }
}
