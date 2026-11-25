//! Registry loader. The published `config.json` is `{"vlm_family": "qwen3_vl"}`, so the family is selected
//! with an explicit `model_name = "lingbot_vla2"`; the architecture is then proven against the tensor shapes.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};

use super::backend::{self, ops};
use super::config::LingbotVla2Config;
use super::model_runner::LingbotVla2ModelRunner;
use super::weights::model_weights::LingbotWeights;
use crate::auto::{cuda_recipe_options, LoadOptions, LoadedModel, ModelPrecision};

pub(super) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other(
            "LingBot-VLA 2.0 requires a CUDA device".into(),
        ));
    }
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "LingBot-VLA 2.0 uses model_variant instead of precision".into(),
        ));
    }
    check_variant(options.model_variant.as_deref())?;
    let config = LingbotVla2Config {
        flow_steps: super::config::FLOW_STEPS,
        views: options.num_views.unwrap_or(super::config::NUM_VIEWS),
    };
    config.validate()?;
    let cuda = backend::downcast_arc(backend)
        .ok_or_else(|| Error::Other("LingBot-VLA 2.0 is only registered for CUDA".into()))?;
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    let weights = Arc::new(LingbotWeights::load(
        cuda.context(),
        dir,
        config.flow_steps,
    )?);
    let recipe = cuda_recipe_options(options)?;
    let gemm = ops::GemmPolicy {
        cache_dir: recipe.cache_dir.clone(),
        online_tune: recipe.online_tune,
        ..ops::GemmPolicy::default()
    };
    let attention = ops::AttentionPolicy {
        cache_dir: recipe.cache_dir,
        online_tune: recipe.online_tune,
        ..ops::AttentionPolicy::default()
    };
    Ok(LoadedModel::Vla(Box::new(LingbotVla2ModelRunner::new(
        cuda, weights, gemm, attention,
    ))))
}

/// Only the INT8 ConvRot build exists: BF16 weights are 12.5 GB and do not fit the 8 GB card this port targets.
fn check_variant(variant: Option<&str>) -> Result<()> {
    match variant {
        None | Some("auto") | Some("int8_convrot") => Ok(()),
        Some(other) => Err(Error::Other(format!(
            "LingBot-VLA 2.0 supports model_variant int8_convrot (BF16 weights are 12.5 GB and do not fit an 8 GB card), got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::check_variant;

    #[test]
    fn accepts_the_int8_build_and_its_aliases() {
        for v in [None, Some("auto"), Some("int8_convrot")] {
            assert!(check_variant(v).is_ok(), "{v:?}");
        }
    }

    #[test]
    fn rejects_every_other_precision_with_the_reason() {
        for v in ["bf16", "fp8_static", "int8_dynamic", ""] {
            let message = check_variant(Some(v)).unwrap_err().to_string();
            assert!(
                message.contains("int8_convrot") && message.contains(&format!("got {v}")),
                "{v}: {message}"
            );
        }
    }
}
