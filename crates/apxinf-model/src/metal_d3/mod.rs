//! Native d3 decisions on Apple Metal: the checkpoint's decision path as an
//! ApxInf-owned MLX composition, registered as `metal_d3`.
//!
//! The port is independent of the CUDA `d3` module at `src/d3/`: it copies the
//! package contract it needs (see `doc/model-layer-architecture.md`, "Where
//! backend-native model code lives") and admits only the d3-lite geometry.
//!
//! Text requests run `language_model.*` (24 layers, 18 GatedDeltaNet + 6
//! non-causal full attention) and the FP32 255-code readout, one compiled
//! forward per request, no decode and no KV cache. Image requests add
//! `visual.*` (12 blocks + spatial merger), the checkpoint's
//! resize/normalize/patchify contract, and the multimodal placeholder/mRoPE
//! wiring for one still image. Video inputs are not wired. See `README.md`.

mod config;
mod model;
mod preprocess;
mod profiles;
mod vision;
mod weights;

pub use config::{Config, Geometry, PreprocessorContract, VisionContract};
pub use model::Model;
pub use preprocess::{preprocess, smart_resize, PreparedImage};
pub use profiles::ProfileLimits;
pub use weights::Weights;

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{MlxBackend, Stream};

use crate::decision::{
    CodeDecisionRuntime, DecisionInput, DecisionModel, DecisionOutput, DecisionScore,
};
use crate::llm_trait::LlmCapabilities;
use crate::{LoadOptions, LoadedModel, ModelPrecision};

use weights::VisionWeights;

/// Device-independent option admission, checked before the checkpoint is read.
/// One BF16 composition of the checkpoint as stored (`auto` selects it); no
/// CUDA tuning, calibration, synthetic weights or auxiliary assets. Returns the
/// text-profile limits from the shared d3 options.
fn admit(options: &LoadOptions) -> Result<ProfileLimits> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "bf16")
    ) {
        return Err(Error::Contract(
            "d3 Metal supports the auto and bf16 variants",
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
    {
        return Err(Error::Contract(
            "d3 Metal loads the BF16 checkpoint as stored; CUDA tuning, calibration, \
             synthetic weights and auxiliary assets are unsupported",
        ));
    }
    ProfileLimits::text(options)
}

/// `registry::ModelFactory` entry point, resolved by `AutoModel` for
/// `Device::Metal` from the detected `d3` name or `--model-name metal_d3`.
pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let text_limits = admit(options)?;
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract(
            "d3-lite Metal requires the native MLX backend",
        ))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("d3-lite loads a checkpoint directory"));
    }
    let config = Config::from_dir(path)?;
    let stream: Stream = native.stream().clone();
    let weights = Weights::load(config.geometry, &stream, path)?;
    let vision_weights = VisionWeights::load(config.vision, &stream, path)?;
    let model = Model::new(&config, stream, weights, vision_weights, text_limits)?;
    Ok(LoadedModel::Decision(DecisionModel::Code(Box::new(
        D3Runner {
            codes: config.codes.clone(),
            token_ids: config.token_ids.clone(),
            model,
        },
    ))))
}

/// Encoded-prompt decision runtime: the caller owns tokenization
/// and the chat template; this runner scores the leading package codes of one
/// complete prompt.
pub struct D3Runner {
    codes: Vec<String>,
    token_ids: Vec<u32>,
    model: Model,
}

impl CodeDecisionRuntime for D3Runner {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::VISION
    }

    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        let count = input.candidate_count;
        let probabilities = if let Some(image) = input.image {
            let pixels = image.pixel_values.to_f32_vec()?;
            self.model
                .image_probabilities(&pixels, image.grid_thw, input.token_ids, count)?
        } else {
            self.model.probabilities(input.token_ids, count)?
        };
        if probabilities.len() != count {
            return Err(Error::Other(format!(
                "d3-lite decision returned {} probabilities for {count} candidates",
                probabilities.len()
            )));
        }
        if probabilities.iter().any(|value| !value.is_finite()) {
            return Err(Error::Other(
                "d3-lite readout produced non-finite probabilities".into(),
            ));
        }
        // First maximum wins, matching the package reference and the CUDA path.
        let mut selected_index = 0usize;
        for (index, value) in probabilities.iter().enumerate() {
            if *value > probabilities[selected_index] {
                selected_index = index;
            }
        }
        let scores = self.codes[..count]
            .iter()
            .zip(&self.token_ids[..count])
            .zip(&probabilities)
            .map(|((code, &token_id), &probability)| DecisionScore {
                code: code.clone(),
                token_id,
                probability,
            })
            .collect();
        Ok(DecisionOutput {
            selected_index,
            scores,
        })
    }

    fn code_names(&self) -> &[String] {
        &self.codes
    }

    fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert_eq!(
            admit(&LoadOptions::default()).unwrap(),
            ProfileLimits::default()
        );
        for variant in ["auto", "bf16"] {
            assert!(
                admit(&LoadOptions {
                    model_variant: Some(variant.into()),
                    ..LoadOptions::default()
                })
                .is_ok(),
                "{variant}"
            );
        }
        let limited = admit(&LoadOptions {
            d3_text_profile_limit: Some(3),
            d3_text_profile_token_budget: Some(2_048),
            ..LoadOptions::default()
        })
        .unwrap();
        assert_eq!((limited.profiles.get(), limited.tokens.get()), (3, 2_048));
        for options in [
            LoadOptions {
                model_variant: Some("bf16-compiled".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                autotune: true,
                ..LoadOptions::default()
            },
            LoadOptions {
                assets: [("draft".to_owned(), "draft".into())].into(),
                ..LoadOptions::default()
            },
            LoadOptions {
                d3_text_profile_limit: Some(0),
                ..LoadOptions::default()
            },
            LoadOptions {
                precision: ModelPrecision::Fp8,
                ..LoadOptions::default()
            },
        ] {
            assert!(admit(&options).is_err(), "{options:?}");
        }
    }
}
