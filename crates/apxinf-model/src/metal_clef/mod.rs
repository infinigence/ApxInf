//! Native Clef Flash on Apple Metal: Qwen3.5 hybrid backbone (affine Q4) in
//! the stock MLX-VLM composition, and the BF16 joint schema head. No Python
//! model execution.
mod config;
mod encode;
mod model;
mod weights;

pub use config::{Config, MAX_INPUT_TOKENS};

use crate::decision::{
    DecisionAnswer, DecisionInput, DecisionOutput, DecisionRuntime, DecisionSchema, DecisionScore,
};
use crate::llm_trait::LlmCapabilities;
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use apxinf_tokenizer::Tokenizer;
use model::Network;
use std::path::Path;
use std::sync::Arc;

pub struct ClefRunner {
    tokenizer: Tokenizer,
    network: Network,
}

/// Device-independent option admission, checked before the checkpoint is read.
/// `reference` is the only composition; `auto` selects it.
fn admit(options: &LoadOptions) -> Result<()> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "reference")
    ) {
        return Err(Error::Contract(
            "Clef MLX supports the auto and reference variants",
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
            "Clef MLX loads the affine Q4 checkpoint as stored; CUDA tuning, \
             calibration, synthetic weights and auxiliary assets are unsupported",
        ));
    }
    Ok(())
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    admit(options)?;
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract("Clef requires the native MLX backend"))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("Clef loads a checkpoint directory"));
    }
    let read = |name: &str| {
        std::fs::read_to_string(path.join(name))
            .map_err(|e| Error::Other(format!("cannot read Clef {name}: {e}")))
    };
    let config = Config::from_json(&read("config.json")?, &read("joint_head_config.json")?)?;
    let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))?;
    let stream = native.stream().clone();
    let backbone = weights::Backbone::load(&config, &stream, path)?;
    let head = weights::Head::load(&config, &stream, &path.join("joint_head.safetensors"))?;
    Ok(LoadedModel::Decision(Box::new(ClefRunner {
        tokenizer,
        network: Network::new(config, stream, backbone, head)?,
    })))
}

impl ClefRunner {
    fn decide_schema(&self, schema: &DecisionSchema) -> Result<DecisionOutput> {
        let encoded = encode::encode(&self.tokenizer, schema)?;
        let logits = self.network.logits(&encoded)?;
        self.network.stream().eval(&logits)?;
        let probabilities = logits
            .iter()
            .map(|l| l.cast(MlxDType::F32)?.softmax_with(-1, false))
            .collect::<Result<Vec<_>>>()?;
        let values = Array::concat(&probabilities.iter().collect::<Vec<_>>(), 0)?.to_f32_vec()?;
        let mut offset = 0;
        let mut answers = Vec::with_capacity(encoded.questions.len());
        for question in encoded.questions {
            let count = question.option_ids.len();
            let row = &values[offset..offset + count];
            offset += count;
            if row.iter().any(|p| !p.is_finite()) {
                return Err(Error::Other(format!(
                    "Clef produced non-finite probabilities for `{}`",
                    question.id
                )));
            }
            let mut selected_index = 0;
            for (i, &p) in row.iter().enumerate() {
                if p > row[selected_index] {
                    selected_index = i;
                }
            }
            answers.push(DecisionAnswer {
                question_id: Some(question.id),
                selected_index,
                scores: question
                    .option_ids
                    .into_iter()
                    .zip(row)
                    .map(|(code, &probability)| DecisionScore {
                        code,
                        token_id: None,
                        probability,
                    })
                    .collect(),
            });
        }
        Ok(DecisionOutput { answers })
    }
}

impl DecisionRuntime for ClefRunner {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        match input {
            DecisionInput::Schema(schema) => self.decide_schema(schema),
            DecisionInput::Encoded { .. } => Err(Error::Other(
                "Clef encodes its own prompt; pass a schema request".into(),
            )),
        }
    }

    fn code_names(&self) -> &[String] {
        &[]
    }

    fn token_ids(&self) -> &[u32] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&LoadOptions::default()).is_ok());
        for variant in ["auto", "reference"] {
            let options = LoadOptions {
                model_variant: Some(variant.into()),
                ..LoadOptions::default()
            };
            assert!(admit(&options).is_ok(), "{variant}");
        }
        for options in [
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                model_variant: Some("g20".into()),
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
        ] {
            assert!(admit(&options).is_err());
        }
    }
}
