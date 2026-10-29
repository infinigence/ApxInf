//! Native Clef Flash on Apple Metal: Qwen3.5 hybrid backbone (affine Q4) in
//! the stock MLX-VLM composition or the optimized GPU composition, and the
//! BF16 joint schema head. No Python model execution.
mod config;
mod encode;
mod model;
mod weights;

pub use config::{Config, MAX_INPUT_TOKENS};

use crate::decision::{
    DecisionInput, DecisionOutput, DecisionRuntime, DecisionSchema, SchemaDecisionAnswer,
    SchemaDecisionOutput, SchemaDecisionScore,
};
use crate::llm_trait::LlmCapabilities;
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use apxinf_tokenizer::Tokenizer;
use model::{Composition, Network};
use std::path::Path;
use std::sync::Arc;

pub struct ClefRunner {
    tokenizer: Tokenizer,
    network: Network,
}

/// MLX architectures where the optimized GPU composition passed its quality
/// and paired latency gates: the M1 Pro (2026-10-17). `auto` uses the
/// reference composition everywhere else.
const GPU_QUALIFIED_ARCHITECTURES: &[&str] = &["applegpu_g13s"];

/// Device-independent option admission, checked before the checkpoint is read.
/// Returns the requested composition; `None` is `auto`, which picks the
/// composition qualified on the device.
fn admit(options: &LoadOptions) -> Result<Option<Composition>> {
    let requested = match options.model_variant.as_deref() {
        None | Some("auto") => None,
        Some("reference") => Some(Composition::Reference),
        Some(_) => {
            return Err(Error::Contract(
                "Clef MLX supports the auto and reference variants",
            ))
        }
    };
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
    Ok(requested)
}

/// `auto` runs an optimized composition only on an architecture where it
/// qualified.
fn select(requested: Option<Composition>, architecture: &str) -> Composition {
    requested.unwrap_or(if GPU_QUALIFIED_ARCHITECTURES.contains(&architecture) {
        Composition::Gpu
    } else {
        Composition::Reference
    })
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
    let requested = admit(options)?;
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
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    let composition = select(requested, architecture);
    if requested.is_none() && composition == Composition::Reference {
        eprintln!(
            "Clef: no optimized composition is qualified on {architecture:?}; using the reference composition"
        );
    }
    let backbone = weights::Backbone::load(&config, &stream, path)?;
    let head = weights::Head::load(&config, &stream, &path.join("joint_head.safetensors"))?;
    Ok(LoadedModel::Decision(Box::new(ClefRunner {
        tokenizer,
        network: Network::new(config, stream, backbone, head, composition)?,
    })))
}

impl ClefRunner {
    fn score_schema(&self, schema: &DecisionSchema) -> Result<SchemaDecisionOutput> {
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
            answers.push(SchemaDecisionAnswer {
                question_id: question.id,
                selected_index,
                scores: question
                    .option_ids
                    .into_iter()
                    .zip(row)
                    .map(|(code, &probability)| SchemaDecisionScore { code, probability })
                    .collect(),
            });
        }
        Ok(SchemaDecisionOutput { answers })
    }
}

impl DecisionRuntime for ClefRunner {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn decide(&mut self, _input: DecisionInput<'_>) -> Result<DecisionOutput> {
        Err(Error::Other(
            "Clef encodes its own prompt; pass a schema request".into(),
        ))
    }

    fn decide_schema(&mut self, schema: &DecisionSchema) -> Result<SchemaDecisionOutput> {
        self.score_schema(schema)
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
        assert_eq!(admit(&LoadOptions::default()).unwrap(), None);
        for (variant, composition) in [("auto", None), ("reference", Some(Composition::Reference))]
        {
            let options = LoadOptions {
                model_variant: Some(variant.into()),
                ..LoadOptions::default()
            };
            assert_eq!(admit(&options).unwrap(), composition, "{variant}");
        }
        for options in [
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                model_variant: Some("fast".into()),
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

    #[test]
    fn auto_runs_the_gpu_composition_only_on_qualified_architectures() {
        assert_eq!(select(None, "applegpu_g13s"), Composition::Gpu);
        for other in ["applegpu_g16g", "applegpu_g14g", "applegpu_g13g", ""] {
            assert_eq!(select(None, other), Composition::Reference, "{other}");
        }
        for architecture in ["applegpu_g13s", "applegpu_g16g"] {
            assert_eq!(
                select(Some(Composition::Reference), architecture),
                Composition::Reference
            );
        }
    }
}
