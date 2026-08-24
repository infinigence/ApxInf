//! Layered generation settings and Hugging Face generation-config loading.

use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result, RngKey, TokenPenalties, TokenSamplingParams, TokenSelection};
use serde::Deserialize;

/// Explicit selection policy for categorical token generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    Greedy,
    Random,
}

/// Public, partially specified generation settings.
///
/// The same type represents model defaults, deployment overrides, and request
/// overrides. `None` means that the current layer does not set that field. A
/// higher-priority layer is applied with [`Self::overlay`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationOptions {
    pub max_new_tokens: Option<usize>,
    pub eos_token_ids: Option<Vec<u32>>,
    pub sampling_mode: Option<SamplingMode>,
    pub temperature: Option<f32>,
    /// Positive values enable top-k. Zero and negative values disable it.
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub return_logprob: Option<bool>,
}

/// Complete, normalized execution settings. This is deliberately crate-local:
/// callers submit [`GenerationOptions`], while the runtime consumes this type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedGenerationOptions {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub sampling: TokenSamplingParams,
    pub rng: RngKey,
}

impl GenerationOptions {
    pub const DEFAULT_MAX_NEW_TOKENS: usize = 50;

    /// ApxInf's compatibility defaults when neither the model nor the caller
    /// supplies a value. Historical generation remains greedy.
    pub fn apxinf_defaults() -> Self {
        Self {
            max_new_tokens: Some(Self::DEFAULT_MAX_NEW_TOKENS),
            eos_token_ids: Some(Vec::new()),
            sampling_mode: Some(SamplingMode::Greedy),
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            seed: Some(0),
            return_logprob: Some(false),
        }
    }

    /// Preserve the historical explicit greedy API, including neutral
    /// penalties, regardless of model-provided sampling defaults.
    pub fn greedy(max_new_tokens: usize, eos_token_id: Option<u32>) -> Self {
        Self {
            max_new_tokens: Some(max_new_tokens),
            eos_token_ids: Some(eos_token_id.into_iter().collect()),
            sampling_mode: Some(SamplingMode::Greedy),
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            seed: Some(0),
            return_logprob: Some(false),
        }
    }

    /// Apply a higher-priority settings layer. Only fields explicitly present
    /// in `higher` replace the current layer.
    pub fn overlay(mut self, higher: &Self) -> Self {
        macro_rules! replace_some {
            ($field:ident) => {
                if higher.$field.is_some() {
                    self.$field = higher.$field.clone();
                }
            };
        }

        replace_some!(max_new_tokens);
        replace_some!(eos_token_ids);
        replace_some!(temperature);
        replace_some!(top_k);
        replace_some!(top_p);
        replace_some!(repetition_penalty);
        replace_some!(frequency_penalty);
        replace_some!(presence_penalty);
        replace_some!(seed);
        replace_some!(return_logprob);

        // An explicit mode always wins. Otherwise, an explicitly supplied
        // temperature/filter implies a sampling choice for that layer.
        self.sampling_mode = higher
            .sampling_mode
            .or_else(|| higher.inferred_sampling_mode())
            .or(self.sampling_mode);
        self
    }

    fn inferred_sampling_mode(&self) -> Option<SamplingMode> {
        if let Some(temperature) = self.temperature {
            return Some(if temperature == 0.0 {
                SamplingMode::Greedy
            } else {
                SamplingMode::Random
            });
        }
        (self.top_k.is_some() || self.top_p.is_some()).then_some(SamplingMode::Random)
    }

    /// Parse the subset of Hugging Face `generation_config.json` that ApxInf
    /// currently implements. Unknown fields are ignored.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let config: HuggingFaceGenerationConfig = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("parse generation config: {error}")))?;
        Ok(config.into_options())
    }

    pub(crate) fn resolve(&self) -> Result<ResolvedGenerationOptions> {
        let complete = Self::apxinf_defaults().overlay(self);
        let max_new_tokens = complete.max_new_tokens.expect("framework default");
        let eos_token_ids = complete.eos_token_ids.expect("framework default");
        let mode = complete.sampling_mode.expect("framework default");
        let temperature = complete.temperature.expect("framework default");
        let top_p = complete.top_p.expect("framework default");
        let raw_top_k = complete.top_k.expect("framework default");
        let repetition = complete.repetition_penalty.expect("framework default");
        let frequency = complete.frequency_penalty.expect("framework default");
        let presence = complete.presence_penalty.expect("framework default");

        if !temperature.is_finite() || temperature < 0.0 {
            return Err(Error::Other(
                "generation temperature must be finite and non-negative".into(),
            ));
        }
        if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
            return Err(Error::Other(
                "generation top-p must be finite and in the interval (0, 1]".into(),
            ));
        }
        if !repetition.is_finite() || repetition <= 0.0 {
            return Err(Error::Other(
                "generation repetition penalty must be finite and greater than zero".into(),
            ));
        }
        if !frequency.is_finite() || !presence.is_finite() {
            return Err(Error::Other(
                "generation frequency and presence penalties must be finite".into(),
            ));
        }

        let top_k =
            if raw_top_k <= 0 {
                None
            } else {
                Some(usize::try_from(raw_top_k).map_err(|_| {
                    Error::Other(format!("generation top-k is too large: {raw_top_k}"))
                })?)
            };
        let selection = if mode == SamplingMode::Greedy || temperature == 0.0 {
            TokenSelection::Greedy
        } else {
            TokenSelection::Random {
                temperature,
                top_k,
                top_p,
            }
        };

        Ok(ResolvedGenerationOptions {
            max_new_tokens,
            eos_token_ids,
            sampling: TokenSamplingParams {
                selection,
                penalties: TokenPenalties {
                    repetition,
                    frequency,
                    presence,
                },
                return_logprob: complete.return_logprob.expect("framework default"),
            },
            rng: RngKey::new(complete.seed.expect("framework default"), 0, 0),
        })
    }
}

/// Where a model load obtains its default generation settings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GenerationConfigSource {
    /// Read `generation_config.json` from the loaded model directory when it
    /// exists. A missing file is not an error.
    #[default]
    Auto,
    /// Ignore model-provided generation settings and use ApxInf defaults.
    ApxInf,
    /// Read a specific JSON file or a directory containing the file.
    Path(PathBuf),
}

impl GenerationConfigSource {
    pub fn from_cli_value(value: &str) -> Self {
        match value {
            "auto" => Self::Auto,
            "apxinf" => Self::ApxInf,
            path => Self::Path(PathBuf::from(path)),
        }
    }
}

pub(crate) fn load_generation_options(
    model_path: &Path,
    source: &GenerationConfigSource,
) -> Result<GenerationOptions> {
    let optional = matches!(source, GenerationConfigSource::Auto);
    let config_path = match source {
        GenerationConfigSource::ApxInf => return Ok(GenerationOptions::default()),
        GenerationConfigSource::Auto => artifact_root(model_path).join("generation_config.json"),
        GenerationConfigSource::Path(path) if path.is_dir() => path.join("generation_config.json"),
        GenerationConfigSource::Path(path) => path.clone(),
    };

    if optional && !config_path.is_file() {
        return Ok(GenerationOptions::default());
    }
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|error| Error::Other(format!("read {}: {error}", config_path.display())))?;
    GenerationOptions::from_json_str(&raw)
        .map_err(|error| Error::Other(format!("{}: {error}", config_path.display())))
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

#[derive(Debug, Deserialize)]
struct HuggingFaceGenerationConfig {
    max_new_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<i64>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    do_sample: Option<bool>,
    eos_token_id: Option<OneOrManyTokenIds>,
}

impl HuggingFaceGenerationConfig {
    fn into_options(self) -> GenerationOptions {
        let sampling_mode = self.do_sample.map(|enabled| {
            if enabled {
                SamplingMode::Random
            } else {
                SamplingMode::Greedy
            }
        });
        GenerationOptions {
            max_new_tokens: self.max_new_tokens,
            eos_token_ids: self.eos_token_id.map(OneOrManyTokenIds::into_vec),
            sampling_mode,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: None,
            return_logprob: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_config_parses_scalar_and_list_eos() {
        let scalar = GenerationOptions::from_json_str(
            r#"{"max_new_tokens":64,"temperature":0.7,"top_k":40,"top_p":0.9,"do_sample":true,"eos_token_id":2}"#,
        )
        .unwrap();
        assert_eq!(scalar.max_new_tokens, Some(64));
        assert_eq!(scalar.sampling_mode, Some(SamplingMode::Random));
        assert_eq!(scalar.eos_token_ids, Some(vec![2]));

        let list = GenerationOptions::from_json_str(r#"{"eos_token_id":[2,3]}"#).unwrap();
        assert_eq!(list.eos_token_ids, Some(vec![2, 3]));
    }

    #[test]
    fn higher_layers_override_only_present_fields() {
        let model = GenerationOptions::from_json_str(
            r#"{"max_new_tokens":64,"temperature":0.7,"top_p":0.8,"do_sample":true}"#,
        )
        .unwrap();
        let request = GenerationOptions {
            temperature: Some(0.0),
            ..GenerationOptions::default()
        };
        let resolved = model.overlay(&request).resolve().unwrap();

        assert_eq!(resolved.max_new_tokens, 64);
        assert_eq!(resolved.sampling.selection, TokenSelection::Greedy);
    }

    #[test]
    fn non_positive_top_k_disables_filtering() {
        let options = GenerationOptions {
            sampling_mode: Some(SamplingMode::Random),
            top_k: Some(-1),
            ..GenerationOptions::default()
        };
        let resolved = options.resolve().unwrap();
        assert!(matches!(
            resolved.sampling.selection,
            TokenSelection::Random { top_k: None, .. }
        ));
    }

    #[test]
    fn auto_source_tolerates_a_missing_file() {
        let missing = std::env::temp_dir().join(format!(
            "apxinf-generation-config-missing-{}",
            std::process::id()
        ));
        let options = load_generation_options(&missing, &GenerationConfigSource::Auto).unwrap();
        assert_eq!(options, GenerationOptions::default());
    }

    #[test]
    fn explicit_directory_source_reads_generation_config() {
        let dir = std::env::temp_dir().join(format!(
            "apxinf-generation-config-explicit-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"max_new_tokens":17,"do_sample":false}"#,
        )
        .unwrap();

        let options = load_generation_options(
            Path::new("unused-model-path"),
            &GenerationConfigSource::Path(dir.clone()),
        )
        .unwrap();
        assert_eq!(options.max_new_tokens, Some(17));
        assert_eq!(options.sampling_mode, Some(SamplingMode::Greedy));

        std::fs::remove_dir_all(dir).unwrap();
    }
}
