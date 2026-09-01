//! Unified model loading for text, vision-language, and VLA models.

use std::path::{Path, PathBuf};

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::accelerator::create_backend;
use crate::builtin::register_builtin_models;
use crate::generation_config::{
    load_generation_options, GenerationConfigSource, GenerationOptions,
};
use crate::llm_trait::{
    GeneratedToken, GenerationOutput, GenerationRequest, LlmCapabilities, LlmInput, LlmTrait,
};
use crate::pi05::Pi05Config;
use crate::profiling::GenerationProfile;
use crate::registry;
use crate::vla::{Action, InferenceSpec, PreparedInference, VlaRequest, VlaRuntime};

/// User-level precision policy. Hardware/tactic dispatch remains in kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelPrecision {
    #[default]
    Auto,
    Bf16,
    Fp8,
    W8A8,
}

/// Request checkpoint-free random weights for a benchmark. Latency depends only
/// on shape and dtype, so no trained weights are needed to measure the engine.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticWeights {
    pub seed: u64,
}

/// Optional loading policy for models that have calibrated or tuned variants.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Registry name override. `None` detects the model from
    /// `config.json:model_type`.
    pub model_name: Option<String>,
    pub precision: ModelPrecision,
    /// Optional text-model weight dtype. `None` preserves checkpoint dtype
    /// (except CPU backends, which currently require f32).
    pub text_weight_dtype: Option<DType>,
    pub calibration_path: Option<PathBuf>,
    pub tuning_path: Option<PathBuf>,
    /// Enable online GEMM autotuning from real inference requests. When false,
    /// missing records resolve once to a safe inference fallback.
    pub autotune: bool,
    /// Explicit architecture config, overriding any on-disk `config.json`.
    pub config: Option<Pi05Config>,
    /// When set, load deterministic random weights instead of a checkpoint.
    pub synthetic: Option<SyntheticWeights>,
    /// Uniform FP8 activation scale, replacing a calibration file (synthetic use).
    pub uniform_fp8_scale: Option<f32>,
    /// Source for autoregressive text/VLM generation defaults.
    pub generation_config: GenerationConfigSource,
    /// Deployment-level settings applied over model defaults and under each
    /// request's explicit [`GenerationOptions`].
    pub generation_overrides: GenerationOptions,
}

/// A loaded autoregressive language model (text-only or VLM), or a VLA model.
/// Language generation and observation-to-action inference intentionally use
/// separate traits.
pub enum LoadedModel {
    Text {
        model: Box<dyn LlmTrait>,
        generation_defaults: GenerationOptions,
    },
    Vla(Box<dyn VlaRuntime>),
}

impl LoadedModel {
    pub fn text(model: Box<dyn LlmTrait>) -> Self {
        Self::Text {
            model,
            generation_defaults: GenerationOptions::default(),
        }
    }

    pub fn text_mut(&mut self) -> Result<&mut dyn LlmTrait> {
        match self {
            Self::Text { model, .. } => Ok(&mut **model),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    pub fn vla(&self) -> Result<&dyn VlaRuntime> {
        match self {
            Self::Vla(model) => Ok(&**model),
            Self::Text { .. } => Err(Error::Other("loaded model is text, not VLA".into())),
        }
    }

    pub fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.text_mut()?.forward(token_ids, start_pos)
    }

    pub fn text_capabilities(&self) -> Result<LlmCapabilities> {
        match self {
            Self::Text { model, .. } => Ok(model.capabilities()),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    /// Model plus deployment generation settings before request overrides.
    pub fn generation_defaults(&self) -> Result<&GenerationOptions> {
        match self {
            Self::Text {
                generation_defaults,
                ..
            } => Ok(generation_defaults),
            Self::Vla(_) => Err(Error::Other(
                "VLA models do not use autoregressive generation defaults".into(),
            )),
        }
    }

    /// Generate from the same request shape for text-only and VLM models.
    pub fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        mut on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        let options = GenerationOptions::greedy(max_new_tokens, eos_token_id);
        let output = self
            .generate_streaming_with_options(input, &options, |token| on_token(token.token_id))?;
        Ok((output.token_ids(), output.profile))
    }

    /// Sampling-aware generation for text and vision-language models.
    pub fn generate_streaming_with_options(
        &mut self,
        input: LlmInput<'_>,
        options: &GenerationOptions,
        mut on_token: impl FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        match self {
            Self::Text {
                model,
                generation_defaults,
            } => {
                let effective = GenerationOptions::apxinf_defaults()
                    .overlay(generation_defaults)
                    .overlay(options);
                model.generate_streaming_with_options_dyn(
                    GenerationRequest {
                        input,
                        options: &effective,
                    },
                    &mut on_token,
                )
            }
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.text_mut()?.reset();
        Ok(())
    }

    pub fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.vla()?.infer(request)
    }

    /// Run VLA inference and copy the action to host as `f32`. Convenience for
    /// callers that need host values without holding a backend handle.
    pub fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.vla()?.infer_host_f32(request)
    }

    pub fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.vla()?.prepare(spec)
    }
}

/// Stateless unified frontend. It creates one shared backend, loads weights,
/// and dispatches by model name plus device-specific registry suffix.
pub struct AutoModel;

impl AutoModel {
    /// Read a Hugging Face `model_type` and return its registry name.
    pub fn detect_model_name(path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|error| Error::Other(format!("read {}: {error}", config_path.display())))?;
        let config: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", config_path.display())))?;
        let model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::Other(format!(
                    "{} does not contain a string model_type",
                    config_path.display()
                ))
            })?;
        Ok(model_type.to_owned())
    }

    /// Load any supported model through one entry point.
    ///
    /// By default, the registry name is detected from
    /// `config.json:model_type`. Set [`LoadOptions::model_name`] only when a
    /// checkpoint needs an explicit registry-name override.
    pub fn load_model(
        device: Device,
        path: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<LoadedModel> {
        let path = path.as_ref();
        let detected_model_name;
        let model_name = match options.model_name.as_deref() {
            Some(model_name) => model_name,
            None => {
                detected_model_name = Self::detect_model_name(path)?;
                &detected_model_name
            }
        };

        register_builtin_models();
        let backend = create_backend(device)?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = crate::accelerator::cuda::downcast(&*backend) {
            configure_cuda_tuning(cuda, path, options)?;
        }
        let device_name = match device {
            Device::Cuda(_) => Some("cuda"),
            Device::Cpu => None,
        };

        let specific_name = device_name.map(|suffix| format!("{model_name}-{suffix}"));
        let factory = specific_name
            .as_deref()
            .and_then(registry::get)
            .or_else(|| registry::get(model_name))
            .ok_or_else(|| {
                Error::Other(format!(
                    "no model implementation for `{model_name}` on {device}"
                ))
            })?;
        let mut loaded = factory(path, device, backend, options)?;
        if let LoadedModel::Text {
            generation_defaults,
            ..
        } = &mut loaded
        {
            let model_defaults = load_generation_options(path, &options.generation_config)?;
            let defaults = model_defaults.overlay(&options.generation_overrides);
            // Reject malformed model/deployment values at load time rather
            // than waiting for the first generation request.
            defaults.resolve()?;
            *generation_defaults = defaults;
        }
        Ok(loaded)
    }
}

#[cfg(feature = "cuda")]
fn configure_cuda_tuning(
    cuda: &apxinf_cuda::CudaBackend,
    model_path: &Path,
    options: &LoadOptions,
) -> Result<()> {
    use apxinf_cuda::tuning::{TuningDb, TuningMode, TuningPaths};

    let default_paths = TuningPaths::for_cuda("configs/tuning", cuda.context().caps());
    let model_root = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };
    let legacy_path = model_root.join("tactics.json");
    let selected = select_cuda_tuning_database_path(&default_paths.tactics, &legacy_path, options);
    let database = selected
        .as_deref()
        .map(TuningDb::from_json_file)
        .transpose()?;
    let paths = options
        .tuning_path
        .clone()
        .map(TuningPaths::from_tactics)
        .unwrap_or(default_paths);
    let mode = if options.autotune {
        TuningMode::AutoTune
    } else {
        TuningMode::Inference
    };
    apxinf_cuda::kernels::gemm::configure_tuning(
        cuda.context(),
        mode,
        database.as_ref().map(std::slice::from_ref).unwrap_or(&[]),
        Some(paths),
    )
}

#[cfg(any(feature = "cuda", test))]
fn select_cuda_tuning_database_path(
    default_path: &Path,
    legacy_path: &Path,
    options: &LoadOptions,
) -> Option<PathBuf> {
    match options.tuning_path.as_ref() {
        Some(path) if path.is_file() => Some(path.clone()),
        Some(_) if options.autotune => None,
        Some(path) => Some(path.clone()),
        None => default_path
            .is_file()
            .then(|| default_path.to_path_buf())
            .or_else(|| {
                (options.synthetic.is_none() && legacy_path.is_file())
                    .then(|| legacy_path.to_path_buf())
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "apxinf-auto-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn synthetic_load_ignores_model_local_legacy_tactics() {
        let directory = temporary_directory("synthetic-tactics");
        let default_path = directory.join("missing-hardware-tactics.json");
        let legacy_path = directory.join("tactics.json");
        std::fs::write(&legacy_path, "{}").unwrap();
        let options = LoadOptions {
            synthetic: Some(SyntheticWeights { seed: 0 }),
            ..LoadOptions::default()
        };

        assert_eq!(
            select_cuda_tuning_database_path(&default_path, &legacy_path, &options),
            None
        );
        assert_eq!(
            select_cuda_tuning_database_path(&default_path, &legacy_path, &LoadOptions::default()),
            Some(legacy_path)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn synthetic_load_still_uses_hardware_or_explicit_tactics() {
        let directory = temporary_directory("explicit-tactics");
        let default_path = directory.join("hardware.json");
        let legacy_path = directory.join("tactics.json");
        let explicit_path = directory.join("explicit.json");
        std::fs::write(&default_path, "{}").unwrap();
        std::fs::write(&legacy_path, "{}").unwrap();
        std::fs::write(&explicit_path, "{}").unwrap();
        let mut options = LoadOptions {
            synthetic: Some(SyntheticWeights { seed: 0 }),
            ..LoadOptions::default()
        };

        assert_eq!(
            select_cuda_tuning_database_path(&default_path, &legacy_path, &options),
            Some(default_path.clone())
        );
        options.tuning_path = Some(explicit_path.clone());
        assert_eq!(
            select_cuda_tuning_database_path(&default_path, &legacy_path, &options),
            Some(explicit_path)
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
