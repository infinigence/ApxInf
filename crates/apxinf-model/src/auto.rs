//! Unified model loading for text, vision-language, and VLA models.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::accelerator::create_backend;
use crate::builtin::register_builtin_models;
use crate::d3::{DecisionInput, DecisionOutput, DecisionRuntime};
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
    /// Model-local compute implementation identifier. PI0.5 accepts
    /// auto, bf16, fp8_static and int8_dynamic. Native Metal Qwen3/MiniCPM5
    /// variants are parsed by their family loader. Other families migrate separately.
    pub model_variant: Option<String>,
    /// Legacy selection for model families not yet migrated to model_variant.
    pub precision: ModelPrecision,
    /// Optional text-model weight dtype. `None` preserves checkpoint dtype
    /// (except CPU backends, which currently require f32).
    pub text_weight_dtype: Option<DType>,
    pub calibration_path: Option<PathBuf>,
    /// For cuda-new, override the recipe directory. None uses the shared
    /// hardware/toolkit tuning directory; legacy JSON records are not loaded.
    pub tuning_path: Option<PathBuf>,
    /// Additional named model artifacts that are not embedded in the primary
    /// checkpoint. Model loaders must reject missing or unknown required
    /// assets instead of discovering them through process-global state.
    pub assets: BTreeMap<String, PathBuf>,
    /// Enable online operator autotuning (GEMM and, on cuda-new, Attention)
    /// on recipe misses during preparation or first execution. When false,
    /// missing records resolve once to a safe inference fallback.
    pub autotune: bool,
    /// Maximum number of fixed-length d3 text CUDA Graph profiles retained in
    /// memory. `None` uses the d3 runtime's safe default.
    pub d3_text_profile_limit: Option<usize>,
    /// Sum of token capacities across retained d3 text profiles. `None` uses
    /// the d3 runtime's safe default. A single request may exceed the budget.
    pub d3_text_profile_token_budget: Option<usize>,
    /// Serve fewer camera views than the checkpoint declares.
    pub num_views: Option<usize>,
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
    Decision(Box<dyn DecisionRuntime>),
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
            Self::Decision(_) => Err(Error::Other("loaded model is decision, not text".into())),
        }
    }

    pub fn vla(&self) -> Result<&dyn VlaRuntime> {
        match self {
            Self::Vla(model) => Ok(&**model),
            Self::Text { .. } => Err(Error::Other("loaded model is text, not VLA".into())),
            Self::Decision(_) => Err(Error::Other("loaded model is decision, not VLA".into())),
        }
    }

    pub fn decision_mut(&mut self) -> Result<&mut dyn DecisionRuntime> {
        match self {
            Self::Decision(model) => Ok(&mut **model),
            Self::Text { .. } => Err(Error::Other("loaded model is text, not decision".into())),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not decision".into())),
        }
    }

    pub fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        self.decision_mut()?.decide(input)
    }

    pub fn decision_code_names(&self) -> Result<&[String]> {
        match self {
            Self::Decision(model) => Ok(model.code_names()),
            Self::Text { .. } => Err(Error::Other("loaded model is text, not decision".into())),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not decision".into())),
        }
    }

    pub fn decision_token_ids(&self) -> Result<&[u32]> {
        match self {
            Self::Decision(model) => Ok(model.token_ids()),
            Self::Text { .. } => Err(Error::Other("loaded model is text, not decision".into())),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not decision".into())),
        }
    }

    pub fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.text_mut()?.forward(token_ids, start_pos)
    }

    pub fn text_capabilities(&self) -> Result<LlmCapabilities> {
        match self {
            Self::Text { model, .. } => Ok(model.capabilities()),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
            Self::Decision(model) => Ok(model.capabilities()),
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
            Self::Decision(_) => Err(Error::Other(
                "decision models do not use autoregressive generation defaults".into(),
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
            Self::Decision(_) => Err(Error::Other("loaded model is decision, not text".into())),
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

    /// Run a VLA inference with optional model-local phase timings.
    pub fn infer_host_f32_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, BTreeMap<String, f64>)> {
        self.vla()?.infer_host_f32_profiled(request)
    }

    /// Discrete action-token output shape, for autoregressive token VLAs.
    ///
    /// `None` means the loaded runtime emits continuous actions only.
    pub fn action_token_shape(&self) -> Result<Option<[usize; 2]>> {
        Ok(self.vla()?.action_token_shape())
    }

    /// Run VLA inference and return the raw discrete action tokens, for
    /// autoregressive token VLAs only. `stop_token` ends the decode early (see
    /// [`VlaRuntime::infer_action_tokens`]).
    pub fn infer_action_tokens(
        &self,
        request: &VlaRequest<'_>,
        stop_token: Option<u32>,
    ) -> Result<Tensor> {
        self.vla()?.infer_action_tokens(request, stop_token)
    }

    pub fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        self.vla()?.calibration_amax(request)
    }

    /// Calibration capture that a token decoder may end at `stop_token`.
    pub fn calibration_amax_stop(
        &self,
        request: &VlaRequest<'_>,
        stop_token: Option<u32>,
    ) -> Result<BTreeMap<String, f32>> {
        self.vla()?.calibration_amax_stop(request, stop_token)
    }

    pub fn calibration_plan(&self) -> Result<Vec<String>> {
        self.vla()?.calibration_plan()
    }

    pub fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.vla()?.prepare(spec)
    }

    pub fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: crate::ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.vla()?.prepare_with_policy(spec, policy)
    }

    pub fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: crate::ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.vla()?.prepare_for(sample, policy)
    }

    pub fn clear_prepared(&self) -> Result<()> {
        self.vla()?.clear_prepared()
    }
}

/// Stateless unified frontend. It creates one shared backend, loads weights,
/// and dispatches by model name with an explicit Metal prefix or CUDA suffix.
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
        // Decision 3.0 deliberately preserves the base Qwen3.5 model_type in
        // config.json. Its package-level format is the discriminator; without
        // this check it is incorrectly routed to the Qwen3.8 NVFP4 loader.
        let decision_path = model_dir.join("decision_config.json");
        if decision_path.is_file() {
            let raw = std::fs::read_to_string(&decision_path).map_err(|error| {
                Error::Other(format!("read {}: {error}", decision_path.display()))
            })?;
            let decision: serde_json::Value = serde_json::from_str(&raw).map_err(|error| {
                Error::Other(format!("parse {}: {error}", decision_path.display()))
            })?;
            if decision
                .get("format_id")
                .and_then(serde_json::Value::as_str)
                == Some("d3-code-readout-v1")
            {
                return Ok("d3".to_owned());
            }
        }
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

        if options.model_variant.is_some() && !supports_model_variant(model_name, device) {
            return Err(Error::Other(format!(
                "model {model_name} does not yet support model_variant"
            )));
        }

        if matches!(model_name, "pi05" | "pi05-cuda") {
            #[cfg(feature = "cuda")]
            return crate::pi05::load_with_cuda_new(path, device, options);

            #[cfg(not(feature = "cuda"))]
            return Err(Error::Other("PI0.5 requires CUDA support".into()));
        }

        if matches!(model_name, "d3" | "d3-cuda") {
            #[cfg(feature = "cuda")]
            return crate::d3::load::load_with_cuda_new(path, device, options);

            #[cfg(not(feature = "cuda"))]
            return Err(Error::Other("d3-mini requires CUDA support".into()));
        }

        register_builtin_models();
        let device_name = match device {
            Device::Cuda(_) => Some("cuda"),
            Device::Hip(_) => Some("hip"),
            Device::Metal(_) | Device::Cpu => None,
        };

        let specific_name = device_name.map(|suffix| format!("{model_name}-{suffix}"));
        let factory = if matches!(device, Device::Metal(_)) {
            // Only a Metal-owned factory can admit this device. Resolve once
            // during loading; model execution remains concrete native MLX.
            registry::get(&metal_registry_name(model_name))
                .ok_or(Error::UnsupportedDevice(device))?
        } else {
            specific_name
                .as_deref()
                .and_then(registry::get)
                .or_else(|| registry::get(model_name))
                .ok_or_else(|| {
                    Error::Other(format!(
                        "no model implementation for `{model_name}` on {device}"
                    ))
                })?
        };
        // Resolve admission before touching a device or loading checkpoint data.
        let backend = create_backend(device)?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = crate::accelerator::cuda::downcast(&*backend) {
            configure_cuda_tuning(cuda, path, options)?;
        }
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

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CudaRecipeOptions {
    pub cache_dir: Option<String>,
    pub online_tune: bool,
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn cuda_recipe_options(options: &LoadOptions) -> Result<CudaRecipeOptions> {
    let cache_dir = options
        .tuning_path
        .as_deref()
        .map(exact_recipe_directory)
        .map(|path| {
            path.to_str().map(str::to_owned).ok_or_else(|| {
                Error::Other(format!(
                    "exact recipe cache path is not valid UTF-8: {}",
                    path.display()
                ))
            })
        })
        .transpose()?;
    Ok(CudaRecipeOptions {
        cache_dir,
        online_tune: options.autotune,
    })
}

#[cfg(any(feature = "cuda", test))]
fn exact_recipe_directory(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.to_path_buf();
    }
    let legacy_json_name = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
    if path.is_file() || legacy_json_name {
        path.with_extension("recipes")
    } else {
        path.to_path_buf()
    }
}

// Canonical Metal registration is explicit and independent of future backend
// composition APIs. Preserve selectors used by the initial MLX preview.
fn metal_registry_name(model_name: &str) -> String {
    if model_name.starts_with("metal_") {
        model_name.to_owned()
    } else {
        format!(
            "metal_{}",
            model_name.strip_suffix("-mlx").unwrap_or(model_name)
        )
    }
}

// Admission only: supported variant names and precision semantics are owned by
// each family loader, not a second central implementation registry.
fn supports_model_variant(model_name: &str, device: Device) -> bool {
    matches!(
        model_name,
        "pi05"
            | "pi05-cuda"
            | "qwen_drive"
            | "qwen_drive-cuda"
            | "d3"
            | "d3-cuda"
            | "smolvla"
            | "smolvla_libero"
    ) || (matches!(device, Device::Metal(_))
        && matches!(
            model_name,
            "qwen3" | "metal_qwen3" | "qwen3-mlx" | "minicpm5" | "metal_minicpm5" | "minicpm5-mlx"
        ))
}

#[cfg(feature = "cuda")]
fn configure_cuda_tuning(
    cuda: &apxinf_cuda::CudaBackend,
    model_path: &Path,
    options: &LoadOptions,
) -> Result<()> {
    use apxinf_cuda::tuning::{TuningDb, TuningMode, TuningPaths};

    let default_paths = TuningPaths::resolve_for_cuda(
        "configs/tuning",
        cuda.context().caps(),
        cuda.context().library_versions(),
    );
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

    #[test]
    fn cuda_recipe_tuning_is_explicit_and_independent_of_directory() {
        let mut options = LoadOptions::default();
        let recipe = cuda_recipe_options(&options).unwrap();
        assert_eq!(recipe.cache_dir, None);
        assert!(!recipe.online_tune);
        options.autotune = true;
        assert!(cuda_recipe_options(&options).unwrap().online_tune);
        options.tuning_path = Some(PathBuf::from("custom-recipes"));
        let recipe = cuda_recipe_options(&options).unwrap();
        assert_eq!(recipe.cache_dir.as_deref(), Some("custom-recipes"));
        assert!(recipe.online_tune);
        options.autotune = false;
        assert!(!cuda_recipe_options(&options).unwrap().online_tune);
    }

    #[test]
    fn metal_rejects_legacy_registry_before_device_or_checkpoint_access() {
        fn legacy_factory(
            _path: &Path,
            _device: Device,
            _backend: std::sync::Arc<dyn apxinf_core::Backend>,
            _options: &LoadOptions,
        ) -> Result<LoadedModel> {
            Err(Error::Contract("legacy test factory reached"))
        }
        registry::register("test_metal_admission_legacy", legacy_factory);
        let path = Path::new("checkpoint-must-not-be-opened");
        let options = LoadOptions {
            model_name: Some("test_metal_admission_legacy".into()),
            ..LoadOptions::default()
        };
        // The existing CPU generic-registry route remains callable.
        assert!(matches!(
            AutoModel::load_model(Device::Cpu, path, &options),
            Err(Error::Contract("legacy test factory reached"))
        ));
        for name in ["test_metal_admission_legacy", "llama", "qwen3_vl"] {
            let options = LoadOptions {
                model_name: Some(name.into()),
                ..LoadOptions::default()
            };
            assert!(
                matches!(
                    AutoModel::load_model(Device::Metal(0), path, &options),
                    Err(Error::UnsupportedDevice(Device::Metal(0)))
                ),
                "accepted {name}"
            );
        }
    }

    #[test]
    fn metal_variant_admission_is_family_and_device_specific() {
        for name in [
            "qwen3",
            "metal_qwen3",
            "qwen3-mlx",
            "minicpm5",
            "metal_minicpm5",
            "minicpm5-mlx",
        ] {
            assert!(supports_model_variant(name, Device::Metal(0)));
            assert!(!supports_model_variant(name, Device::Cpu));
            assert!(!supports_model_variant(name, Device::Cuda(0)));
        }
        assert!(!supports_model_variant("llama", Device::Metal(0)));
        assert!(!supports_model_variant("qwen3_vl", Device::Metal(0)));
        assert!(supports_model_variant("pi05-cuda", Device::Cuda(0)));
        assert!(supports_model_variant("qwen_drive", Device::Cpu));
        assert!(supports_model_variant("smolvla", Device::Cuda(0)));
        assert!(supports_model_variant("smolvla_libero", Device::Cuda(0)));
    }

    #[test]
    fn metal_names_resolve_without_double_prefixes() {
        for family in ["qwen3", "minicpm5"] {
            let expected = format!("metal_{family}");
            assert_eq!(metal_registry_name(family), expected);
            assert_eq!(metal_registry_name(&expected), expected);
            assert_eq!(metal_registry_name(&format!("{family}-mlx")), expected);
            #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
            {
                register_builtin_models();
                assert!(registry::get(&expected).is_some());
            }
        }
        assert_eq!(metal_registry_name("llama"), "metal_llama");
    }

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

    #[test]
    fn d3_package_is_not_misdetected_as_generic_qwen3_5() {
        let directory = temporary_directory("d3-detection");
        std::fs::write(directory.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();
        std::fs::write(
            directory.join("decision_config.json"),
            r#"{"format_id":"d3-code-readout-v1"}"#,
        )
        .unwrap();

        assert_eq!(AutoModel::detect_model_name(&directory).unwrap(), "d3");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn non_d3_qwen3_5_detection_is_unchanged() {
        let directory = temporary_directory("qwen35-detection");
        std::fs::write(directory.join("config.json"), r#"{"model_type":"qwen3_5"}"#).unwrap();

        assert_eq!(AutoModel::detect_model_name(&directory).unwrap(), "qwen3_5");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
