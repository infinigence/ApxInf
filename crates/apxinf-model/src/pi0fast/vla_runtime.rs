//! Owning VLA frontend for the BF16 π0-FAST runtime.
//!
//! π0-FAST is the first autoregressive token VLA in ApxInf: the model emits
//! discrete action tokens, not continuous actions. This frontend therefore
//! exposes them through [`VlaRuntime::infer_action_tokens`]; the continuous
//! [`Action`] tensor the trait still requires carries the same ids encoded as
//! `f32` so generic consumers keep working, and the Python policy layer turns
//! them back into actions with the FAST tokenizer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, Observation, PreparedInference, VlaContract, VlaRequest,
    VisionObservation, VlaRuntime,
};

use super::backend::{
    kernels, DeviceBuffer as CudaBuffer, ImageLayout as KernelImageLayout, RuntimeBackend,
};
use super::fp8_calibration::{checkpoint_identity, Pi0FastFp8Calibration};
use super::{
    Pi0FastBf16Runtime, Pi0FastCalibrationPlan, Pi0FastConfig, Pi0FastFp8Runtime,
    Pi0FastFp8Scales, Pi0FastWeights, StaticBf16Pi0FastWeights, StaticFp8Pi0FastWeights,
};

/// Explicit opt-in for an uncalibrated FP8 bring-up, in the form
/// `APXINF_PI0FAST_FP8_ACTIVATION_SCALE=<positive float>`.
///
/// FP8 π0-FAST requires measured activation scales. Setting this variable is
/// the only way to run without a profile, and it is deliberately a variable a
/// caller has to set on purpose rather than a default that silently applies: a
/// uniform scale is known to break the greedy token stream (see
/// `devlocal/pi0-fast/reports/08-libero10-accuracy-and-fp8-diagnosis.md`), so it
/// must never be what someone gets by forgetting to calibrate.
const FP8_ACTIVATION_SCALE_ENV: &str = "APXINF_PI0FAST_FP8_ACTIVATION_SCALE";

/// Translate the public image layout into the kernel's own enum.
fn kernel_image_layout(layout: crate::vla::ImageLayout) -> KernelImageLayout {
    match layout {
        crate::vla::ImageLayout::Nhwc => KernelImageLayout::Nhwc,
        crate::vla::ImageLayout::Nchw => KernelImageLayout::Nchw,
    }
}

/// Checkpoint directory for a path that may point at the safetensors file.
fn artifact_root(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
    }
}

/// Both runtimes expose the same `infer` entry point; only the precision of the
/// projections differs, so the frontend dispatches once here rather than
/// duplicating the VLA plumbing.
enum RuntimeVariant {
    Bf16(Arc<Pi0FastBf16Runtime>),
    Fp8(Arc<Pi0FastFp8Runtime>),
}

impl RuntimeVariant {
    fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Bf16(runtime) => runtime.infer(patches, token_ids, token_count, stop_token),
            Self::Fp8(runtime) => runtime.infer(patches, token_ids, token_count, stop_token),
        }
    }

    /// Record BF16 activation maxima from one full inference.
    ///
    /// Only the BF16 runtime can produce a profile: an FP8 run would quantize
    /// each activation before the observer saw it, so the recorded maxima would
    /// describe the already-quantized distribution and the derived scales would
    /// ratchet further down on every regeneration.
    fn calibrate(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        stop_token: Option<u32>,
    ) -> Result<BTreeMap<String, f32>> {
        match self {
            Self::Bf16(runtime) => runtime.calibrate(patches, token_ids, token_count, stop_token),
            Self::Fp8(_) => Err(Error::Other(
                "π0-FAST FP8 calibration needs the BF16 runtime: load the checkpoint with \
                 precision=\"bf16\" (or model_variant bf16) to collect a profile"
                    .into(),
            )),
        }
    }
}

pub struct Pi0FastVlaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi0FastConfig>,
    runtime: RuntimeVariant,
}

impl Pi0FastVlaRuntime {
    fn patch_shape(&self) -> [usize; 2] {
        let config = &self.config;
        [
            config.patch_tokens(),
            3 * config.patch_size * config.patch_size,
        ]
    }

    fn device_tokens(&self, tokens: &[u32]) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_ne_bytes()).collect();
        let buffer = CudaBuffer::alloc(bytes.len(), self.backend.device_id())
            .map_err(Error::Cuda)?;
        buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(buffer)
    }

    /// Materialize the observation's camera input as device patches.
    ///
    /// π0-FAST's patch tensor is FP32 because PaliGemma runs the SigLIP patch
    /// embedding at FP32 precision; see [`super::VisionPatchEmbeddingF32`].
    fn device_patches(&self, observation: &Observation) -> Result<Tensor> {
        match &observation.vision {
            VisionObservation::Patches(tensor) => {
                if tensor.dtype() != DType::F32 {
                    return Err(Error::Other(format!(
                        "π0-FAST expects FP32 patches, got {}",
                        tensor.dtype()
                    )));
                }
                self.backend.to_device(tensor)
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                let config = &self.config;
                let patches =
                    self.backend
                        .to_device(&Tensor::zeros(self.patch_shape().to_vec(), DType::F32))?;
                let images = CudaBuffer::alloc(bytes.len(), self.backend.device_id())
                    .map_err(Error::Cuda)?;
                images.copy_from_host(bytes).map_err(Error::Cuda)?;
                kernels::preprocess::rgb_u8_to_patches_f32(
                    self.backend.context(),
                    &images,
                    &patches,
                    config.effective_views(),
                    config.image_size,
                    config.patch_size,
                    kernel_image_layout(*layout),
                )?;
                Ok(patches)
            }
        }
    }

    /// Validate one Observation and materialize its device patches and tokens.
    fn device_inputs(&self, observation: &Observation) -> Result<(Tensor, CudaBuffer)> {
        observation.validate()?;
        if observation.token_ids.len() > self.config.max_token_len {
            return Err(Error::Other(format!(
                "π0-FAST token count {} exceeds maximum {}",
                observation.token_ids.len(),
                self.config.max_token_len
            )));
        }
        let patches = self.device_patches(observation)?;
        let tokens = self.device_tokens(&observation.token_ids)?;
        Ok((patches, tokens))
    }

    fn run(&self, observation: &Observation, stop_token: Option<u32>) -> Result<Vec<u32>> {
        let (patches, tokens) = self.device_inputs(observation)?;
        self.runtime
            .infer(&patches, &tokens, observation.token_ids.len(), stop_token)
    }

    /// Collect one Observation's BF16 activation maxima for FP8 calibration.
    ///
    /// `stop_token` ends the decode exactly where inference would, so the
    /// captured decode-length distribution is the deployed one rather than the
    /// full `max_action_tokens` budget the policy never pays.
    fn calibrate_request(
        &self,
        observation: &Observation,
        stop_token: Option<u32>,
    ) -> Result<BTreeMap<String, f32>> {
        let (patches, tokens) = self.device_inputs(observation)?;
        self.runtime
            .calibrate(&patches, &tokens, observation.token_ids.len(), stop_token)
    }

    fn token_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let values: Vec<f32> = tokens.iter().map(|token| *token as f32).collect();
        Tensor::from_f32(vec![1, values.len()], &values)
    }
}

impl VlaRuntime for Pi0FastVlaRuntime {
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [1, self.config.max_action_tokens],
            patch_shape: self.patch_shape(),
            max_token_len: self.config.max_token_len,
            // The wire contract names the cameras the caller must supply; the
            // checkpoint's declared-empty views are padding the runtime drops.
            num_views: self.config.effective_views(),
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
            accepts_rgb_u8: true,
        }
    }

    fn action_token_shape(&self) -> Option<[usize; 2]> {
        Some([1, self.config.max_action_tokens])
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let tokens = self.run(request.observation, None)?;
        Ok(Action::new(self.token_tensor(&tokens)?))
    }

    fn infer_action_tokens(
        &self,
        request: &VlaRequest<'_>,
        stop_token: Option<u32>,
    ) -> Result<Tensor> {
        let tokens = self.run(request.observation, stop_token)?;
        self.token_tensor(&tokens)
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let tokens = self.run(request.observation, None)?;
        Ok(tokens.iter().map(|token| *token as f32).collect())
    }

    fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        self.calibrate_request(request.observation, None)
    }

    fn calibration_amax_stop(
        &self,
        request: &VlaRequest<'_>,
        stop_token: Option<u32>,
    ) -> Result<BTreeMap<String, f32>> {
        self.calibrate_request(request.observation, stop_token)
    }

    fn calibration_plan(&self) -> Result<Vec<String>> {
        Ok(Pi0FastCalibrationPlan::for_config(&self.config)
            .sites()
            .to_vec())
    }

    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "π0-FAST does not implement prepared/captured execution yet".into(),
        ))
    }
}

/// π0-FAST resolves `auto` to BF16 deliberately.
///
/// The FP8 path is a different numerical regime for an autoregressive argmax
/// decoder, and π0.5's rule (auto selects FP8 on SM100+) was validated on a flow
/// matcher. Until FP8 token-stream agreement is measured, the fast path is opted
/// into explicitly with `precision="fp8"`.
fn resolve_precision(requested: ModelPrecision) -> Result<ModelPrecision> {
    match requested {
        ModelPrecision::Auto | ModelPrecision::Bf16 => Ok(ModelPrecision::Bf16),
        ModelPrecision::Fp8 => Ok(ModelPrecision::Fp8),
        ModelPrecision::W8A8 => Err(Error::Other(
            "π0-FAST has no W8A8 path: the autoregressive decode is BF16 or FP8 E4M3".into(),
        )),
    }
}

/// The explicitly requested uncalibrated bring-up scale, if any.
fn fp8_activation_scale() -> Result<Option<f32>> {
    let raw = match std::env::var(FP8_ACTIVATION_SCALE_ENV) {
        Ok(raw) => raw,
        Err(_) => return Ok(None),
    };
    let scale = raw.parse::<f32>().map_err(|error| {
        Error::Other(format!("{FP8_ACTIVATION_SCALE_ENV} must be a float: {error}"))
    })?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!(
            "π0-FAST FP8 activation scale must be finite and positive, got {scale}"
        )));
    }
    Ok(Some(scale))
}

/// Resolve the FP8 activation scales for one checkpoint.
///
/// Order: an explicit `calibration=` path, then the checkpoint's own
/// `calibration.json`, then — only when the caller asked for it by setting
/// `APXINF_PI0FAST_FP8_ACTIVATION_SCALE` — a uniform bring-up scale. There is no
/// implicit fallback: a uniform scale does not preserve π0-FAST's greedy token
/// stream, so running FP8 without calibration has to be a decision rather than
/// an omission.
fn resolve_fp8_scales(
    config: &Pi0FastConfig,
    root: &Path,
    checkpoint_path: &Path,
    options: &LoadOptions,
) -> Result<Pi0FastFp8Scales> {
    let calibration_path = options.calibration_path.clone().or_else(|| {
        let candidate = root.join("calibration.json");
        candidate.is_file().then_some(candidate)
    });
    if let Some(calibration_path) = calibration_path {
        let checkpoint = checkpoint_identity(checkpoint_path)?;
        let calibration =
            Pi0FastFp8Calibration::from_json_file(&calibration_path, config, &checkpoint)?;
        eprintln!(
            "[apxinf] π0-FAST FP8 calibration={} ({} activation scales, data={})",
            calibration_path.display(),
            calibration.len(),
            checkpoint,
        );
        return Pi0FastFp8Scales::from_calibration(config, &calibration);
    }
    if let Some(scale) = fp8_activation_scale()? {
        eprintln!(
            "[apxinf] warning: π0-FAST FP8 is running on the uniform bring-up scale \
             {scale} from {FP8_ACTIVATION_SCALE_ENV}; this path does not preserve the \
             greedy token stream and is not a deployment configuration"
        );
        return Pi0FastFp8Scales::uniform(config, scale);
    }
    Err(Error::Other(format!(
        "π0-FAST FP8 requires measured activation scales: pass calibration=<profile.json>, \
         place calibration.json in {}, or set {FP8_ACTIVATION_SCALE_ENV}=<scale> to \
         explicitly request the uncalibrated bring-up path \
         (scripts/calibrate_pi0fast.py generates a profile)",
        root.display()
    )))
}

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("π0-FAST is only registered for CUDA".into()))?;
    if options.config.is_some() {
        return Err(Error::Other(
            "π0-FAST takes its shape from the checkpoint's config.json; \
             action_horizon/num_views overrides are not supported"
                .into(),
        ));
    }
    if options.synthetic.is_some() {
        return Err(Error::Other(
            "π0-FAST does not support synthetic weights yet".into(),
        ));
    }
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    let config = Arc::new(if config_path.is_file() {
        Pi0FastConfig::from_json_file(&config_path)?
    } else {
        Pi0FastConfig::default()
    });
    let host_weights = Pi0FastWeights::from_safetensors(&config, path)?;
    let runtime = match resolve_precision(options.precision)? {
        ModelPrecision::Fp8 => {
            let weights = Arc::new(StaticFp8Pi0FastWeights::from_host(
                &host_weights,
                &config,
                &*backend,
            )?);
            let scales = Arc::new(resolve_fp8_scales(&config, &root, path, options)?);
            RuntimeVariant::Fp8(Arc::new(Pi0FastFp8Runtime::new(
                Arc::clone(&backend),
                Arc::clone(&config),
                weights,
                scales,
            )?))
        }
        _ => {
            let weights = Arc::new(StaticBf16Pi0FastWeights::from_host(
                &host_weights,
                &config,
                &*backend,
            )?);
            RuntimeVariant::Bf16(Arc::new(Pi0FastBf16Runtime::new(
                Arc::clone(&backend),
                Arc::clone(&config),
                weights,
            )?))
        }
    };
    Ok(LoadedModel::Vla(Box::new(Pi0FastVlaRuntime {
        backend,
        config,
        runtime,
    })))
}
