//! Owning VLA frontend for the BF16 π0-FAST runtime.
//!
//! π0-FAST is the first autoregressive token VLA in ApxInf: the model emits
//! discrete action tokens, not continuous actions. This frontend therefore
//! exposes them through [`VlaRuntime::infer_action_tokens`]; the continuous
//! [`Action`] tensor the trait still requires carries the same ids encoded as
//! `f32` so generic consumers keep working, and the Python policy layer turns
//! them back into actions with the FAST tokenizer.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel};
use crate::vla::{
    Action, InferenceSpec, Observation, PreparedInference, VlaContract, VlaRequest,
    VisionObservation, VlaRuntime,
};

use super::backend::{
    kernels, DeviceBuffer as CudaBuffer, ImageLayout as KernelImageLayout, RuntimeBackend,
};
use super::{
    Pi0FastBf16Runtime, Pi0FastConfig, Pi0FastWeights, StaticBf16Pi0FastWeights,
};

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

pub struct Pi0FastVlaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi0FastConfig>,
    runtime: Arc<Pi0FastBf16Runtime>,
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

    fn run(&self, observation: &Observation, stop_token: Option<u32>) -> Result<Vec<u32>> {
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
        self.runtime.infer(&patches, &tokens, observation.token_ids.len(), stop_token)
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

    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "π0-FAST does not implement prepared/captured execution yet".into(),
        ))
    }
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
    let weights = Arc::new(StaticBf16Pi0FastWeights::from_host(
        &host_weights,
        &*backend,
    )?);
    let runtime = Arc::new(Pi0FastBf16Runtime::new(
        Arc::clone(&backend),
        Arc::clone(&config),
        weights,
    )?);
    Ok(LoadedModel::Vla(Box::new(Pi0FastVlaRuntime {
        backend,
        config,
        runtime,
    })))
}
