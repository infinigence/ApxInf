//! Static-shape π0.5 inference orchestration for CUDA.

use std::sync::Arc;

pub use super::backend::ImageLayout as Pi05ImageLayout;
use apxinf_core::{Backend, Error, Graph, Result, Tensor};
use half::f16;

use super::backend::{kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use kernels::preprocess;

use super::{sinusoidal_time_embedding, Pi05Config, StaticFp8Pi05Weights};

pub use super::static_weights::Pi05ActivationScales;

pub use super::blocks::PrefixKvCache;
use super::blocks::{Fp8Blocks, Pi05StepStyles};
use super::network::Pi05Fp8Network;

/// A fixed-address, replayable full π0.5 inference graph.
///
/// Input tensor contents may be updated in place between replays, but their
/// addresses and `token_count` must remain unchanged. Shared owners keep the
/// runtime weights, backend, and all fixed-address inputs alive.
pub struct Pi05CapturedGraph {
    // Drop the executable before any memory it references.
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    _styles: Vec<Pi05StepStyles>,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    _network: Arc<Pi05Fp8Network>,
    workspace: kernels::GraphWorkspace,
}

impl Pi05CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn replay_and_synchronize(&self) -> Result<()> {
        self.graph.replay()?;
        self.backend.synchronize()
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    /// Raw-image layout captured into the graph, or `None` for the legacy
    /// normalized FP16 patch input path.
    pub fn raw_image_layout(&self) -> Option<Pi05ImageLayout> {
        self.raw_image_layout
    }

    fn update_tokens(&self, token_ids: &[u32]) -> Result<()> {
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    /// Replace captured inputs while preserving every device address.
    /// `patches` and `noise` must be CPU tensors with the captured shapes.
    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    /// Replace non-random captured inputs while retaining the existing device
    /// latent. Used when a bound device generator fills `noise` in place.
    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "π0.5 graph was captured for raw RGB input; use update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        // Do not overwrite an input while a preceding replay still reads it.
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    /// Replace a captured raw RGB `uint8` batch and the shared prompt/noise
    /// inputs without changing any graph-visible device address.
    pub fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<()> {
        self.update_raw_image_inputs_without_noise(images, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_raw_image_inputs_without_noise(
        &self,
        images: &[u8],
        token_ids: &[u32],
    ) -> Result<()> {
        let raw_images = self.raw_images.as_ref().ok_or_else(|| {
            Error::Other("π0.5 graph was captured for FP16 patches; use update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        self.backend.synchronize()?;
        raw_images.copy_from_host(images).map_err(Error::Cuda)?;
        self.update_tokens(token_ids)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace.capacity()
    }

    pub fn workspace_used_bytes(&self) -> usize {
        self.workspace.used()
    }
}

#[derive(Clone)]
pub struct Pi05CudaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    network: Arc<Pi05Fp8Network>,
    scales: Arc<Pi05ActivationScales>,
}

impl Pi05CudaRuntime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticFp8Pi05Weights>,
        scales: Arc<Pi05ActivationScales>,
    ) -> Result<Self> {
        let network = Arc::new(Pi05Fp8Network::from_blocks(Fp8Blocks::new(
            Arc::clone(&backend),
            Arc::clone(&config),
            weights,
            Arc::clone(&scales),
        )?));
        Ok(Self {
            backend,
            config,
            network,
            scales,
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    /// Input patches are already normalized and flattened as
    /// `[views*patches_per_view, 3*patch_size*patch_size]` FP16.
    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        self.network.encode_vision(patches)
    }

    pub fn embed_prefix(
        &self,
        vision_tokens: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
    ) -> Result<Tensor> {
        self.network
            .embed_prefix(vision_tokens, token_ids, token_count)
    }

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<PrefixKvCache> {
        self.network.prefix_forward(prefix)
    }

    fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<Pi05StepStyles>> {
        self.network.prepare_all_styles(time_embeddings)
    }

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        self.network.denoise_step(state, time_embedding, prefix, dt)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &PrefixKvCache,
    ) -> Result<Tensor> {
        self.network
            .denoise_all_steps(noise, time_embeddings, prefix)
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        self.network
            .infer(patches, token_ids, token_count, noise, time_embeddings)
    }

    /// Run eager inference when RGB preprocessing has already produced
    /// calibrated E4M3 patch tokens.
    pub fn infer_fp8_patches(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        self.network
            .infer_native(patches, token_ids, token_count, noise, time_embeddings)
    }

    fn infer_with_styles(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        self.network
            .infer_with_styles(patches, token_ids, token_count, noise, styles)
    }

    fn infer_with_styles_fp8_patches(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        self.network
            .infer_with_native_styles(patches, token_ids, token_count, noise, styles)
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_captured_inputs(
        &self,
        patches: &Tensor,
        raw_images: Option<&CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => self.infer_with_styles(patches, token_ids, token_count, noise, styles),
            (Some(images), Some(layout)) => {
                preprocess::rgb_u8_to_patches_e4m3(
                    self.ctx(),
                    images,
                    patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    layout,
                    self.scales.vision_patch_input,
                )?;
                self.infer_with_styles_fp8_patches(patches, token_ids, token_count, noise, styles)
            }
            _ => Err(Error::Other(
                "π0.5 raw image buffer/layout capture state is inconsistent".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_infer_impl(
        &self,
        patches: Tensor,
        raw_images: Option<CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 raw image buffer/layout capture state is inconsistent".into(),
            ));
        }
        // Timestep embeddings are constant for the reverse-flow schedule, so
        // all AdaRMS projections can be excluded from steady-state replay.
        let styles = self.prepare_all_styles(time_embeddings)?;
        backend.synchronize()?;
        let (max_activation_elements, max_weight_elements) =
            self.config.fp8_emulation_scratch_elements(token_count)?;
        let workspace = kernels::GraphWorkspace::new_fp8(
            self.config.cuda_graph_workspace_bytes(token_count)?,
            max_activation_elements,
            max_weight_elements,
            self.ctx().device_id(),
        )?;

        // Fail shape, calibration, and workspace checks before beginning a
        // stream capture, where recovery from a rejected launch is harder.
        // Online tuning may publish several exact winners during the first
        // eager traversal. Run one more traversal after the generation stops
        // changing so every cached plan is prepared from the final snapshot
        // before CUDA begins capture.
        let mut stable = false;
        for _ in 0..4 {
            let generation = self.ctx().tuning().generation();
            let eager_output = kernels::prepare_with_workspace(&workspace, || {
                self.infer_captured_inputs(
                    &patches,
                    raw_images.as_ref(),
                    raw_image_layout,
                    token_ids,
                    token_count,
                    noise,
                    &styles,
                )
            })?;
            backend.synchronize()?;
            drop(eager_output);
            if self.ctx().tuning().generation() == generation {
                stable = true;
                break;
            }
        }
        if !stable {
            return Err(Error::Other(
                "GEMM tactic store did not stabilize before PI0.5 graph capture".into(),
            ));
        }

        let (graph, output) = backend.capture_graph(|| {
            kernels::with_workspace(&workspace, || {
                self.infer_captured_inputs(
                    &patches,
                    raw_images.as_ref(),
                    raw_image_layout,
                    token_ids,
                    token_count,
                    noise,
                    &styles,
                )
            })
        })?;
        Ok(Pi05CapturedGraph {
            graph,
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            _styles: styles,
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _network: Arc::clone(&self.network),
            workspace,
        })
    }

    /// Validate once eagerly, then capture the complete fixed-shape inference
    /// schedule into a CUDA graph backed by a persistent device arena.
    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05CapturedGraph> {
        self.capture_infer_impl(
            patches.clone(),
            None,
            None,
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }

    /// Capture full inference beginning with already resized RGB `uint8`
    /// images. The graph owns a stable raw-image device buffer, and its first
    /// node performs fused normalization, patchification, and E4M3
    /// quantization. Call `update_raw_image_inputs` before each replay.
    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes = self
            .config
            .num_views
            .checked_mul(3)
            .and_then(|value| value.checked_mul(self.config.image_size))
            .and_then(|value| value.checked_mul(self.config.image_size))
            .ok_or_else(|| Error::Other("π0.5 raw image size overflow".into()))?;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches = backend.to_device(&Tensor::zeros(
            vec![patch_rows, patch_width],
            apxinf_core::DType::F8E4M3,
        ))?;
        self.capture_infer_impl(
            patches,
            Some(raw_images),
            Some(layout),
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }
}

/// Precompute the fixed reverse-flow timesteps before CUDA graph capture.
pub fn upload_time_embeddings(config: &Pi05Config, backend: &dyn Backend) -> Result<Vec<Tensor>> {
    (0..config.num_flow_steps)
        .map(|step| {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(f16::from_f32)
            .collect::<Vec<_>>();
            let tensor = Tensor::from_f16(vec![1, config.action_expert.width], &values)?;
            backend.to_device(&tensor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_scales_match_model_depths() {
        let config = Pi05Config::thor_two_view();
        let scales = Pi05ActivationScales::uniform(&config, 0.01).unwrap();
        assert_eq!(scales.vision_layers.len(), 27);
        assert_eq!(scales.language_layers.len(), 18);
        assert_eq!(scales.action_layers.len(), 18);
    }
}
